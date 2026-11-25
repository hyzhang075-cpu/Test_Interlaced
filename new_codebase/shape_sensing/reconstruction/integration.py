"""纯数组曲率到三维形状的积分实现。"""
import numpy as np


_KAPPA_EPS = 1e-12


def _homogeneous(kappa, theta_abs, ds):
    """构造单段 4×4 齐次变换矩阵（论文 A 矩阵，来源：新算法参考/ht_algorithm.py）。

    其中 θ 为相对当前局部 x 轴的方位角，φ = kappa·ds，R = 1/kappa；
    κ≈0 的直线段退化为沿局部 z 轴平移 ds。

    Args:
        kappa: 本段曲率大小，单位 1/m。
        theta_abs: 本段曲率方向角，单位 rad。
        ds: 本段弧长，单位 m。
    Returns:
        A: (4, 4) 齐次矩阵；若 ds 极短返回单位矩阵。
    计算:
        转弯角 phi = kappa * ds，半径 R = 1/kappa；曲率为零时 A 表示
        沿局部 z 轴平移 ds。位移项用 sinc 形式避免小角度相减造成精度损失。
    """
    A = np.eye(4)
    if ds <= 1e-15:
        return A

    if kappa <= _KAPPA_EPS:
        # 直段：无旋转，沿 z 平移 ds
        A[:3, 3] = [0.0, 0.0, ds]
        return A

    phi = kappa * ds

    cth, sth = np.cos(theta_abs), np.sin(theta_abs)
    cphi, sphi = np.cos(phi), np.sin(phi)

    # ---- 旋转块：Rz(θ) · Ry(φ) ----
    A[0, 0] = cth * cphi
    A[0, 1] = -sth
    A[0, 2] = cth * sphi

    A[1, 0] = sth * cphi
    A[1, 1] = cth
    A[1, 2] = sth * sphi

    A[2, 0] = -sphi
    A[2, 1] = 0.0
    A[2, 2] = cphi

    # ---- 平移列：R(1-cosφ) 与 R sinφ 的数值稳定形式 ----
    half = phi / 2.0
    lateral = ds * np.sin(half) * np.sinc(half / np.pi)  # = R(1 - cosφ)
    z_comp = ds * np.sinc(phi / np.pi)                   # = R sinφ

    A[0, 3] = lateral * cth
    A[1, 3] = lateral * sth
    A[2, 3] = z_comp

    return A



class ShapeIntegrationMixin:
    """复用重构器原有标架积分方法，不构造传感器或读取数据。"""

    def _reconstruct_from_interpolated_curvature(
        self, cur_interp, ang_interp, interp_dist, init_T
    ):
        """Build the curve and local frame from the exact interpolated curvature used."""
        cur_interp = np.asarray(cur_interp, dtype=float)
        ang_interp = np.asarray(ang_interp, dtype=float)
        interp_dist = np.asarray(interp_dist, dtype=float)
        init_T = np.asarray(init_T, dtype=float)
        if (
            cur_interp.ndim != 1
            or ang_interp.ndim != 1
            or interp_dist.ndim != 1
            or cur_interp.shape != ang_interp.shape
            or cur_interp.size != interp_dist.size
            or cur_interp.size < 2
            or init_T.shape != (3,)
            or not np.all(np.isfinite(cur_interp))
            or not np.all(np.isfinite(ang_interp))
            or not np.all(np.isfinite(interp_dist))
            or not np.all(np.isfinite(init_T))
            or not np.all(np.diff(interp_dist) > 0.0)
        ):
            raise ValueError("interpolated curvature inputs must be finite, ordered, and shape-compatible")

        interp_ss = np.diff(interp_dist)
        init_N = np.array([0.0, np.cos(ang_interp[0]), np.sin(ang_interp[0])])
        if self.frame_rule == 'TNB':
            ang_diff = np.diff(ang_interp)
            torsion = (ang_diff - 2 * np.pi * np.floor((ang_diff + np.pi) / (2 * np.pi))) / interp_ss
            return self._F_S_frame(
                kappa=cur_interp,
                tau=torsion,
                ss=interp_ss,
                init_T=init_T,
                init_N=init_N,
            )
        if self.frame_rule == 'TNN':
            r, T, Y, Z, _, _ = self._parallel_frame(
                kappa=cur_interp,
                angles=ang_interp,
                ss=interp_ss,
                init_T=init_T,
                init_N=init_N,
                return_principal_frame=True,
            )
            return r, T, Y, Z
        if self.frame_rule == 'ht':
            return self._ht_frame(cur_interp, ang_interp, interp_dist, init_T, init_N)
        raise ValueError(f"unsupported frame_rule: {self.frame_rule}")

    @staticmethod
    def _principal_frame_from_transport(T, N1, N2, initial_angle):
        """由平行输运标架恢复光纤的局部横截面 Y/Z 轴。

        输入: T/N1/N2 是全局坐标系下等长 (N,3) 方向数组；
        initial_angle 是首点曲率角，单位 rad。
        返回: Y/Z 两个 (N,3) 数组；每点 Y 与 T 正交归一，Z=T×Y。
        计算: 先抵消 N1/N2 的固定初始角旋转，再去除 Y 在 T 上的投影并归一化。
        若任一构造轴退化则抛 ValueError。
        """
        Y = np.cos(initial_angle) * N1 - np.sin(initial_angle) * N2
        Y = Y - np.sum(Y * T, axis=1, keepdims=True) * T
        Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
        eps = np.finfo(float).eps
        if np.any(Y_norm <= eps):
            raise ValueError("local y axis is numerically degenerate")
        Y = Y / Y_norm

        Z = np.cross(T, Y)
        Z_norm = np.linalg.norm(Z, axis=1, keepdims=True)
        if np.any(Z_norm <= eps):
            raise ValueError("local z axis is numerically degenerate")
        Z = Z / Z_norm
        return Y, Z


    @staticmethod
    def _local_to_global_frame(r_local, T_local, N1_local, N2_local, init_T, init_N):
        """把 HT 的参考局部系点列和标架映射到项目全局系。

        输入: 四个 *_local 数组均为 (N,3)；参考系 z 为切向、x 为 Core1 方位；
        init_T/init_N 是项目全局系入口切向/法向。
        返回: 全局系的 r,T,N1,N2 四个 (N,3) 数组。
        计算: 将 init_T 归一化为 t，将 init_N 去除沿 t 的投影后归一化为 n，
        再令 b=t×n，旋转基 R=[n,b,t]；行向量坐标乘 R.T。
        """
        norm = lambda v: v / (np.linalg.norm(v) + 1e-300)
        t = norm(np.asarray(init_T, dtype=float))
        n = norm(np.asarray(init_N, dtype=float) - np.dot(init_N, t) * t)
        b = np.cross(t, n)
        R = np.column_stack([n, b, t])
        return r_local @ R.T, T_local @ R.T, N1_local @ R.T, N2_local @ R.T


    def _ht_frame(self, kappa, angles, s, init_T, init_N):
        """逐段 4×4 齐次变换递推（无扭转标架，段中点线性插值）。
        返回 (r, T, Y, Z)，均已映射到项目全局系。来源：新算法参考/ht_algorithm.py。
        输入: kappa (1/m)、angles (rad)、s (m) 为同长度曲率网格；
        init_T/init_N 是入口初始切向、法向。
        流程: 将角度相对首点归一成 kx/ky；每段中点插值并得到曲率模长/方向；
        单段使用 Rz(theta)*Ry(kappa*ds)*Rz(-theta)，消除弯曲平面基的额外滚转；
        按 H_(i+1)=H_i*A_i 累乘单段刚体变换，每段将旋转块投影到 SO(3)，
        保留平移列，修正后的旋转参与下一段递推；提取位置和标架后映射到全局系。
        输出: 全局系 r(m)、T、Y、Z，形状均为 (N,3)。
        """
        kappa = np.asarray(kappa, dtype=float)
        angles = np.asarray(angles, dtype=float)
        s = np.asarray(s, dtype=float)
        n = len(s)
        kx = kappa * np.cos(angles - angles[0])
        ky = kappa * np.sin(angles - angles[0])

        # ---- 预计算每段在平行输运基中的 (κ, θ) ----
        s_mid = 0.5 * (s[:-1] + s[1:])
        kx_seg = np.interp(s_mid, s, kx)
        ky_seg = np.interp(s_mid, s, ky)
        kappa_seg = np.hypot(kx_seg, ky_seg)

        theta_seg = np.arctan2(ky_seg, kx_seg)

        # ---- 主递推 H_{i+1} = H_i · A_i ----
        H = np.eye(4)
        pts_local = np.zeros((n, 3))
        T_local = np.zeros((n, 3))
        N1_local = np.zeros((n, 3))
        N2_local = np.zeros((n, 3))
        T_local[0] = [0.0, 0.0, 1.0]
        N1_local[0] = [1.0, 0.0, 0.0]
        N2_local[0] = [0.0, 1.0, 0.0]
        for i in range(n - 1):
            ds = s[i + 1] - s[i]
            A = _homogeneous(kappa_seg[i], theta_seg[i], ds)
            if kappa_seg[i] > _KAPPA_EPS:
                # 论文 A 的 Rz(theta)*Ry(phi) 输出的是弯曲平面基。
                # 右乘 Rz(-theta) 恢复无扭转的横截面基；位移及 T 不变。
                # 此时每段使用绝对方向，而非上一弯曲平面基的角度差。
                c, sn = np.cos(theta_seg[i]), np.sin(theta_seg[i])
                undo_plane_rotation = np.array([[c, sn, 0.0],
                                                [-sn, c, 0.0],
                                                [0.0, 0.0, 1.0]])
                A[:3, :3] = A[:3, :3] @ undo_plane_rotation
            H = H @ A
            # SVD 极分解给出最近的正交旋转；仅修正旋转块，保留该段位移。
            U, _, Vt = np.linalg.svd(H[:3, :3])
            rotation = U @ Vt
            if np.linalg.det(rotation) < 0.0:
                U[:, -1] *= -1.0
                rotation = U @ Vt
            H[:3, :3] = rotation
            pts_local[i + 1] = H[:3, 3]
            T_local[i + 1] = H[:3, 2]
            N1_local[i + 1] = H[:3, 0]
            N2_local[i + 1] = H[:3, 1]

        r, T, N1, N2 = self._local_to_global_frame(
            pts_local, T_local, N1_local, N2_local, init_T, init_N,
        )
        Y, Z = self._principal_frame_from_transport(T, N1, N2, float(angles[0]))
        return r, T, Y, Z


    def _F_S_frame(self, kappa, tau, ss, init_T, init_N):
        """
        使用 Frenet-Serret 标架公式递推计算三维曲线及 T/N/B 标架。

        说明：
        - 当某点曲率小于 self.TNB_threshold 时，认为局部形状变化过小，直接沿用上一点的 T/N/B。
        - 当前实现的返回顺序为 (r, T, N, B)。

        输入：
        - kappa: 曲率数组，形状为 (num_points,)
        - tau: 扭率数组，形状通常为 (num_points - 1,) 或插入后与步长对齐
        - ss: 相邻点之间的弧长间隔数组
        - init_T: 初始切向量
        - init_N: 初始法向量

        输出：
        - r: 三维曲线位置数组，形状为 (num_points, 3)
        - N: 法向量数组，形状为 (num_points, 3)
        - T: 切向量数组，形状为 (num_points, 3)
        - B: 副法向量数组，形状为 (num_points, 3)
        """
        cal_norm = lambda v: v / np.linalg.norm(v)

        num_points = len(ss) + 1  #range函数会不包括这个值

        # 初始化 T, N, B, r
        r = np.zeros((num_points, 3))  # 三维位置向量
        T = np.zeros((num_points, 3))  # 切向量
        N = np.zeros((num_points, 3))  # 法向量
        B = np.zeros((num_points, 3))  # 副法向量

        # 起点的初始值
        T[0] = init_T
        N[0] = init_N
        B[0] = np.cross(init_T, init_N)

        # 在开头插入 0，保证各数组的长度相同
        ss = np.insert(ss, 0, 0)
        tau = np.insert(tau, 0, 0)

        # 迭代计算
        for i in range(1, num_points):
            if kappa[i] < self.TNB_threshold:
                N[i] = N[i-1]
                B[i] = B[i-1]
                T[i] = T[i-1]
            else:
                # 计算 N(i)
                numerator = ss[i] * tau[i] * B[i-1] - ss[i] * kappa[i] * T[i-1] + N[i-1]
                denominator = 1 + ss[i]**2 * tau[i]**2 + ss[i]**2 * kappa[i]**2
                N[i] = numerator / denominator
                N[i] = cal_norm(np.array(N[i]))

                # 计算 T(i)
                T[i] = ss[i] * kappa[i] * N[i] + T[i-1]
                T[i] = cal_norm(np.array(T[i]))

                # 计算 B(i)
                B[i] = -ss[i] * tau[i] * N[i] + B[i-1]
                B[i] = cal_norm(np.array(B[i]))

            # 计算 r(i)
            r[i] = ss[i] * T[i] + r[i-1]

        return r, T, N, B

    def _parallel_frame(self, kappa, angles, ss, init_T, init_N, return_principal_frame=False):
        """
        With ``return_principal_frame=False`` (default), return
        ``(r, T, N1, N2)`` for backward compatibility. When enabled, return
        ``(r, T, Y, Z, N1, N2)``, where Y and Z are the fiber-local y/z axes
        expressed in the global coordinate system.

        使用平行标架 / TNN 形式递推计算三维曲线及 T/N1/N2 标架。

        说明：
        - 根据 angles 相对于首点角度的变化，将曲率分解为 kappa1 与 kappa2。
        - N1 由 init_N 初始化，N2 由 np.cross(init_T, init_N) 初始化。
        - 当前实现不使用扭率 tau，也不使用 self.TNB_threshold。
        - 当前实现的返回顺序为 (r, T, N1, N2)。

        输入：
        - kappa: 曲率大小数组，形状为 (num_points,)
        - angles: 曲率方向角数组，形状为 (num_points,)
        - ss: 相邻插值点之间的弧长间隔数组，形状为 (num_points - 1,)
        - init_T: 初始切向量
        - init_N: 初始 N1 方向

        输出：
        - r: 三维曲线位置数组，形状为 (num_points, 3)
        - T: 切向量数组，形状为 (num_points, 3)
        - N1: 第一法向基数组，形状为 (num_points, 3)
        - N2: 第二法向基数组，形状为 (num_points, 3)
        """
        # Define normlize func
        cal_norm = lambda v: v / np.linalg.norm(v)

        num_points = len(ss) + 1  #range函数会不包括这个值

        # 初始化 T, N, B, r
        r = np.zeros((num_points, 3))  # 三维位置向量
        T = np.zeros((num_points, 3))  # 切向量
        N1 = np.zeros((num_points, 3))  # 法向量
        N2 = np.zeros((num_points, 3))  # 副法向量

        # 起点的初始值
        T[0] = init_T
        N1[0] = init_N
        N2[0] = np.cross(init_T, init_N) # 也可以反过来init_N\times init_T

        # 首个测点的 Fiber A 方向定义为局部零度；从局部 +Y
        # 向 +Z 为正方向。所以曲率分量使用相对首点的角度。
        angles_rel = angles - angles[0]
        kappa1 = kappa * np.cos(angles_rel)
        kappa2 = kappa * np.sin(angles_rel)

        # 在开头插入 0，保证各数组的长度相同
        ss = np.insert(ss, 0, 0)

        # 迭代计算
        for i in range(1, num_points):
            numerator = T[i - 1] + ss[i] * (kappa1[i] * N1[i - 1] + kappa2[i] * N2[i - 1])
            denominator = 1.0 + ss[i]**2 * (kappa1[i]**2 + kappa2[i]**2)
            T[i] = numerator / denominator
            T[i] = cal_norm(np.array(T[i]))

            N1[i] = N1[i - 1] - ss[i] * kappa1[i] * T[i]
            N1[i] = cal_norm(N1[i])

            N2[i] = N2[i - 1] - ss[i] * kappa2[i] * T[i]
            N2[i] = cal_norm(N2[i])

            r[i] = ss[i] * T[i] + r[i-1]

        if not return_principal_frame:
            return r, T, N1, N2

        # N1/N2 的初始方向相对全局 Y/Z 轴旋转了 angles[0]。
        # 将该固定初始旋转取消，得到光纤局部 y 轴在全局
        # 坐标系中的方向；再用 T x Y 得到右手系的局部 z 轴。
        initial_angle = angles[0]
        Y = np.cos(initial_angle) * N1 - np.sin(initial_angle) * N2
        Y = Y - np.sum(Y * T, axis=1, keepdims=True) * T
        Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
        eps = np.finfo(float).eps
        if np.any(Y_norm <= eps):
            raise ValueError("TNN local y axis is numerically degenerate")
        Y = Y / Y_norm

        Z = np.cross(T, Y)
        Z_norm = np.linalg.norm(Z, axis=1, keepdims=True)
        if np.any(Z_norm <= eps):
            raise ValueError("TNN local z axis is numerically degenerate")
        Z = Z / Z_norm

        return r, T, Y, Z, N1, N2

    @staticmethod
    def _build_end_pose(T, Y, Z, rot_matrix, rot_ang_deg):
        """Rotate the terminal frame globally, then undo rot_ang about its rotated T axis."""
        T = np.asarray(T, dtype=float)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=float)
        rot_matrix = np.asarray(rot_matrix, dtype=float)
        rot_ang_deg = float(rot_ang_deg)

        if T.ndim != 2 or Y.ndim != 2 or Z.ndim != 2:
            raise ValueError("T, Y and Z must be two-dimensional frame arrays")
        if T.shape[1] != 3 or Y.shape[1] != 3 or Z.shape[1] != 3:
            raise ValueError("T, Y and Z must have shape (num_points, 3)")
        if not (len(T) == len(Y) == len(Z)):
            raise ValueError("T, Y and Z must contain the same number of points")
        if rot_matrix.shape != (3, 3) or not np.all(np.isfinite(rot_matrix)):
            raise ValueError("rot_matrix must be a finite matrix with shape (3, 3)")
        if not np.isfinite(rot_ang_deg):
            raise ValueError("rot_ang_deg must be finite")
        if len(T) == 0:
            return None

        terminal_t = T[-1] @ rot_matrix.T
        terminal_y = Y[-1] @ rot_matrix.T
        terminal_z = Z[-1] @ rot_matrix.T
        t_norm = float(np.linalg.norm(terminal_t))
        if not np.isfinite(t_norm) or t_norm <= np.finfo(float).eps:
            raise ValueError("terminal T axis must be finite and non-zero")
        local_axis = terminal_t / t_norm
        theta = np.deg2rad(-rot_ang_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        def rotate_about_terminal_t(vector):
            return (
                vector * cos_theta
                + np.cross(local_axis, vector) * sin_theta
                + local_axis * np.dot(local_axis, vector) * (1.0 - cos_theta)
            )

        return np.stack(
            (
                terminal_t,
                rotate_about_terminal_t(terminal_y),
                rotate_about_terminal_t(terminal_z),
            )
        )

    def _get_rot_x(self, direct, rot_dir=-1):
        """
        生成绕 x 轴旋转的旋转矩阵。

        输入：
        - direct: 当前方向角（度）

        输出：
        - rot_matrix: 3×3 的绕 x 轴旋转矩阵
        """
        sigma = np.deg2rad((direct) * rot_dir + self.rot_ang)
        rot_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(sigma), -np.sin(sigma)],
            [0.0, np.sin(sigma), np.cos(sigma)]
        ])
        return rot_matrix


class ShapeIntegrator(ShapeIntegrationMixin):
    """由曲率、平面方向和弧长直接积分形状。

    输入数组长度一致且至少有两个样本；曲率单位为 1/m，角度为 rad，弧长为 m，
    init_T 为全局单位切向量。返回 position、T、Y、Z 四个 (N, 3) 数组。
    不创建 Sensor、不读取外部文件；TNB 返回 position/T/N/B，TNN/ht 返回 position/T/Y/Z，
    保留旧实现的分支返回顺序。
    """
    def __init__(self, frame_rule='TNB', TNB_threshold=0.15, rot_ang=0.0):
        self.frame_rule = frame_rule
        self.TNB_threshold = TNB_threshold
        self.rot_ang = rot_ang

    def integrate(self, cur_interp, ang_interp, interp_dist, init_T):
        """对已插值曲率积分并返回位置与局部标架数组。"""
        return self._reconstruct_from_interpolated_curvature(
            cur_interp, ang_interp, interp_dist, init_T)
