"""原UDP协议与异步记录器；不导入Qt/VTK。"""
import pickle
import queue
from datetime import datetime
import time
import socket
import struct
import json
import threading
import numpy as np
from collections import deque
RECEIVER_IP = '192.168.110.71'
# RECEIVER_IP = '192.168.111.123'
RECEIVER_PORT = 8888
SYNC_IP = '127.0.0.1'
SYNC_PORT = 50000
SENSOR_TIP_SETBACK_M = 0.0


def _build_sync_packet(timestamp, r_results_postrot, end_pose, cur_interp,
                       raw_data, n_pts, n_raw):
    """Build the legacy fixed-length binary packet used by the C++ recorder."""
    r_flat = np.asarray(r_results_postrot, dtype=float).flatten()
    r_flat = np.pad(r_flat, (0, max(0, 3 * n_pts - len(r_flat))))[:3 * n_pts]
    ep = np.asarray(end_pose if end_pose is not None else [0.0, 0.0, 0.0], dtype=float).flatten()
    ep = np.pad(ep, (0, max(0, 3 - len(ep))))[:3]
    cur_flat = np.asarray(cur_interp if cur_interp is not None else [], dtype=float).flatten()
    cur_flat = np.pad(cur_flat, (0, max(0, n_pts - len(cur_flat))))[:n_pts]
    raw_flat = np.asarray(raw_data if raw_data is not None else [], dtype=float).flatten()
    raw_flat = np.pad(raw_flat, (0, max(0, n_raw - len(raw_flat))))[:n_raw]
    values = np.concatenate((np.asarray([timestamp], dtype=float), r_flat, ep, cur_flat, raw_flat))
    return struct.pack('<' + 'd' * len(values), *values)


def _shape_for_software_send(shape_mm, anchor_enabled, insert_value_mm):
    """Undo X insertion only for anchored JSON data that the receiver re-applies."""
    adjusted = np.asarray(shape_mm, dtype=float).copy()
    if not anchor_enabled or insert_value_mm is None:
        return adjusted
    try:
        values = np.asarray(insert_value_mm, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return adjusted
    if values.size != 1 or not np.isfinite(values[0]):
        return adjusted
    adjusted[:, 0] -= float(values[0])
    return adjusted


def _anchor_point_for_software_send(
    anchor_point_m,
    anchor_enabled,
    insert_value_mm,
):
    """
    将当前帧的入口点/旋转中心转换到软件端坐标系。

    `anchor_point_m` 使用重构坐标系，单位为 m；软件端数据使用 mm，并会自行
    叠加机械臂的绝对推进量。因此这里与 `_shape_for_software_send` 保持一致：
    先将锚点转换为 mm，再从 X 坐标减去当前机械臂绝对推进量。

    仅旋转锚定时，该点是当前帧插值切向 T 的位置，同时也是旋转中心；
    平移锚定开启时，该点是平移后的固定入口位置。无法形成有效软件坐标时
    返回 None，使 JSON 中的 `anchor_point` 序列化为 null。
    """
    if not anchor_enabled or anchor_point_m is None or insert_value_mm is None:
        return None
    try:
        point_m = np.asarray(anchor_point_m, dtype=float)
        values = np.asarray(insert_value_mm, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if (
        point_m.shape != (3,)
        or not np.all(np.isfinite(point_m))
        or values.size != 1
        or not np.isfinite(values[0])
    ):
        return None
    point_mm = point_m * 1000.0
    point_mm[0] -= float(values[0])
    return point_mm


def _build_send_obj(
    end_pose,
    r_results_postrot,
    raw_data,
    intermediates,
    anchor_point_mm=None,
):
    """
    构造发送给软件端的 JSON 对象。

    `r_results_postrot` 和 `anchor_point_mm` 必须已经转换到同一个软件端毫米
    坐标系。`anchor_point_mm` 为 None 时仍保留字段并输出 null，便于接收端
    使用固定的数据结构判断当前帧是否有有效入口点。
    """
    return {
        'end_pose': end_pose.tolist() if end_pose is not None else [0.0, 0.0, 0.0],
        'data': r_results_postrot.tolist(),
        'anchor_point': (
            anchor_point_mm.tolist() if anchor_point_mm is not None else None
        ),
        'raw_data': raw_data.tolist() if raw_data is not None else [],
        'intermediates': {
            'magnitude': intermediates['magnitude'].tolist(),
            'angle': intermediates['angle'],
            'cur_kal': intermediates['cur_kal'].tolist(),
            'ang_filter': intermediates['ang_filter'].tolist(),
            'cur_interp': intermediates['cur_interp'].tolist(),
            'ang_interp': intermediates['ang_interp'].tolist(),
        },
        'cur_interp': intermediates['cur_interp'].tolist(),
    }


def _rotate_end_pose_for_anchor(end_pose, rotation_matrix):
    """Apply the global anchor rotation after End_pose roll compensation."""
    if end_pose is None:
        return None
    end_pose = np.asarray(end_pose, dtype=float)
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if end_pose.shape != (3, 3):
        raise ValueError("end_pose must have shape (3, 3)")
    if rotation_matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3)")
    return end_pose @ rotation_matrix.T





# ==============================================================================
# 数据记录器类：用于后台异步保存数据，避免阻塞 VTK 渲染主线程
# ==============================================================================
class DataRecorder:
    def __init__(self, filename_prefix="ShapeData"):
        start_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{filename_prefix}_{start_str}.pkl"
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print(f"--- 数据记录已启动，保存至: {self.filename} ---")

    def save(
        self,
        timestamp,
        r_results,
        end_pose,
        cur_interp,
        raw_data,
        intermediates,
        insert_value=None,
        r_results_before_anchor=None,
        anchor_point_m=None,
    ):
        """
        将当前帧的重构结果打包后放入后台写入队列。

        步骤：
        1. 若记录器已停止运行，则直接返回。
        2. 将输入的时间戳、重构结果、末端姿态和曲率插值结果打包为字典。
        3. 对支持 copy() 的对象做浅拷贝，避免后续主线程修改影响已排队的数据。
        4. 以非阻塞方式将数据包放入 self.queue。
        5. 若队列已满，则直接丢弃当前帧，保证渲染实时性。

        输入：
        - timestamp: 当前帧时间戳
        - r_results: 当前帧经过入口锚定后的重构三维轨迹结果
        - r_results_before_anchor: 当前帧入口锚定前的重构三维轨迹结果
        - anchor_point_m: 软件坐标转换前、单位为 m 的入口点/旋转中心
        - end_pose: 当前帧末端姿态
        - cur_interp: 当前帧插值后的曲率序列

        输出：
        - 无显式返回
        - 副作用：
            - 向后台写入队列 self.queue 中加入一个数据包

        说明：
        - 当前实现使用 queue.Queue 的非阻塞 put，因此写入失败不会阻塞前台渲染线程
        - cur_interp 会被转换为 list；若其为 None，则保存为空列表
        """
        if not self.running:
            return
        try:
            data_packet = {
                'timestamp': timestamp,
                # 保留旧字段含义：r_results 始终是入口锚定变换后的曲线。
                'r_results': r_results.copy() if hasattr(r_results, 'copy') else r_results,
                # 保存锚定输入，便于离线比较或重新应用其他锚定参数。
                'r_results_before_anchor': (
                    r_results_before_anchor.copy()
                    if hasattr(r_results_before_anchor, 'copy')
                    else r_results_before_anchor
                ),
                # 软件坐标转换前的每帧入口点/旋转中心，单位为 m。
                'anchor_point_m': (
                    anchor_point_m.copy()
                    if hasattr(anchor_point_m, 'copy')
                    else anchor_point_m
                ),
                'end_pose': end_pose.copy() if hasattr(end_pose, 'copy') else end_pose,
                'cur_interp': intermediates['cur_interp'].tolist(),
                'raw_data': list(raw_data) if raw_data is not None else [],
                'magnitude': intermediates['magnitude'].tolist(),
                #'angle': intermediates['angle'].tolist(),
                'angle': intermediates['angle'],
                'cur_kal': intermediates['cur_kal'].tolist(),
                'ang_filter': intermediates['ang_filter'].tolist(),
                'ang_interp': intermediates['ang_interp'].tolist(),
                **({
                    'cur_interp_raw': intermediates['cur_interp_raw'].tolist(),
                    'ang_interp_raw': intermediates['ang_interp_raw'].tolist(),
                } if 'cur_interp_raw' in intermediates else {}),
                'insert_value':insert_value.copy() if hasattr(insert_value, 'copy') else insert_value
            }
            # 非阻塞放入，保证渲染实时性
            self.queue.put(data_packet, block=False)
        except queue.Full:
            pass

    def _worker(self):
        """
        后台循环消费写入队列，并将数据逐包序列化保存到 pkl 文件。

        步骤：
        1. 以写入模式打开当前记录文件。
        2. 在 self.running 为 True 或队列仍非空时持续循环。
        3. 尝试从 self.queue 中取出一个数据包。
        4. 使用 pickle.dump 将数据包顺序写入文件。
        5. 写入成功后调用 task_done 标记当前任务完成。
        6. 当队列暂时为空时继续等待；若写入过程异常，则打印错误信息。

        输入：
        - 无（使用实例中的 self.filename、self.queue 与 self.running）

        输出：
        - 无显式返回
        - 副作用：
            - 持续向 self.filename 对应的 pkl 文件中追加序列化数据包

        说明：
        - 该方法设计为后台守护线程入口函数
        - 文件会在 with 代码块结束后自动关闭
        """
        with open(self.filename, 'wb') as f:
            while self.running or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    pickle.dump(data, f)
                    self.queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"写入错误: {e}")

    def stop(self):
        self.running = False
        self.thread.join()
        print(f"--- 数据记录结束，文件已关闭 ---")

