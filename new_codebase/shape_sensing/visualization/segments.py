"""绘图分段；保留初始化原公式，仅消费显式坐标和配置。"""
import numpy as np

class PlotSegments:
    def __init__(self, inputs):
        """输入VisualizationInputs，输出只持有绘图距离(P,)m的分段器。"""
        self.samp_dist = inputs.dataset.design.sampling_distance_m.copy()
        self.segmentation = inputs.segmentation
        self.discont = inputs.discont_rad
        self.connect = inputs.connect
        self.zero_init = inputs.zero_init
        self.straight_trim = inputs.straight_trim_m

    def _segment_data(self, angles):
        """
        为单条自扭转角度曲线生成对应的距离数组、unwrap 角度数组和拟合分段。

        参数：
        - angles: 单条曲率方向角序列，单位 rad。
        可以是原始 angle，也可以是已经部分展开的角度序列；本方法内部会再次调用 np.unwrap()。

        处理流程：
        1. 使用 np.unwrap(angles, discont=self.discont) 得到连续角度曲线 unwrapped。
        2. 取 unwrapped 与 self.samp_dist 的共同有效长度 raw_valid_length。
        3. 根据 self.zero_init 决定 x/y 数据的起点：
        - 若 self.zero_init=True，则从第 0 个采样点开始使用；
        - 若 self.zero_init=False，则跳过第 0 个采样点，从第 1 个采样点开始使用。
        4. 截断 x_data 和 y_data，使二者长度一致。
        5. 若 total_points<=0，则返回空分段。
        6. 调用 _calculate_segments(total_points) 生成不重叠基础分段 base_segments。
        7. 若 self.connect=False，直接返回 base_segments。
        8. 若 self.connect=True：
        - 第 0 段保持不变；
        - 从第 1 段开始，每段 start 向左扩展 1 个点；
        - end 保持不变；
        - 以此让相邻分段在拟合时共享一个边界点。

        输出：
        - x_data: 与角度曲线对应的采样距离数组，单位 m。
        - y_data: unwrap 后的角度数组，单位 rad。
        - segments: 用于拟合的分段索引区间列表，格式为 [(start, end), ...]。

        说明：
        - segments 采用左闭右开区间。
        - connect=True 只影响拟合阶段的分段索引，不改变原始 x_data/y_data。
        - self.zero_init 的语义来自 Sensor.generate_samp_dist()：当 ad_samp_dist 中需要补起点虚拟点时，
        这里保留第 0 个真实采样点；否则跳过第 0 个点以避免起点对拟合造成影响。
        """
        unwrapped = np.unwrap(angles, discont=self.discont)
        raw_valid_length = min(len(unwrapped), len(self.samp_dist))

        if self.zero_init or self.straight_trim != 0:   #如果是裁剪了，那第一个点也要保留
            x_data = self.samp_dist[:raw_valid_length]
            y_data = unwrapped[:raw_valid_length]
        else:
            x_data = self.samp_dist[1:raw_valid_length]
            y_data = unwrapped[1:raw_valid_length]

        total_points = min(len(x_data), len(y_data))
        x_data = x_data[:total_points]
        y_data = y_data[:total_points]

        if total_points <= 0:
            return x_data, y_data, []

        # Step 1: 生成不重叠基础分段
        base_segments = self._calculate_segments(total_points)

        # Step 2: 如果不启用连接，则直接返回不重叠分段
        if not self.connect:
            return x_data, y_data, base_segments

        # Step 3: 启用连接时，从第二段开始，每段向左拓展 1 个点
        if len(base_segments) <= 1:
            return x_data, y_data, base_segments

        segments = []

        for idx, (start, end) in enumerate(base_segments):
            start = int(start)
            end = int(end)

            # 首段不变；此后每段向左拓展一个点
            if idx > 0:
                start = max(start - 1, 0)

            if end > start:
                segments.append((start, end))

        return x_data, y_data, segments

    def _calculate_segments(self, total_points):
        """
        根据 self.segmentation 和当前数据长度生成不重叠分段。

        参数：
        - total_points: 当前可用于拟合或重构的数据点总数。

        segmentation 规则：
        - self.segmentation = [n1, n2, ..., nk]
        - 列表长度 k 表示目标分段数。
        - 每个元素 ni 表示对应分段的点数。

        处理流程：
        1. 若 total_points <= 0，返回空列表 []。
        2. 检查 self.segmentation 是否为非空 list 或 tuple。
        若非法，则退化为单段 [(0, total_points)]。
        3. 尝试将 segmentation 中每个元素转换为 int。
        若转换失败，则退化为单段。
        4. 若存在负数，则退化为单段。
        5. 删除长度为 0 的分段。
        若删除后为空，则退化为单段。
        6. 从头到尾按 segmentation 指定长度生成左闭右开的不重叠分段。
        7. 最后一段自动吸收所有剩余点，避免尾部数据未被覆盖。
        8. 若 segmentation 总长度超过 total_points，则自动在 total_points 处截断。

        输出：
        - segments: 分段区间列表，格式为 [(start, end), ...]。
        若无法生成有效分段，则返回 [(0, total_points)]。

        说明：
        - 本方法生成的是基础不重叠分段。
        - 若 self.connect=True，相邻段共享边界点的逻辑不在这里执行，而是在 _segment_data() 中执行。
        - 返回分段通常用于 _piecewise_linear_fit() 或后续重构中的分段扭转系数应用。
        """
        if total_points <= 0:
            return []

        seg_cfg = self.segmentation

        # segmentation 必须是非空 list / tuple
        if not isinstance(seg_cfg, (list, tuple)) or len(seg_cfg) == 0:
            return [(0, total_points)]

        # 尝试转成整数列表
        try:
            lengths = [int(x) for x in seg_cfg]
        except (TypeError, ValueError):
            return [(0, total_points)]

        # 不允许负数
        if any(x < 0 for x in lengths):
            return [(0, total_points)]

        # 去掉 0 长度段；如果全为 0，则退化为单段
        lengths = [x for x in lengths if x > 0]
        if len(lengths) == 0:
            return [(0, total_points)]

        segments = []
        start = 0

        for i, seg_len in enumerate(lengths):
            if start >= total_points:
                break

            # 最后一段吸收所有剩余点，避免尾部漏点
            if i == len(lengths) - 1:
                end = total_points
            else:
                end = min(start + seg_len, total_points)

            if end > start:
                segments.append((start, end))

            start = end

        return segments if segments else [(0, total_points)]
