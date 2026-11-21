"""真实点分段几何，保留原计算正文。"""
class SegmentGeometry:
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
