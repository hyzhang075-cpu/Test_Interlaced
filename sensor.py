import glob
import os
import queue
import socket
import struct
import time
import re
import logging
import copy

import numpy as np
import pandas as pd

from pathlib import Path
from collections import defaultdict
from scipy.signal import savgol_filter

from utils import build_valid_row_mask


class Sensor:
    """
    传感器自身物理参数以及数据处理模块
    """
    def __init__(self, sensor_id, start_value, folder_path, centre_fibre=True, active_channels=[1, 3, 4], rot_dir=-1, exp_directs=[0, 120, 240], interrogator='FiberTecc', R=0.235e-3, ideal_len=1.35,
                radii={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300, 200], 'S': [0, 250, 200, 150], 'end': {2000: [30, 20, 15], 1500: [150, 100, 50]}, 
                       'rot0': {100: [10, 40, 70]}, 'up': {200: [0, 25, 50, 100], 100: [0, 25, 50, 100]}, 'down': {200: [0, 25, 50, 100], 100: [0, 25, 50, 100]}}, 
                samp_dist_preset=[510, 540, 570, 630, 710, 810, 910, 1010, 1110, 1210, 1310, 1410, 1510, 1610, 1710, 1815], active_angles=None,
                ss=2.6e-3, N=100, sensing_ranges={1: (583, 1048), 3: (678, 1146), 4: (749, 1215)}, smoothing_params={'window_length': 21,'polyorder': 3},
                IP='192.168.1.75', PORT=26546, data_freq=100, maxsize=100):
        
        # 实验参数
        self.exp_directs = exp_directs
        self.interrogator = interrogator
        self.sensor_id = sensor_id
        self.radii = {exp_direct: copy.deepcopy(radii) for exp_direct in self.exp_directs}
        self.calib_shapes = list(radii.keys())

        # active_channels:
        #   物理通道号 -> 当前运行时矩阵列号（按输入顺序）
        self.active_channels = {phy_ch: i for i, phy_ch in enumerate(active_channels)}
        self.num_channels = len(active_channels)
        self.num_channels_recon = self.num_channels
        self.active_angles = active_angles

        # ch_order:
        #   保持 read_data_FBG 现有逻辑不变：
        #   把“按物理通道升序读出的列”重排成 active_channels 的输入顺序
        self.ch_order = [sorted(active_channels).index(phy_ch) for phy_ch in active_channels]

        self.rot_dir = rot_dir  # 标定实验时旋转传感器的方向，1对应逆时针，-1对应顺时针
        self.centre_fibre = centre_fibre

        # LUNA参数
        self.ss = ss
        self.num_groups = len(radii)
        self.N = N
        if interrogator == 'LUNA':
            missing_ranges = [phy_ch for phy_ch in active_channels if phy_ch not in sensing_ranges]
            if missing_ranges:
                raise ValueError(f"sensing_ranges 缺少这些激活物理通道的配置: {missing_ranges}")

        self.min_len = min(end - start for phy_ch, (start, end) in sensing_ranges.items() if phy_ch in self.active_channels)

        # sensing_ranges 的 key 直接使用物理通道号
        self.sensing_ranges = {phy_ch: (start, start + self.min_len) for phy_ch, (start, _) in sensing_ranges.items() if phy_ch in self.active_channels}
        self.smoothing_params = smoothing_params

        # 传感器物理参数
        self.zero_init = start_value != 0
        self.generate_samp_dist(start_value=start_value, samp_dist_preset=samp_dist_preset)  # 生成传感点在传感器上所处的实际位置（单位：米）
        self.total_points = len(self.samp_dist)
        self.R = R
        self.ideal_len = ideal_len

        # 动态数据接收相关参数
        self.IP = IP
        self.PORT = PORT
        self.data_queue = queue.Queue(maxsize=maxsize)
        self.data_freq = data_freq

        # 动态数据处理相关参数
        self.zero_points = {ch: [None] * self.total_points for ch in active_channels}
        self.is_zero_set = {ch: False for ch in active_channels}
        self.current_snapshot = np.zeros((self.total_points, self.num_channels))
        self.buffer = bytearray()

        if self.interrogator in ['FiberTecc', 'Hippo']:
            self.read_data_FBG(folder_path=folder_path)
            if self.centre_fibre:
                self.centreData, self.outerData, self.processed_Data, self.plot_Data = self._process_centreData(data=self.Data)
                self.num_channels_recon = self.num_channels - 1
            else:
                self.processed_Data = self.Data
                self.plot_Data = self.Data
        elif self.interrogator == 'LUNA':
            self.read_data_OFDR(folder_path=folder_path)
             

    def generate_samp_dist(self, start_value, samp_dist_preset):
        """
        根据当前解调仪类型生成采样距离数组，并更新实例成员。

        步骤：
        1. 若 self.interrogator == 'LUNA'，则按 self.min_len 与 self.ss 直接生成等间距的 self.samp_dist。
        2. 否则，对 samp_dist_preset 做差分并倒序，以 start_value 为起点逐步累加生成 self.samp_dist。
        3. 若 self.zero_init=True，则在 self.samp_dist 开头额外插入 0，得到 self.ad_samp_dist；否则 self.ad_samp_dist 与 self.samp_dist 相同。
        4. 检查生成结果是否为有限值，且 self.ad_samp_dist 是否严格递增。

        输入：
        - start_value: 固定件到第一个有效采样点的距离
        - samp_dist_preset: 原始光栅位置数组或预设距离数组

        输出：
        - 无显式返回
        - 更新以下成员：
            - self.samp_dist: 主采样距离数组，单位为 m
            - self.ad_samp_dist: 若 self.zero_init=True 则在开头额外补 0，否则与 self.samp_dist 相同

        异常：
        - 当 self.samp_dist 包含 NaN / inf 时抛出 ValueError
        - 当 self.ad_samp_dist 不是严格递增数组时抛出 ValueError

        说明：
        - LUNA 模式下不使用 samp_dist_preset 的差分结果，而是直接按空间采样间隔生成
        - self.samp_dist 本身可以包含 0；zero_init 只决定是否在 self.ad_samp_dist 中额外插入一个起始 0
        """
        if self.interrogator == 'LUNA':
            self.samp_dist = np.arange(self.min_len) * self.ss
        else:
            diff = np.diff(samp_dist_preset)
            diff_reversed = diff[::-1]
            samp_dist = [start_value]
            for d in diff_reversed:
                samp_dist.append(samp_dist[-1] + d)
            self.samp_dist = np.array(samp_dist) * 0.001
        if self.zero_init:
            self.ad_samp_dist = np.insert(self.samp_dist, 0, 0)
        else:
            self.ad_samp_dist = self.samp_dist
        if not np.all(np.isfinite(self.samp_dist)):
            raise ValueError("samp_dist 包含非有限值（NaN 或 inf）")
        if not np.all(np.diff(self.ad_samp_dist) > 0):
            raise ValueError("ad_samp_dist 必须是递增的数组")
    
    # -------------------- FBG数据处理 --------------------
    def read_data_FBG(self, folder_path):
        """
        从 FBG 静态标定数据文件夹中读取各类子文件夹下的通道数据，清洗后整理为统一结构，并更新 self.Data。

        支持的子文件夹命名格式：
        1. x-y
        -> 保存到 self.Data["circ"][x][y]
        2. x-y-S
        -> 保存到 self.Data["S"][x][y]
        3. x-y-z
        -> 保存到 self.Data["end"][x][y][z]
        4. x-y-z-R0
        -> 保存到 self.Data["rot0"][x][y][z]
        5. x-y-z-up
        -> 保存到 self.Data["up"][x][y][z]
        6. x-y-z-down
        -> 保存到 self.Data["down"][x][y][z]

        步骤：
        1. 初始化 self.Data 为包含 circ、S、end、rot0、up、down 六类数据的空字典。
        2. 遍历 folder_path 下的各个子文件夹，并根据命名格式解析出 x、y、z 与 folder_type。
        3. 对每个子文件夹中的各个 channel 文件：
            - 根据解调仪类型匹配文件名；
            - 读取 csv；
            - 忽略只有表头没有数据行的空文件；
            - 检查列数是否一致；
            - 去除含缺数据的行；
            - 使用 build_valid_row_mask 去除 outlier 行；
            - 对剩余有效行按列求均值。
        4. 若某个 channel 清洗后无有效行，则跳过该 channel。
        5. 若整个 subfolder 最终没有任何有效 channel，则跳过该 subfolder，并调用 _edit_radii 撤销对应半径项。
        6. 若 active_channels 所要求的任意必需 channel 缺失，则同样跳过该 subfolder，并调用 _edit_radii 撤销对应半径项。
        7. 将保留下来的各必需 channel 均值向量按 active_channels 的输入顺序组装为 data_matrix，并做 1e-6 单位缩放。
        8. 按 folder_type 将 data_matrix 写入 self.Data 的对应层级。
        9. 对各层字典按半径键做降序排序，其中 0 被视为最大值参与排序。

        输入：
        - folder_path: 标定数据总目录路径

        输出：
        - 无显式返回
        - 更新以下成员：
            - self.Data:
                - circ / S 结构为 self.Data[shape][x][y]
                - end / rot0 / up / down 结构为 self.Data[shape][x][y][z]
                - 各叶子节点的值为形状为 (num_points, num_channels) 的 numpy 数组

        说明：
        - 当前实现中，不仅当整个 subfolder 没有任何有效 channel 时会撤销 self.radii，
        当构建 data_matrix 所需的必需 channel 缺失时也会撤销。
        - 缺失非必需 channel 不影响当前 subfolder 保留。
        - 最终 data_matrix 的列顺序由 active_channels 的输入顺序决定。
        - 当前实现若同一 channel 匹配到多个文件，只取第一个文件。
        """
        self.Data = {"circ": {}, "S": {}, "end": {}, "rot0": {}, "up": {}, "down": {}}

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            x, y, z, folder_type = self._classify_folder(subfolder=subfolder)
            if folder_type is None:
                continue

            # key 是真实读取时使用的 channel 编号，value 是该 channel 的均值向量
            channel_data = {}
            valid_subfolder = True
            num_columns = None
            col_start, col_end = None, None

            for channel in range(max(self.active_channels) + 1):
                if self.interrogator == 'FiberTecc':
                    matched_files = glob.glob(os.path.join(subfolder_path, f"channel_{channel}_*.csv"))
                elif self.interrogator == 'Hippo':
                    matched_files = glob.glob(os.path.join(subfolder_path, f"CH{channel}_*.csv"))
                else:
                    print(f"Unsupported interrogator type: {self.interrogator}")
                    valid_subfolder = False
                    break

                if not matched_files:
                    continue

                # 默认每个 channel 对应一个文件；如果有多个，只取第一个
                file_path = matched_files[0]

                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    logging.warning(f"文件 {file_path} 读取失败，跳过该 channel。error={e}")
                    continue

                # 一开始就忽略只有 headings 的空 csv
                if df.empty:
                    logging.info(f"文件 {file_path} 只有表头没有数据行，视为未连接通道，已跳过 channel {channel}。")
                    continue

                current_num_cols = df.shape[1]

                if num_columns is None:
                    num_columns = current_num_cols - 2
                    col_start, col_end = 2, current_num_cols
                else:
                    expected_cols = num_columns + 2
                    if current_num_cols != expected_cols:
                        print(f"Column mismatch in {subfolder}: ")
                        print(f"Channel {channel} has {current_num_cols} columns ")
                        print(f"(expected {expected_cols})")
                        valid_subfolder = False
                        break

                # -------------------- 清洗 outlier --------------------
                fbg_cols = list(enumerate(df.columns[col_start:col_end]))
                fbg_col_names = [col for _, col in fbg_cols]

                numeric_fbg = df[fbg_col_names].apply(pd.to_numeric, errors="coerce")

                # 1) 缺数据行排除
                non_missing_mask = ~numeric_fbg.isna().any(axis=1)

                # 2) outlier 行排除
                outlier_valid_mask = build_valid_row_mask(
                    df=df, fbg_cols=fbg_cols, threshold=9000, half_window=5
                )

                valid_rows = non_missing_mask.to_numpy() & outlier_valid_mask

                removed_missing_count = int((~non_missing_mask).sum())
                removed_outlier_count = int((non_missing_mask.to_numpy() & ~outlier_valid_mask).sum())

                if removed_missing_count > 0:
                    logging.warning(f"警告：文件 {file_path} 中有 {removed_missing_count} 行存在缺数据，已跳过相关行。")

                if removed_outlier_count > 0:
                    logging.warning(f"警告：文件 {file_path} 中有 {removed_outlier_count} 行被判定为 outlier，已跳过相关行。")

                cleaned_df = df.loc[valid_rows].copy()

                if cleaned_df.empty:
                    logging.warning(f"文件 {file_path} 在缺数据 / outlier 清洗后已无有效行，该 channel 将被跳过。")
                    continue

                data_slice = cleaned_df.iloc[:, col_start:col_end].apply(pd.to_numeric, errors="coerce")
                channel_avg = data_slice.mean(axis=0).to_numpy()

                channel_data[channel] = channel_avg

            if not valid_subfolder:
                continue

            # 整个 subfolder 没有任何有效 channel
            if len(channel_data) == 0:
                logging.error(f"严重错误：subfolder {subfolder} 中没有任何有效 channel，当前 subfolder 将被跳过，并从 self.radii 中撤销对应项。")
                self._edit_radii(folder_type=folder_type, x=x, y=y, z=z)
                continue

            # 根据 active_channels 构造“必需 channel 列表”
            if self.interrogator == 'Hippo':
                required_channels = [phy_ch - 1 for phy_ch in self.active_channels]
                missing_channels = [ch for ch in required_channels if ch not in channel_data]
                missing_display_channels = [ch + 1 for ch in missing_channels]
            elif self.interrogator == 'FiberTecc':
                required_channels = list(self.active_channels)
                missing_channels = [ch for ch in required_channels if ch not in channel_data]
                missing_display_channels = missing_channels
            else:
                logging.error(f"Unsupported interrogator type: {self.interrogator}")
                continue

            # 只要缺少任一必需 channel，就撤销对应 radii
            if missing_channels:
                logging.error(f"严重错误：subfolder {subfolder} 缺少构建 data_matrix 所需的必需通道{missing_display_channels}，当前 subfolder 将被跳过，并从 self.radii 中撤销对应项。")
                self._edit_radii(folder_type=folder_type, x=x, y=y, z=z)
                continue

            data_matrix = np.column_stack([channel_data[ch] for ch in required_channels]) * 1e-6

            if folder_type in ("circ", "S"):
                self.Data[folder_type].setdefault(x, {})
                self.Data[folder_type][x][y] = data_matrix

            elif folder_type in ("end", "rot0", "up", "down"):
                self.Data[folder_type].setdefault(x, {})
                self.Data[folder_type][x].setdefault(y, {})
                self.Data[folder_type][x][y][z] = data_matrix

        for shape in ["circ", "S"]:
            for x in list(self.Data[shape].keys()):
                self.Data[shape][x] = dict(sorted(self.Data[shape][x].items(), key=lambda item: float('inf') if item[0] == 0 else item[0], reverse=True))
            self.Data[shape] = dict(sorted(self.Data[shape].items(), key=lambda item: float('inf') if item[0] == 0 else item[0], reverse=True))

        for shape in ["end", "rot0", "up", "down"]:
            for x in list(self.Data[shape].keys()):
                for y in list(self.Data[shape][x].keys()):
                    self.Data[shape][x][y] = dict(sorted(self.Data[shape][x][y].items(), key=lambda item: float('inf') if item[0] == 0 else item[0], reverse=True))
                self.Data[shape][x] = dict(sorted(self.Data[shape][x].items(), key=lambda item: float('inf') if item[0] == 0 else item[0], reverse=True))
            self.Data[shape] = dict(sorted(self.Data[shape].items(), key=lambda item: float('inf') if item[0] == 0 else item[0], reverse=True))

    def _classify_folder(self, subfolder):
        # 注意顺序：先判四段，再判三段字母，再判三段数字，最后判两段
        m_rot = re.fullmatch(r"(\d+)-(\d+)-(\d+)-(R0|up|down)", subfolder)
        m_S = re.fullmatch(r"(\d+)-(\d+)-([A-Z]+)", subfolder)
        m_end = re.fullmatch(r"(\d+)-(\d+)-(\d+)", subfolder)
        m_circ = re.fullmatch(r"(\d+)-(\d+)", subfolder)

        if m_rot:
            x, y, z = map(int, m_rot.groups()[:3])
            suffix = m_rot.group(4)
            if suffix == "R0":
                folder_type = "rot0"
            else:
                folder_type = suffix  # "up" 或 "down"

        elif m_S:
            x, y = map(int, m_S.groups()[:2])
            z = None
            suffix = m_S.group(3)
            if suffix == "S":
                folder_type = "S"
            else:
                print(f"Unsupported suffix in subfolder: {subfolder}")

        elif m_end:
            x, y, z = map(int, m_end.groups())
            folder_type = "end"

        elif m_circ:
            x, y = map(int, m_circ.groups())
            z = None
            folder_type = "circ"

        else:
            print(f"Invalid subfolder format: {subfolder}")
            x, y, z, folder_type = None, None, None, None
        
        return x, y, z, folder_type
    
    def _pick_first_nonempty_csv(self, file_list):
        """
        从候选 CSV 列表中选择第一个真正包含数据行的文件。
        若所有文件都只有表头或无法读取，则返回 None。
        """
        for file_path in sorted(file_list):
            try:
                df_tmp = pd.read_csv(file_path)
                if not df_tmp.empty:
                    return file_path
                else:
                    logging.warning(f"跳过只有表头无数据的文件: {file_path}")
            except Exception as e:
                logging.warning(f"读取文件失败，已跳过: {file_path}, error={e}")
        return None

    def _edit_radii(self, folder_type, x, y, z):
        """
        当某个 subfolder 最终没有任何有效 channel、因此被整体判定为无效时，
        同步从 self.radii 中撤销该 subfolder 对应的半径配置。

        步骤：
        1. 根据 folder_type 判断当前 subfolder 对应的是二级还是三级半径结构。
        2. 若 folder_type 为 "circ" 或 "S"，则从 self.radii[x][folder_type] 中移除 y。
        3. 若 folder_type 为 "end"、"rot0"、"up" 或 "down"，则从 self.radii[x][folder_type][y] 中移除 z。
        4. 若三级结构中的某个 y 对应容器在删除后为空，则进一步删除该 y 键。

        输入：
        - folder_type: 子文件夹对应的数据类型，可为 "circ"、"S"、"end"、"rot0"、"up" 或 "down"
        - x: 实验方向，对应 self.radii 的一级 key
        - y: 二级半径键或半径值
        - z: 三级结构中的末级半径值；对 "circ" 与 "S" 类型可为 None

        输出：
        - 无显式返回
        - 原地修改 self.radii

        说明：
        - 本方法只在调用者已决定“整个 subfolder 无效”时执行撤销
        - 本方法支持 self.radii 中相关容器为 list 或 set
        """
        def _remove_from_container(container, value):
            if isinstance(container, set):
                container.discard(value)
            elif isinstance(container, list):
                while value in container:
                    container.remove(value)

        if folder_type in ("circ", "S"):
            # 1. x-y     -> self.radii['circ'] 去掉 y
            # 2. x-y-S   -> self.radii['S']    去掉 y
            if folder_type in self.radii[x]:
                _remove_from_container(self.radii[x][folder_type], y)
                
        elif folder_type in ("end", "rot0", "up", "down"):
            # 3. x-y-z       -> self.radii['end'][y]   去掉 z
            # 4. x-y-z-R0    -> self.radii['rot0'][y]  去掉 z
            # 5. x-y-z-up    -> self.radii['up'][y]    去掉 z
            # 6. x-y-z-down  -> self.radii['down'][y]  去掉 z
            if folder_type in self.radii[x]:
                # 如果 key 用的是字符串，可以把下面的 y_key 改成 str(y)
                y_key = y if y in self.radii[x][folder_type] else str(y) if str(y) in self.radii[x][folder_type] else None

                if y_key is not None:
                    _remove_from_container(self.radii[x][folder_type][y_key], z)

                    # 可选：如果删完后这个 y 对应的列表/集合空了，就把这个键也删掉
                    if len(self.radii[x][folder_type][y_key]) == 0:
                        del self.radii[x][folder_type][y_key]

    def _process_centreData(self, data):
        """
        将包含中心光纤与外围光纤的通道数据拆分，并生成中心补偿后的结果。

        步骤：
        1. 若输入为字典，则递归处理其各级 value，并保持原有嵌套结构不变。
        2. 若输入为 numpy 数组，则默认最后一个维度的第 1 列为中心光纤通道，其余列为外围光纤通道。
        3. 对数组输入：
            - 取第 1 列作为 centre_Data；
            - 取其余列作为 outer_Data；
            - 计算 processed_Data = outer_Data - centre_Data；
            - 构造 plot_Data = [centre_Data, outer_Data] 的列拼接结果。

        输入：
        - data:
            - 若为字典，则其叶子节点应为 numpy 数组；
            - 若为 numpy 数组，则默认第 1 列为中心光纤通道，其余列为外围光纤通道

        输出：
        - 若输入为字典，返回：
            - centre_Data_dict
            - outer_Data_dict
            - processed_Data_dict
            - plot_Data_dict
        - 若输入为 numpy 数组，返回：
            - centre_Data: 形状为 (..., 1) 的中心光纤数据
            - outer_Data: 形状为 (..., num_outer_channels) 的外围光纤数据
            - processed_Data: 中心补偿后的外围光纤数据
            - plot_Data: 由 centre_Data 与 outer_Data 按最后一个维度拼接得到的数据

        说明：
        - 本方法不修改输入 data，而是返回拆分及补偿后的结果
        - 当前实现使用第 1 列作为中心光纤通道
        """
        if isinstance(data, dict):
            centre_Data_dict = {}
            outer_Data_dict = {}
            processed_Data_dict = {}
            plot_Data_dict = {}
            for k, v in data.items():
                centre_Data_dict[k], outer_Data_dict[k], processed_Data_dict[k], plot_Data_dict[k] = self._process_centreData(data=v)
            return centre_Data_dict, outer_Data_dict, processed_Data_dict, plot_Data_dict
        elif isinstance(data, np.ndarray):
            # 使用 ... 适配任意维度，[0] 保持维度以便于后续广播相减
            centre_Data = data[..., [0]]
            outer_Data = data[..., 1:]
            processed_Data = outer_Data - centre_Data
            # 在最后一个维度上进行拼接
            plot_Data = np.concatenate([centre_Data, outer_Data], axis=-1)
            return centre_Data, outer_Data, processed_Data, plot_Data

    # -------------------- LUNA数据处理 --------------------
    def read_data_OFDR(self, folder_path):
        """
        从 LUNA/ODiSI 导出的多方向静态标定数据中读取、分组、预处理并重组通道数据。

        约定：
        - 数据目录按方向划分，即 folder_path 下包含多个以 direct 命名的子目录
        - 每个方向目录中存放若干 `_full.tsv` 文件
        - 文件名格式应匹配：
        ODiSI_xxxx_YYYY-MM-DD_HH-MM-SS_chX_full.tsv
        或
        ODiSI_xxxx_YYYY-MM-DD_HH-MM-SS_chX_gages_full.tsv
        - 同一时间戳下的多个通道文件视为同一组实验数据

        步骤：
        1. 遍历每个实验方向 direct 对应的子目录。
        2. 按文件名中的时间戳对 tsv 文件分组，每组对应一次采集。
        3. 从文件名解析物理通道号 phy_ch，并只保留激活通道对应的文件。
        4. 对每个物理通道、每个时间组：
            - 读取 tsv 数据；
            - 选取 NaN 最少的前 self.N 行求列均值；
            - 对缺失值做线性插值；
            - 用倍率阈值法修正局部突变点。
        5. 根据 self.sensing_ranges 提取各通道统一长度的有效传感区间。
        6. 对提取后的曲线执行平滑计算；当前版本仅计算平滑结果，不将其写入 self.Data。
        7. 先按物理通道升序组织成矩阵，再用 self.ch_order 重排到 active_channels 的输入顺序。
        8. 将排序后的时间戳依次映射到当前方向下的实验半径列表 self.radii[direct]['circ']。
        9. 最终写入 self.Data["circ"][direct][radius]。

        输入：
        - folder_path: LUNA 数据总目录路径，其下应包含以方向命名的子目录

        输出：
        - 无显式返回
        - 更新以下成员：
            - self.Data["circ"][direct][radius]
            - 值为形状为 (num_points, self.num_channels) 的 numpy 数组，单位缩放为 1e-6

        异常：
        - 当某个方向目录不存在时抛出 FileNotFoundError
        - 当某个物理通道缺少 OFDR 文件时抛出 ValueError
        - 当某物理通道缺少 sensing_ranges 配置时抛出 ValueError
        - 当某时间戳对应数组长度不足以完成切片时抛出 ValueError
        - 当某方向的时间组数超过当前方向预设半径数时抛出 ValueError

        说明：
        - 当前实现会计算 smoothed_data，但最终写入 self.Data 的仍是未平滑的 selected_interp_data
        - 最终列顺序语义与 read_data_FBG 保持一致
        """
        self.Data = {'circ': {}}

        physical_channels_sorted = sorted(self.active_channels.keys())

        for direct in self.exp_directs:
            file_folder = f"{folder_path}/{direct}"
            directory = Path(file_folder)

            if not directory.exists():
                raise FileNotFoundError(f"目录不存在: {directory}")

            pattern = re.compile(
                r'^ODiSI_\d+_\d{4}-\d{2}-\d{2}_(\d{2}-\d{2}-\d{2})_ch(\d+)(?:_gages)?_full$',
                re.IGNORECASE
            )

            # time_part -> {phy_ch: file_path}
            groups = defaultdict(dict)
            tsv_files = sorted(directory.glob('*_full.tsv'))

            for file in tsv_files:
                filename = file.stem
                match = pattern.match(filename)
                if not match:
                    logging.warning(f"文件名不符合预期格式，无法分组: {file.name}")
                    continue

                time_part = match.group(1)
                phy_ch = int(match.group(2))

                if phy_ch in self.active_channels:
                    groups[time_part][phy_ch] = file

            if len(groups) != len(self.radii['circ']):
                logging.warning(
                    f"预期分为 {len(self.radii['circ'])} 组，但实际分为 {len(groups)} 组。"
                )

            # grouped_data[time_part][phy_ch] = dataframe
            grouped_data = {}
            for group, file_map in groups.items():
                dataframes = {}
                for phy_ch, file in file_map.items():
                    try:
                        df = pd.read_csv(
                            file,
                            skiprows=31,
                            header=None,
                            sep='\t',
                            encoding='utf-8',
                            on_bad_lines='skip'
                        )
                        dataframes[phy_ch] = df
                    except Exception as e:
                        logging.error(f"错误: 读取文件 {file.name} 时发生错误: {e}")
                grouped_data[group] = dataframes

            interp_data = {}

            # 这里统一用“物理通道号”作为 key，与 self.sensing_ranges 对齐
            for phy_ch in physical_channels_sorted:
                interp_data_ch = {}

                for specific_time in sorted(grouped_data.keys()):
                    if phy_ch not in grouped_data[specific_time]:
                        raise ValueError(
                            f"方向 {direct}、时间 {specific_time} 缺少物理通道 {phy_ch} 的 OFDR 文件。"
                        )

                    ch_data = grouped_data[specific_time][phy_ch]
                    all_frame_data = ch_data.iloc[2:, 3:]

                    frame_data = self._process_dataframe(df=all_frame_data)
                    frame_data_interp = pd.Series(frame_data).interpolate(
                        method='linear',
                        limit_direction='both'
                    )
                    processed_frame_data_interp = self._remove_outliers_by_ratio(
                        array=frame_data_interp.to_numpy()
                    )

                    interp_data_ch[specific_time] = processed_frame_data_interp

                interp_data[phy_ch] = interp_data_ch

            selected_interp_data = self._extract_sensing_ranges(interp_data=interp_data)

            # 保留你原来的平滑流程；如果你想让最终 self.Data 使用平滑后的数据，
            # 只需把下面的 data_source 改成 smoothed_data。
            smoothed_data = {}
            for group_key, curves in selected_interp_data.items():
                smoothed_data[group_key] = {}
                for key, y in curves.items():
                    smoothed_data[group_key][key] = self._smooth_curve(y=y)

            # 为了最小改动，这里仍沿用你旧版的“未平滑结果写入 self.Data”行为
            data_source = selected_interp_data

            if not data_source:
                self.Data["circ"][direct] = {}
                continue

            timestamps = sorted(next(iter(data_source.values())).keys())

            if len(timestamps) > len(self.radii['circ']):
                raise ValueError(
                    f"方向 {direct} 的时间组数 {len(timestamps)} 超过预设半径数 {len(self.radii['circ'])}。"
                )

            timestamp_to_exp_samps = {
                timestamp: self.radii['circ'][idx]
                for idx, timestamp in enumerate(timestamps)
            }

            restructured_data = {}
            for timestamp in timestamps:
                if timestamp not in timestamp_to_exp_samps:
                    raise KeyError(f"Timestamp {timestamp} not found in timestamp_to_exp_samps")

                new_key = timestamp_to_exp_samps[timestamp]

                # 先按物理通道升序组装，语义与 read_data_FBG 的原始 data_matrix 保持一致
                data_matrix = np.column_stack(
                    [data_source[phy_ch][timestamp] for phy_ch in physical_channels_sorted]
                ) * 1e-6

                # 再使用现有 read_data_FBG 同语义的 self.ch_order 重排到 active_channels 输入顺序
                if isinstance(data_matrix, np.ndarray):
                    data_matrix = data_matrix[:, self.ch_order]
                else:
                    data_matrix = [[row[i] for i in self.ch_order] for row in data_matrix]

                restructured_data[new_key] = data_matrix

            self.Data["circ"][direct] = restructured_data

    def _process_dataframe(self, df):
        """
        处理 DataFrame：选出包含 NaN 值最少的前 N 行，并计算每列的平均值。
        """
        # 计算每行 NaN 值的数量
        nan_counts = df.isna().sum(axis=1)
        
        # 按 NaN 值数量排序，选出包含 NaN 值最少的前 N 行
        rows_with_least_nan = df.loc[nan_counts.nsmallest(self.N).index]
        
        # 对每列计算平均值（跳过 NaN）
        column_means = rows_with_least_nan.mean()
        
        return column_means
    
    def _remove_outliers_by_ratio(self, array, ratio_threshold=5):
        """
        按与相邻值的相对倍率关系修正一维数组中的局部突变点。

        步骤：
        1. 创建输入数组的副本作为输出数组。
        2. 遍历数组中除首尾之外的各个位置。
        3. 若当前位置与前后相邻值的差值都超过 ratio_threshold 倍的相邻值绝对值，
        则将该位置视为局部突变点。
        4. 将突变点替换为前后相邻值的均值。

        输入：
        - array: 输入的一维数组
        - ratio_threshold: 判定局部突变点的倍率阈值；例如 5 表示差值超过相邻值绝对值的 5 倍时判为异常

        输出：
        - corrected_array: 修正后的一维数组，长度与输入相同

        说明：
        - 本方法不会删除数组元素，只会对满足条件的内部点做替换修正
        - 首尾两个元素不会被处理
        """
        corrected_array = array.copy()  # 创建副本以进行修改

        for i in range(1, len(array) - 1):  # 遍历数组，避开首尾两个元素
            prev_value = array[i - 1]  # 前一个值
            next_value = array[i + 1]  # 后一个值

            # 判断当前值是否为突变点（与前后值的差值超过 ratio_threshold 倍）
            if (abs(array[i] - prev_value) > ratio_threshold * abs(prev_value)) and \
            (abs(array[i] - next_value) > ratio_threshold * abs(next_value)):
                # 如果是突变点，替换为相邻值的均值
                corrected_array[i] = (prev_value + next_value) / 2

        return corrected_array
    
    def _extract_sensing_ranges(self, interp_data):
        """
        从 interp_data 中按 self.sensing_ranges 指定的区间提取切片，并返回相同层级结构的新字典。

        步骤：
        1. 遍历 interp_data 中的每个物理通道。
        2. 根据 self.sensing_ranges[channel] 读取该通道对应的 start 与 end。
        3. 对该通道下的每个时间戳对应数组执行切片 array[start:end]。
        4. 将切片结果写入新的二级字典中并返回。

        输入：
        - interp_data: 二级字典
            - 一级 key: 物理通道号
            - 二级 key: 时间戳（如 "05-37-36"）
            - value: 一维数组

        输出：
        - result: 与 interp_data 具有相同层级结构的二级字典
            - 一级 key: 物理通道号
            - 二级 key: 时间戳
            - value: 提取后的数组切片

        异常：
        - 当某物理通道在 self.sensing_ranges 中缺少配置时抛出 ValueError
        - 当某时间戳对应数组长度不足以完成切片时抛出 ValueError
        """
        result = {}

        for channel, time_data in interp_data.items():
            # 获取当前 channel 的 start 和 end
            if channel not in self.sensing_ranges:
                raise ValueError(f"Missing range for {channel} in ranges dictionary.")
            start, end = self.sensing_ranges[channel]
            

            # 初始化存储当前通道的结果
            result[channel] = {}

            # 遍历时间戳，提取指定范围的切片
            for timestamp, array in time_data.items():
                if len(array) < end:
                    raise ValueError(f"Array under timestamp {timestamp} in {channel} is too short for range {start}:{end}.")
                # 提取切片并存储到新的字典中
                result[channel][timestamp] = array[start:end]

        return result
    
    def _smooth_curve(self, y):
        """
        对单条一维曲线做 Savitzky-Golay 平滑。

        处理流程：
        1. 从 self.smoothing_params 中读取 window_length 与 polyorder。
        2. 根据输入曲线长度对 window_length 做有限修正，确保：
        - 不超过输入长度；
        - 尽量保持为奇数。
        3. 对输入曲线执行平滑滤波并返回结果。

        输入：
        - y: 一维数组，表示某一通道在空间上的应变/信号曲线

        输出：
        - 平滑后的一维 numpy 数组，长度与输入一致

        说明：
        - 该方法仅负责单条曲线的平滑，不负责异常值处理与插值
        - 平滑参数由 self.smoothing_params 统一控制
        - 当前实现仅自动修正 window_length，不额外修正 polyorder
        """
        window_length = self.smoothing_params['window_length']
        polyorder = self.smoothing_params['polyorder']
        window_length = min(len(y) - 1 if len(y) % 2 == 0 else len(y), window_length)
        if window_length % 2 == 0:
            window_length += 1
        return savgol_filter(y, window_length, polyorder)

        
    # -------------------- FiberTecc动态数据接收 --------------------
    def receive_data_FiberTecc(self):
        """
        持续从解调仪 TCP 端口接收动态数据，并按设定频率推送处理后的数据帧到队列中。

        处理流程：
        1. 创建 TCP socket，并连接到 self.IP:self.PORT。
        2. 持续接收网络字节流，将其追加到 self.buffer。
        3. 调用 _process_buffer() 对缓存中的完整数据包进行解析与路由。
        4. 按 self.data_freq 控制推送频率，周期性调用 _prep_frame() 生成当前帧。
        5. 若队列已满，则丢弃最旧数据，再将新帧压入 self.data_queue。
        6. 当连接断开或发生异常时退出循环，并关闭 socket。

        输入：
        - 无（使用实例中的网络配置与缓存状态）

        输出：
        - 无直接返回值
        - 副作用：
        - 更新 self.buffer、self.current_snapshot、self.zero_points、self.is_zero_set
        - 将形状为 (1, self.total_points, self.num_channels) 的数据帧放入 self.data_queue

        异常：
        - 网络通信、连接中断、解析等异常会被捕获并打印
        - 无论是否异常退出，最终都会关闭 socket 连接

        说明：
        - 该方法通常运行于独立线程中，负责持续接收数据
        - 实际数据解析与预处理分别由 _process_buffer() 和 _prep_frame() 完成
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536) # 增大接收缓冲
            sock.connect((self.IP, self.PORT))
            print(f"已连接到解调仪 {self.IP}:{self.PORT}")
            print(f"激活通道配置: {self.active_channels}")

            last_push_time = time.time()        
            while True:
                # 接收网络数据
                try:
                    chunk = sock.recv(4096)
                    if not chunk: 
                        break
                    self.buffer.extend(chunk)
                except Exception:
                    break
                
                # 处理粘包
                self._process_buffer()

                # 推送数据
                now = time.time()
                if now - last_push_time >= 1 / self.data_freq:
                    data = self._prep_frame()
                    if self.data_queue.full():
                        try: 
                            self.data_queue.get_nowait()
                        except: 
                            pass
                    self.data_queue.put(data)
                    last_push_time = now

        except Exception as e:
            print(f"通信线程异常: {e}")
        finally:
            if sock: sock.close()

    def _process_buffer(self):
        """
        从接收缓存中解析完整数据包，并将解析出的传感器数据路由到对应通道与采样点。

        数据包约定：
        - 缓存前 4 字节为无符号整数，表示后续包体长度 N
        - 后续 N 字节为一个完整 packet body
        - packet body 的具体解析由 _parse_packet() 完成

        处理流程：
        1. 检查 self.buffer 中是否至少包含 4 字节包头。
        2. 读取包体长度 N，判断当前缓存是否已包含完整数据包。
        3. 若完整，则截取 packet body，并从缓存中移除该包。
        4. 调用 _parse_packet() 得到 (sensor_id, wavelength) 列表。
        5. 对每个传感器数据：
        - 解析物理通道号 phy_ch 与传感器索引 idx；
        - 检查是否属于激活通道且索引合法；
        - 将数据写入 self.current_snapshot 对应位置。
        6. 若某通道尚未设置归零基准，则逐点填充 zero_points；
        当该通道所有点均已初始化后，标记 is_zero_set[phy_ch] = True。

        输入：
        - 无（使用 self.buffer 中当前缓存的字节流）

        输出：
        - 无直接返回值
        - 副作用：
        - 消费 self.buffer 中已完整到达的数据包
        - 更新 self.current_snapshot
        - 可能更新 self.zero_points 与 self.is_zero_set

        说明：
        - 该方法可处理粘包情况，即一次接收中包含多个完整包
        - 对不完整数据包会保留在缓存中，等待后续字节到达
        """
        # 处理粘包
        while len(self.buffer) >= 4:
            N = struct.unpack('<I', self.buffer[0:4])[0]
            if len(self.buffer) < 4 + N: 
                break
            packet_body = self.buffer[4 : 4 + N]
            self.buffer = self.buffer[4 + N:]

            # 2. 解析数据
            sensor_list = self._parse_packet(packet_body)
            
            # 3. 数据路由 (Data Routing) - 列表驱动
            for sensor_id, wavelength in sensor_list:
                sid_str = str(sensor_id)
                if len(sid_str) < 3: 
                    continue
                try:
                    phy_ch = int(sid_str[0])    # 物理通道号 
                    idx = int(sid_str[1:]) - 1  # 传感器索引 (0~15)
                except: 
                    continue
                
                # 检查是否在白名单中
                if phy_ch in self.active_channels and 0 <= idx < self.total_points:
                    # 找到对应的列
                    col = self.active_channels[phy_ch]
                    
                    # 更新快照
                    self.current_snapshot[idx, col] = wavelength
                    
                    # 4. 归零基准设置 (首次收到有效数据时)
                    if not self.is_zero_set[phy_ch]:
                        if self.zero_points[phy_ch][idx] is None:
                            self.zero_points[phy_ch][idx] = wavelength
                        
                        # 检查是否该通道已满
                        if all(v is not None for v in self.zero_points[phy_ch]):
                            self.is_zero_set[phy_ch] = True
                            print(f"物理通道 {phy_ch} 基准已设置")
        
    def _parse_packet(self, packet_body):
        """
        解析单个数据包包体，提取其中包含的传感器编号与数据。

        数据格式约定：
        - 包体前 2 字节表示传感器数量 sensor_count
        - 此后每个传感器块固定占 20 字节
        - 其中：
        - 偏移 2:4 为传感器 ID（unsigned short）
        - 偏移 16:20 为数据（float）

        处理流程：
        1. 读取包体中的 sensor_count。
        2. 按固定 20 字节步长遍历各个传感器块。
        3. 对每个块提取 sensor_id 与 wavelength。
        4. 将结果存储为 (sensor_id, wavelength) 元组列表返回。
        5. 若包体长度不足以支撑完整块，则停止解析剩余部分。

        输入：
        - packet_body: 单个完整数据包的包体字节串

        输出：
        - sensor_data_list: 列表，元素为 (sensor_id, wavelength) 二元组

        异常：
        - 若解析过程中发生异常，则打印错误信息并返回空列表

        说明：
        - 本方法只负责二进制解包，不负责通道映射、索引检查与数据写入
        - 后续的数据路由由 _process_buffer() 完成
        """
        try:
            sensor_count = struct.unpack('<H', packet_body[0:2])[0]
            sensor_data_list = []
            offset = 2

            for _ in range(sensor_count):
                if offset + 20 > len(packet_body): 
                    break
                sensor_block = packet_body[offset : offset + 20]
                # 偏移2-3: ID, 偏移16-19: 数据
                sensor_id = struct.unpack('<H', sensor_block[2:4])[0]
                wavelength = struct.unpack('<f', sensor_block[16:20])[0]
                sensor_data_list.append((sensor_id, wavelength))
                offset += 20

            return sensor_data_list
        
        except Exception as e:
            print(f"解析错误: {e}")
            return []

    def _prep_frame(self):
        """
        根据当前快照与归零基准，生成可供下游使用的单帧动态数据。

        处理流程：
        1. 初始化形状为 (self.total_points, self.num_channels) 的零矩阵。
        2. 遍历所有激活物理通道 phy_ch：
        - 若该通道已完成归零：
            1) 读取当数据 raw_vals；
            2) 读取对应基准值 base_vals；
            3) 计算差值 (raw_vals - base_vals) * 1000.0；
            4) 写入对应列。
        - 若尚未归零，则该通道整列保持为 0。
        3. 将结果 reshape 为 (1, self.total_points, self.num_channels) 并返回。

        输入：
        - 无（使用 self.current_snapshot、self.zero_points、self.is_zero_set）

        输出：
        - frame_to_send: numpy 数组，形状为 (1, self.total_points, self.num_channels)

        说明：
        - 返回结果中的数值表示相对于归零基准的变化量
        - 当前实现中使用乘以 1000.0 的缩放方式，具体物理意义取决于解调仪输出单位与后续处理约定
        - 该方法不会修改内部状态，仅根据当前缓存状态生成一帧数据
        """
        # 准备发送的数据帧 (1, self.total_points, self.num_channels)
        frame_to_send = np.zeros((self.total_points, self.num_channels))
        
        # 遍历白名单列表，按顺序填充数据
        for phy_ch in self.active_channels:
            i = self.active_channels[phy_ch]
            if self.is_zero_set[phy_ch]:
                raw_vals = self.current_snapshot[:, i]
                base_vals = np.array(self.zero_points[phy_ch]) # 处理可能存在的None。虽然is_zero_set为True理论上不会有None，但为了robustness，这里假设一旦set为True，base_vals全为float
                diff = (raw_vals - base_vals) * 1000.0 # 向量化计算: (当前值 - 基准值) * 1000
                frame_to_send[:, i] = diff
            else:
                # 未归零前发送0
                frame_to_send[:, i] = 0.0

        # 构造成 (1, self.total_points, self.num_channels) 的形状
        return frame_to_send.reshape(1, self.total_points, self.num_channels)
    
    # -------------------- Hippo动态数据接收 --------------------
    def receive_data_Hippo(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(3.0)
        sock.sendto(b'\x01\x02', (self.IP, self.PORT))
        sock.recv(2048)
        sock.sendto(b'\x01\x0A\x55', (self.IP, self.PORT))
        time.sleep(0.05)

        while True:
            try:
                wave8x16 = self._recv_010c_once(sock)          # (8,16) nm
            except socket.timeout:
                continue

            # data_received = {ch: wave8x16[i] for i, ch in enumerate(self.active_channels)}
            data_received = {phy_ch: wave8x16[phy_ch-1] for phy_ch in self.active_channels}

            for ch in self.active_channels:
                if self.zero_points[ch][0] is None:
                    self.zero_points[ch] = data_received[ch]
                    print(f"通道{ch}归零基准已设置")

            data = np.empty((1, 16, self.num_channels))
            for phy_ch, i in self.active_channels:
                data[0, :, i] = (data_received[phy_ch] - self.zero_points[phy_ch]) * 1000

            if self.data_queue.full():
                self.data_queue.get()
            self.data_queue.put(data)   

    def _recv_010c_once(self, sock):
        while True:
            data, _ = sock.recvfrom(2048)
            if data[:2] == b'\x01\x0C':
                break
        gratings = []
        off = 2
        while off < len(data):
            ch, m = data[off], data[off+1]
            off += 4
            waves = []
            for _ in range(m):
                w = struct.unpack('<I', data[off:off+4])[0] / 1000.0   # pm → nm
                waves.append(w)
                off += 4
            gratings.append((ch, m, waves))
        wave2d = np.zeros((8, 16))
        for ch, m, waves in gratings:
            if ch < 8 and waves:
                wave2d[ch, :len(waves)] = waves[:16]
        return wave2d