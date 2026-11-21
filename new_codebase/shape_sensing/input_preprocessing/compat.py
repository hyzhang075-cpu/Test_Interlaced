# 历史类兼容入口集中于本模块；新生产入口请使用 shape_sensing.input_preprocessing.InputService。
"""旧Sensor调用边界；公式、读取和预处理仅存在于本模块InputKernel。"""
from ._kernel import *

class Sensor(InputKernel):
    """兼容历史构造即加载；生产模块使用InputService显式load_data。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_data(self.folder_path if hasattr(self, 'folder_path') else (kwargs.get('folder_path') if 'folder_path' in kwargs else args[3]))

# Keep the old documented constructor signature while delegating all work.
import inspect
Sensor.__init__.__signature__ = inspect.signature(InputKernel.__init__)
