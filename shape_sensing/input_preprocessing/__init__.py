"""输入预处理实现与当前任务上下文入口。"""
from .sensor import Sensor, WavelengthContinuityGuard
class InputPreprocessor:
    """通过任务上下文提供原始传感器读入和帧处理。"""
    def __init__(self, context): self.context=context
    @property
    def engine(self): return self.context.engine
    def read(self,*args,**kwargs):
        mode=getattr(self.engine,"interrogator",None)
        method_name={"FiberTecc":"read_data_FBG","Hippo":"read_data_FBG","LUNA":"read_data_OFDR"}.get(mode)
        if method_name is None: raise ValueError(f"不支持的解调仪: {mode!r}")
        return getattr(self.engine,method_name)(*args,**kwargs)
    def receive(self,*args,**kwargs):
        mode=getattr(self.engine,"interrogator",None)
        method_name={"FiberTecc":"receive_data_FiberTecc","Hippo":"receive_data_Hippo"}.get(mode)
        if method_name is None: raise ValueError(f"{mode!r} 不支持实时接收")
        return getattr(self.engine,method_name)(*args,**kwargs)
    def preprocess(self,data,*args,**kwargs):
        for name in ("_process_centreData","_process_Data","_process_data"):
            method=getattr(self.engine,name,None)
            if method: return method(data,*args,**kwargs)
        raise AttributeError("当前 Sensor 没有可调用的数据预处理实现")

__all__=["Sensor","WavelengthContinuityGuard","InputPreprocessor"]
