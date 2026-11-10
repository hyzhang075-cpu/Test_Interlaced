"""离线与实时输入适配。"""
class InputPreprocessor:
    """按解调仪选择旧读取/接收函数；预处理仅在显式调用时改变引擎数据状态。

    离线读取路径参数沿用旧 Sensor（folder_path、传感点坐标，单位 mm）；
    输出形状和波长/应变单位保持旧实现约定。实时 receive 会产生设备 I/O。
    """
    def __init__(self, context): self.context=context
    @property
    def engine(self): return self.context.engine
    def read(self, *args, **kwargs):
        """调用对应模式的离线读取函数；未知 interrogator 抛 ValueError。"""
        mode=getattr(self.engine,'interrogator',None)
        method_name={'FiberTecc':'read_data_FBG','Hippo':'read_data_FBG','LUNA':'read_data_OFDR'}.get(mode)
        if method_name is None: raise ValueError(f'不支持的离线解调仪: {mode!r}')
        method=getattr(self.engine,method_name)
        return method(*args,**kwargs)
    def receive(self, *args, **kwargs):
        """显式启动对应的旧实时接收方法；会访问解调仪。"""
        mode=getattr(self.engine,'interrogator',None)
        method_name={'FiberTecc':'receive_data_FiberTecc','Hippo':'receive_data_Hippo'}.get(mode)
        if method_name is None: raise ValueError(f'{mode!r} 不支持在线接收')
        return getattr(self.engine,method_name)(*args,**kwargs)
    def preprocess(self,data,*args,**kwargs):
        """复用旧中心纤维/多通道预处理；输入 data 的布局沿用 Sensor.Data。"""
        for name in ('_process_centreData','_process_Data','_process_data'):
            method=getattr(self.engine,name,None)
            if method: return method(data,*args,**kwargs)
        raise AttributeError('旧 Sensor 没有可调用的数据预处理函数')
