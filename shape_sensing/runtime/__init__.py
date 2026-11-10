"""实时动态系统的延迟适配。"""
class RealtimeSystem:
    """显式创建 Demonstrator；只在 run 调用时创建 GUI 对象，不在导入时连设备。

    动态曲线坐标单位和每帧数组维度沿旧 Reconstructor/Demonstrator 约定。
    """
    def __init__(self,context,**kwargs): self.context=context; self.kwargs=kwargs; self._demo=None
    def run(self):
        """创建并返回窗口对象；设备接收仍由调用者显式启动线程。"""
        from demonstrator import Demonstrator
        self._demo=Demonstrator(recon=self.context.engine,**self.kwargs)
        return self._demo
    def close(self):
        """若旧窗口提供 close，则关闭窗口并释放其资源。"""
        if self._demo is not None:
            method=getattr(self._demo,'close',None)
            if method: return method()
