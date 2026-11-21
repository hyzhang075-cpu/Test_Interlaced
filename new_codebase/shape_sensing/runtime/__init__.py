"""实时演示入口；仅运行时加载可选 Qt/VTK 依赖。"""
class RealtimeSystem:
    """按需创建演示窗口并运行其 Qt 生命周期。"""
    def __init__(self, context, **kwargs): self.context=context; self.kwargs=kwargs; self._demo=None
    def run(self):
        """按需导入 Qt/VTK Demonstrator 并返回创建的窗口。"""
        from .compat import Demonstrator
        self._demo=Demonstrator(recon=self.context.engine,**self.kwargs)
        return self._demo
    def close(self):
        if self._demo is not None: self._demo.close()

from .service import RuntimeService, RuntimeFrame, SessionState, FrameSource, FrameProcessor, run_demo
