"""可视化接口；只在调用视图时导入 matplotlib/VTK。"""
class Visualizer:
    """延迟创建包内绘图实现或调用方提供的绘图工厂。"""
    def __init__(self, context, legacy_factory=None):
        self.context=context; self._legacy=None; self._factory=legacy_factory
    def _get(self):
        if self._legacy is None:
            if self._factory is None:
                from .plots import Visualisor
                self._factory=Visualisor
            self._legacy=self._factory(recon=self.context.engine)
        return self._legacy
    @property
    def legacy(self): return self._get()
    def show_reconstruction(self,*args,**kwargs): return self._get().show_reconstruction(*args,**kwargs)
