"""延迟创建旧绘图工具，导入时不加载 GUI。"""
class Visualizer:
    """显式绘图适配器；构造 legacy 窗口只在使用时发生。"""
    def __init__(self, context, legacy_factory=None): self.context=context; self._legacy=None; self._factory=legacy_factory
    def _get(self):
        if self._legacy is None:
            if self._factory is None:
                from visualisor import Visualisor
                self._factory=Visualisor
            self._legacy=self._factory(recon=self.context.engine)
        return self._legacy
    @property
    def legacy(self): return self._get()
    def show_reconstruction(self,*args,**kwargs): return self._get().show_reconstruction(*args,**kwargs)
