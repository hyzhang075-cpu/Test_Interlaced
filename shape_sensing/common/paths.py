"""包资源与用户相对路径的统一解析。"""
from pathlib import Path
from importlib.resources import files

def working_root():
    """返回命令调用目录；配置文件中的相对路径由配置加载器单独解析。"""
    return Path.cwd()

def resource_path(*parts):
    """返回已安装包内资源的读取句柄。"""
    return files('shape_sensing').joinpath(*parts)
