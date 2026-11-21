"""严格 YAML 工作配置与参数目录。"""
from pathlib import Path
from typing import Any
import json, math
import yaml

class _UniqueLoader(yaml.SafeLoader): pass

def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result: raise ValueError(f"YAML 存在重复键: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result
_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
_CATALOG = json.loads((Path(__file__).with_name('parameters.json')).read_text(encoding='utf-8'))
CONTROL_KEYS = {'task', 'config', 'resume', 'sensor_params', 'schema_version'}
SCHEMA_VERSION = 1
_REQUIRED_DESIGN = {'sensor_id', 'R', 'samp_dist_preset'}

def parameter_catalog():
    """返回 main 原 parser 的110项参数目录，包含类型、解释、choices 和默认值。"""
    return json.loads(json.dumps(_CATALOG))

def _type_ok(value, kind, name=None):
    if value is None: return True
    if name == 'rot_dir':
        # 兼容方向角字典、按标定形状排列的旧列表和统一旋转符号。
        return isinstance(value, (dict, list, tuple)) or (
            type(value) in (int, float) and math.isfinite(value))
    if kind in ('bool', '_str2bool') or (kind in (None, 'None') and isinstance(_CATALOG.get(name, {}).get('default'), bool)): return type(value) is bool
    if kind in ('float', '_finite_nonnegative_float'):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    if kind == 'int': return isinstance(value,int) and not isinstance(value,bool)
    if kind == 'str': return isinstance(value,str)
    if kind == 'list':
        default=_CATALOG.get(name,{}).get('default')
        if isinstance(default,dict): return isinstance(value,dict)
        return isinstance(value,list)
    if kind == 'dict':
        default=_CATALOG.get(name,{}).get('default')
        if isinstance(default,list): return isinstance(value,list)
        return isinstance(value,dict)
    if kind == 'literal_eval':
        expected = _CATALOG.get(name, {}).get('default')
        if expected is None: return isinstance(value,(list,dict,tuple))
        if isinstance(expected,dict): return isinstance(value,dict)
        if isinstance(expected,(list,tuple)): return isinstance(value,(list,tuple))
        return type(value) is type(expected)
    return True

def _validate_nested(name, value, template, path):
    if not isinstance(value,dict) or not isinstance(template,dict): return
    unknown=set(value)-set(template)
    if unknown: raise ValueError(f"未知配置字段 {path}.{sorted(unknown)[0]}")
    for key, child in value.items():
        if isinstance(child,dict) and isinstance(template.get(key),dict): _validate_nested(name,child,template[key],f'{path}.{key}')

def validate_working_config(data):
    """验证原参数名、类型、choices 和区域嵌套；显式 null/0/false 均保留。"""
    if not isinstance(data,dict): raise ValueError('配置文件根节点必须是映射')
    allowed=set(_CATALOG)|{'task','config','resume','schema_version'}
    unknown=set(data)-allowed
    if unknown: raise ValueError(f"未知配置字段: {sorted(unknown)[0]}")
    version=data.get('schema_version', SCHEMA_VERSION)
    if version != SCHEMA_VERSION: raise ValueError(f'不支持的 schema_version={version!r}，当前支持 {SCHEMA_VERSION}')
    missing=_REQUIRED_DESIGN-set(data)
    if missing: raise ValueError(f"缺少传感器设计必填字段: {', '.join(sorted(missing))}")
    empty={key for key in _REQUIRED_DESIGN if data.get(key) is None}
    if empty: raise ValueError(f"传感器设计必填字段不能为 null: {', '.join(sorted(empty))}")
    for name,value in data.items():
        meta=_CATALOG.get(name)
        if meta is None: continue
        kind=meta.get('type')
        if value is None and meta.get('default') is not None:
            raise ValueError(f'配置字段 {name} 不允许 null')
        if not _type_ok(value,kind,name): raise ValueError(f"配置字段 {name} 类型错误，期望 {kind or type(meta.get('default')).__name__}")
        choices=meta.get('choices')
        if value is not None and choices and value not in choices: raise ValueError(f"配置字段 {name} 必须为 {choices}")
        default=meta.get('default')
        if name=='calibration_regions' and value is not None:
            if not isinstance(value,dict): raise ValueError('calibration_regions 必须是映射')
            _validate_nested(name,value,default,'calibration_regions')
    def normalize_direction_keys(value, path):
        if not isinstance(value,dict): return value
        normalized={}
        for key,child in value.items():
            new_key=key
            if isinstance(key,str) and key.lstrip('-').isdigit(): new_key=int(key)
            if new_key in normalized: raise ValueError(f'{path} 方向键重复: {new_key}')
            normalized[new_key]=normalize_direction_keys(child,f'{path}.{new_key}')
        return normalized
    # YAML/JSON 中的物理通道号同方向角一样，统一为整数键。
    for key in ('weights','slopes','radii','sensing_ranges','rot_dir'):
        if key in data: data[key]=normalize_direction_keys(data[key],key)
    return dict(data)

def load_working_config(path: str | Path) -> dict[str, Any]:
    """安全读取并验证配置；相对资源路径以配置文件目录为基准解析。"""
    source=Path(path).resolve()
    with source.open('r',encoding='utf-8') as stream: data=yaml.load(stream,Loader=_UniqueLoader)
    data=validate_working_config(data)
    for key in ('calibration_data_root','calibration_output_dir','zero_reference_path'):
        value=data.get(key)
        if isinstance(value,str) and value and not Path(value).is_absolute(): data[key]=str((source.parent/value).resolve())
    root=Path(data.get('calibration_data_root', source.parent))
    value=data.get('folder_path')
    if isinstance(value,str) and value and not Path(value).is_absolute(): data['folder_path']=str((root/value).resolve())
    regions=data.get('calibration_regions')
    if isinstance(regions,dict):
        for region,settings in regions.items():
            if isinstance(settings,dict):
                value=settings.get('folder_path')
                if isinstance(value,str) and value and not Path(value).is_absolute(): settings['folder_path']=str((root/value).resolve())
    data['_config_path']=str(source)
    return data
