"""
配置加载与合并工具模块。

本模块提供加载配置文件（YAML/JSON）以及合并配置的功能。
适用于深度学习实验管理，支持基础配置与实验配置的覆盖合并。

假设项目根目录为 UAV/，则导入方式推荐：
    from src.utils.config import load_config, merge_configs

示例配置 (YAML):
    data:
      dataset_name: "uav_bearing_fault_v1"
      augmentation:
        - "random_crop"
        - "normalize"
    model:
      backbone: "resnet18"
      head: "classifier"
    strategy:
      name: "transfer_learning"
    train:
      batch_size: 32
      learning_rate: 0.001
"""

import json
from pathlib import Path
from typing import Any, Dict, Mapping

# 尝试导入 PyYAML，如果未安装则提示
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def load_config(path: str) -> Dict[str, Any]:
    """
    加载指定路径的配置文件（支持 .yaml, .yml, .json）。

    Args:
        path (str): 配置文件路径。

    Returns:
        Dict[str, Any]: 加载后的配置字典。

    Raises:
        FileNotFoundError: 如果文件不存在。
        ValueError: 如果文件格式不支持。
        ImportError: 如果加载 YAML 但未安装 pyyaml。
        TypeError: 如果配置文件顶层不是字典。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {path}")

    suffix = file_path.suffix.lower()
    data = None

    if suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif suffix in ('.yaml', '.yml'):
        if yaml is None:
            raise ImportError("加载 YAML 文件需要安装 PyYAML: pip install pyyaml")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {suffix}。仅支持 .json, .yaml, .yml")

    if data is None:
        return {}
    
    if not isinstance(data, dict):
        raise TypeError(f"配置文件内容必须是字典/映射类型，当前为: {type(data)}")

    return data


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并两个配置字典。override 中的值会覆盖 base 中的值。
    
    合并策略：
    1. 如果 key 在 override 中存在且对应的值也是字典，并且 base 中该 key 对应的值也是字典，则递归合并。
    2. 否则，直接使用 override 中的值覆盖 base。
    3. 如果 key 仅在 base 中存在，则保留 base 的值。

    Args:
        base (Dict[str, Any]): 基础配置。
        override (Dict[str, Any]): 覆盖配置（实验配置）。

    Returns:
        Dict[str, Any]: 合并后的新配置字典（不修改原字典）。
    """
    if not isinstance(base, Mapping) or not isinstance(override, Mapping):
        raise TypeError("merge_configs 的参数必须是字典类型")

    result = dict(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            # 递归合并
            result[key] = merge_configs(dict(result[key]), dict(value))
        else:
            # 覆盖
            result[key] = value
            
    return result
