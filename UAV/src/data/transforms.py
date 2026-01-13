"""
通用增强注册表模块。

提供增强方法的注册机制和构建函数。
"""

from typing import Any, Callable, Dict, List

import torchvision.transforms as T

__all__ = ["AUG_REGISTRY", "register_augment", "build_augmentation"]

AUG_REGISTRY: Dict[str, Callable] = {}


def register_augment(name: str) -> Callable:
    """
    注册增强类的装饰器。

    Args:
        name (str): 增强方法的注册名称。

    Returns:
        Callable: 装饰器函数。
    """
    def decorator(cls: Callable) -> Callable:
        if name in AUG_REGISTRY:
            raise ValueError(f"增强方法 '{name}' 已被注册: {AUG_REGISTRY[name]}")
        AUG_REGISTRY[name] = cls
        return cls

    return decorator


def build_augmentation(aug_config_list: List[Dict[str, Any]]) -> Callable:
    """
    构建增强流水线。

    Args:
        aug_config_list (List[Dict[str, Any]]): 增强配置列表。
            每个元素形如: {"name": "signal_random_crop", "params": {"window_size": 2048}}

    Returns:
        Callable: 组合后的增强函数 (Compose)。
    
    Raises:
        ValueError: 如果配置中的增强名称未注册。
    """
    transforms_list = []
    for config in aug_config_list:
        name = config.get("name")
        if not name:
            continue
        
        params = config.get("params", {})
        if name not in AUG_REGISTRY:
            # 尝试动态导入以确保注册
            # 注意：实际使用时建议在外部显式导入 transforms_signal 和 transforms_image
            raise ValueError(f"增强方法 '{name}' 未找到。请确保已导入对应的增强模块。")
            
        transforms_list.append(AUG_REGISTRY[name](**params))

    return T.Compose(transforms_list)
