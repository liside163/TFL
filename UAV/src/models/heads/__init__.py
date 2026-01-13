"""
Head 注册表模块。

提供 Head 模型的注册和创建功能。
"""

from typing import Any, Callable, Dict, Optional, Type

import torch.nn as nn

__all__ = ["HEAD_REGISTRY", "register_head", "create_head"]

HEAD_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_head(name: str) -> Callable:
    """
    Head 类装饰器，用于注册。

    Args:
        name (str): 注册名称。
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in HEAD_REGISTRY:
            raise ValueError(f"Head '{name}' 已被注册: {HEAD_REGISTRY[name]}")
        HEAD_REGISTRY[name] = cls
        return cls
    return decorator


def create_head(name: str, **kwargs) -> nn.Module:
    """
    创建 Head 实例。

    Args:
        name (str): 注册名称。
        **kwargs: 构造函数参数。

    Returns:
        nn.Module: Head 实例。
    """
    if name not in HEAD_REGISTRY:
        available = list(HEAD_REGISTRY.keys())
        raise ValueError(f"Head '{name}' 未找到。可用模型: {available}")
    return HEAD_REGISTRY[name](**kwargs)
