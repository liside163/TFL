"""
Backbone 注册表模块。

提供 Backbone 模型的注册和创建功能。
"""

from typing import Any, Callable, Dict, Optional, Type

import torch.nn as nn

__all__ = ["BACKBONE_REGISTRY", "register_backbone", "create_backbone", "BaseBackbone"]

BACKBONE_REGISTRY: Dict[str, Type[nn.Module]] = {}


class BaseBackbone(nn.Module):
    """
    所有 Backbone 的基类（可选）。
    
    统一接口：
    输入: x -> 输出: 特征向量 [B, out_dim] 或特征图 [B, C, L]
    """
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x):
        raise NotImplementedError


def register_backbone(name: str) -> Callable:
    """
    Backbone 类装饰器，用于注册。

    Args:
        name (str): 注册名称。
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in BACKBONE_REGISTRY:
            raise ValueError(f"Backbone '{name}' 已被注册: {BACKBONE_REGISTRY[name]}")
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator


def create_backbone(name: str, **kwargs) -> nn.Module:
    """
    创建 Backbone 实例。

    Args:
        name (str): 注册名称。
        **kwargs: 构造函数参数。

    Returns:
        nn.Module: Backbone 实例。
    """
    if name not in BACKBONE_REGISTRY:
        available = list(BACKBONE_REGISTRY.keys())
        raise ValueError(f"Backbone '{name}' 未找到。可用模型: {available}")
    return BACKBONE_REGISTRY[name](**kwargs)
