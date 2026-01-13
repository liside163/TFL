"""
特征提取器注册表模块。

提供特征提取器的注册和工厂方法。
建议在 `src/features/__init__.py` 中导出此模块的内容，以便外部使用。
"""

from typing import Any, Callable, Dict, Optional

__all__ = ["FEATURE_EXTRACTOR_REGISTRY", "register_feature_extractor", "create_feature_extractor"]

# 注册表：存储名称到构造函数（或类）的映射
# 构造函数签名通常为：Constructor(**kwargs) -> Callable[[Input], Output]
FEATURE_EXTRACTOR_REGISTRY: Dict[str, Callable[..., Callable]] = {}


def register_feature_extractor(name: str) -> Callable:
    """
    装饰器，用于注册特征提取器构造函数或类。

    Args:
        name (str): 特征提取器的注册名称。

    Returns:
        Callable: 装饰器函数。
    """
    def decorator(cls_or_fn: Callable) -> Callable:
        if name in FEATURE_EXTRACTOR_REGISTRY:
            raise ValueError(f"特征提取器 '{name}' 已被注册: {FEATURE_EXTRACTOR_REGISTRY[name]}")
        FEATURE_EXTRACTOR_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return decorator


def create_feature_extractor(name: Optional[str], **kwargs) -> Optional[Callable]:
    """
    创建特征提取器实例。

    Args:
        name (Optional[str]): 特征提取器名称。如果为 None，则返回 None。
        **kwargs: 传递给特征提取器构造函数的参数。

    Returns:
        Optional[Callable]: 实例化的特征提取器（可调用对象），或 None。

    Raises:
        ValueError: 如果名称未注册。
    """
    if name is None:
        return None

    if name not in FEATURE_EXTRACTOR_REGISTRY:
        available = list(FEATURE_EXTRACTOR_REGISTRY.keys())
        raise ValueError(f"特征提取器 '{name}' 未找到。可用提取器: {available}")

    # 实例化
    return FEATURE_EXTRACTOR_REGISTRY[name](**kwargs)
