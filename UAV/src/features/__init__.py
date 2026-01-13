"""
特征提取模块初始化。

导出注册表和相关函数。
"""

from src.features.registry import (
    FEATURE_EXTRACTOR_REGISTRY,
    create_feature_extractor,
    register_feature_extractor,
)

__all__ = [
    "FEATURE_EXTRACTOR_REGISTRY",
    "register_feature_extractor",
    "create_feature_extractor",
]
