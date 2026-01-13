# -*- coding: utf-8 -*-
"""
模型模块初始化文件
包含DANN模型、基线模型及自定义层的导出
"""

from .layers import GradientReversalLayer, GradientReversalFunction
from .dann import DANN, FeatureExtractor, FaultClassifier, DomainDiscriminator
from .dann_improved import DANN_Improved, ImprovedFeatureExtractor, FaultClassifierImproved

__all__ = [
    'GradientReversalLayer',
    'GradientReversalFunction',
    'DANN',
    'FeatureExtractor',
    'FaultClassifier',
    'DomainDiscriminator',
    # 改进版
    'DANN_Improved',
    'ImprovedFeatureExtractor',
    'FaultClassifierImproved'
]
