# -*- coding: utf-8 -*-
"""
==============================================================================
自定义神经网络层
==============================================================================
功能：实现DANN所需的自定义层
- 梯度反转层 (Gradient Reversal Layer, GRL)

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Tuple


class GradientReversalFunction(Function):
    """
    梯度反转函数 (Gradient Reversal Function)
    
    DANN的核心组件：在反向传播时将梯度取反
    
    原理说明：
    ---------
    在域对抗训练中：
    - 前向传播：特征直接通过，不做任何变换
    - 反向传播：梯度乘以 -λ，实现梯度反转
    
    这使得特征提取器在优化过程中：
    - 最小化分类损失（正常梯度下降）
    - 最大化域判别损失（梯度反转后变为梯度上升）
    
    最终效果：特征提取器学习"域不变"的特征表示
    
    数学表达：
    ---------
    前向: GRL(x) = x
    反向: ∂L/∂x = -λ * ∂L/∂GRL(x)
    """
    
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        前向传播：直接返回输入，同时保存反转系数
        
        Args:
            ctx: 上下文对象，用于保存反向传播所需的信息
            x: 输入张量，shape = (batch_size, feature_dim)
            alpha: 梯度反转系数 (λ)，控制反转强度
            
        Returns:
            output: 与输入相同的张量，shape = (batch_size, feature_dim)
        
        维度变化：
            输入: (B, H) → 输出: (B, H)
            无变化，直接透传
        """
        # 保存alpha供反向传播使用
        ctx.alpha = alpha
        
        # 前向传播不做任何变换
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        反向传播：将梯度乘以 -alpha 实现反转
        
        Args:
            ctx: 上下文对象
            grad_output: 上游传来的梯度，shape = (batch_size, feature_dim)
            
        Returns:
            grad_input: 反转后的梯度，shape = (batch_size, feature_dim)
            None: alpha参数不需要梯度
        
        维度变化：
            输入梯度: (B, H) → 输出梯度: (B, H)
            梯度值变为 -α * grad_output
        """
        # 获取保存的alpha
        alpha = ctx.alpha
        
        # 梯度反转：乘以 -alpha
        # 关键代码：这就是GRL的精髓！
        grad_input = -alpha * grad_output
        
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    梯度反转层 (Gradient Reversal Layer, GRL)
    
    将GradientReversalFunction封装为nn.Module，便于在网络中使用
    
    使用方式：
    ---------
    grl = GradientReversalLayer(alpha=1.0)
    features = feature_extractor(x)
    reversed_features = grl(features)  # 前向：不变；反向：梯度反转
    domain_output = domain_discriminator(reversed_features)
    
    alpha调度策略：
    --------------
    在训练过程中，α通常从0逐渐增加到1：
    α(p) = 2 / (1 + exp(-γ*p)) - 1
    其中 p = epoch/total_epochs, γ=10
    
    这样做的好处：
    - 训练初期(α≈0)：专注于分类任务，不进行域适应
    - 训练后期(α≈1)：全力进行域适应
    
    Attributes:
        alpha: 当前的梯度反转系数
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        初始化梯度反转层
        
        Args:
            alpha: 初始梯度反转系数，默认为1.0
        """
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征张量，shape = (batch_size, feature_dim)
            
        Returns:
            output: 与输入相同的张量，shape = (batch_size, feature_dim)
        
        维度变化：
            输入: (B, H) = (32, 128) → 输出: (B, H) = (32, 128)
            维度不变，但反向传播时梯度会被反转
        """
        return GradientReversalFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha: float) -> None:
        """
        更新梯度反转系数
        
        Args:
            alpha: 新的梯度反转系数
        """
        self.alpha = alpha


def compute_grl_lambda(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    """
    计算GRL的λ系数（按训练进度调度）
    
    使用sigmoid调度策略：λ逐渐从0增加到1
    
    公式：λ(p) = 2 / (1 + exp(-γ*p)) - 1
    其中 p = epoch / total_epochs
    
    Args:
        epoch: 当前训练轮数（从0开始）
        total_epochs: 总训练轮数
        gamma: 调度参数，控制增长速度，默认10.0
        
    Returns:
        lambda_p: 当前的λ值，范围[0, 1]
    
    示例：
        total_epochs = 100
        epoch=0   → λ≈0.00  (不进行域适应)
        epoch=50  → λ≈0.73  (中等强度)
        epoch=100 → λ≈1.00  (全力域适应)
    """
    import math
    
    # 计算训练进度 p ∈ [0, 1]
    p = epoch / total_epochs
    
    # sigmoid调度
    lambda_p = 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
    
    return lambda_p


if __name__ == "__main__":
    """
    测试GRL层
    """
    print("=" * 60)
    print("梯度反转层 (GRL) 测试")
    print("=" * 60)
    
    # 创建测试输入
    batch_size = 4
    feature_dim = 8
    x = torch.randn(batch_size, feature_dim, requires_grad=True)
    
    print(f"\n输入张量 x:")
    print(f"  shape: {x.shape}")
    print(f"  requires_grad: {x.requires_grad}")
    
    # 测试GRL
    grl = GradientReversalLayer(alpha=1.0)
    y = grl(x)
    
    print(f"\n输出张量 y (经过GRL):")
    print(f"  shape: {y.shape}")
    print(f"  与输入相同: {torch.allclose(x, y)}")
    
    # 测试梯度反转
    loss = y.sum()
    loss.backward()
    
    print(f"\n梯度测试 (alpha=1.0):")
    print(f"  理论梯度: {torch.ones_like(x)[0, :4]}")
    print(f"  实际梯度: {x.grad[0, :4]}")
    print(f"  梯度被反转: {torch.allclose(x.grad, -torch.ones_like(x))}")
    
    # 测试alpha调度
    print(f"\n>>> λ调度测试 (total_epochs=100):")
    for epoch in [0, 10, 25, 50, 75, 100]:
        lambda_p = compute_grl_lambda(epoch, 100)
        print(f"  epoch={epoch:3d} → λ={lambda_p:.4f}")
    
    print("\n测试完成！")
