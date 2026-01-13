# -*- coding: utf-8 -*-
"""
==============================================================================
改进的损失函数模块
==============================================================================
功能：解决类别不平衡问题
- 类别权重计算
- Focal Loss
- Label Smoothing CrossEntropy

作者：UAV-DANN项目改进
日期：2025年
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def compute_class_weights(train_labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    """
    计算类别权重（逆频率加权）

    Args:
        train_labels: 训练集标签，shape (N,)
        num_classes: 类别数量

    Returns:
        class_weights: 形状为(num_classes,)的权重张量

    原理：
        weight[i] = total_samples / (num_classes * count[i])
        少数类获得更高权重
    """
    from sklearn.utils.class_weight import compute_class_weight

    # 获取实际存在的类别
    classes = np.unique(train_labels)

    # 计算逆频率权重
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels
    )

    # 转换为tensor
    class_weights = torch.tensor(weights, dtype=torch.float32)

    print("=" * 60)
    print("类别权重分布:")
    print("-" * 60)
    class_names = ['No_Fault', 'Motor', 'Accelerometer', 'Gyroscope',
                   'Magnetometer', 'Barometer', 'GPS']

    total_samples = len(train_labels)
    for i in range(num_classes):
        count = np.sum(train_labels == i)
        percentage = count / total_samples * 100
        if i < len(weights):
            print(f"  类别 {i} ({class_names[i]:15s}): {count:5d}样本 ({percentage:5.2f}%), 权重={weights[i]:.4f}")
        else:
            print(f"  类别 {i} ({class_names[i]:15s}): {count:5d}样本 ({percentage:5.2f}%), 权重=N/A (缺失)")
    print("=" * 60)

    return class_weights


class FocalLoss(nn.Module):
    """
    Focal Loss: 专注于难分类样本

    论文: Focal Loss for Dense Object Detection (ICCV 2017)
    Lin et al., 2018

    公式:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    其中:
        p_t: 正确类别的预测概率
        α_t: 类别权重（处理类别不平衡）
        γ: 聚焦参数（推荐2.0，降低简单样本权重）

    优势:
        1. 降低简单样本的损失贡献
        2. 强迫模型关注难分类样本
        3. 自动处理极端类别不平衡
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: 类别权重，shape (num_classes,)
            gamma: 聚焦参数，γ=0退化为CE，γ>0增加难样本权重
            reduction: 'mean', 'sum', 或 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits, shape (B, C)
            targets: labels, shape (B,)

        Returns:
            loss: 标量
        """
        # 计算交叉熵（不reduction）
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction='none'
        )

        # 获取正确类别的预测概率
        p_t = torch.exp(-ce_loss)

        # Focal Loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing交叉熵

    论文: Rethinking the Inception Architecture for Computer Vision
    Szegedy et al., 2016

    原理:
        将硬标签 [0, 0, 1, 0, 0] 软化为 [0.05, 0.05, 0.8, 0.05, 0.05]
        防止过拟合，提高泛化能力

    公式:
        ls_loss = (1 - ε) * CE(y) + ε * CE(uniform)
    """

    def __init__(self, num_classes: int, epsilon: float = 0.1, weight: Optional[torch.Tensor] = None):
        """
        Args:
            num_classes: 类别数量
            epsilon: 平滑系数（0.05-0.2常用）
            weight: 类别权重
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits, shape (B, C)
            targets: labels, shape (B,)

        Returns:
            loss: 标量
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            # 创建平滑标签
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
            smooth_targets = (1 - self.epsilon) * targets_one_hot + \
                           self.epsilon / self.num_classes

        # 如果有权重，应用权重
        if self.weight is not None:
            weight = self.weight[targets]
            loss = -(smooth_targets * log_probs).sum(dim=-1) * weight
            return loss.mean()
        else:
            loss = -(smooth_targets * log_probs).sum(dim=-1)
            return loss.mean()


class AdaptiveLossWeighting(nn.Module):
    """
    自适应损失权重平衡

    动态调整分类损失和域损失的权重
    论文: Uncertainties of Domain Adaptation (NeurIPS 2020)

    原理:
        根据任务的不确定性动态调整损失权重
    """

    def __init__(
        self,
        init_cls_weight: float = 1.0,
        init_domain_weight: float = 0.5,
        min_weight: float = 0.1,
        max_weight: float = 2.0
    ):
        super(AdaptiveLossWeighting, self).__init__()
        # 可学习的权重参数
        self.cls_weight = nn.Parameter(torch.tensor(init_cls_weight))
        self.domain_weight = nn.Parameter(torch.tensor(init_domain_weight))
        self.min_weight = min_weight
        self.max_weight = max_weight

    def forward(self, cls_loss: torch.Tensor, domain_loss: torch.Tensor) -> tuple:
        """
        Args:
            cls_loss: 分类损失
            domain_loss: 域损失

        Returns:
            weighted_cls_loss: 加权分类损失
            weighted_domain_loss: 加权域损失
            total_loss: 总损失
        """
        # 限制权重范围
        cls_w = torch.clamp(self.cls_weight, self.min_weight, self.max_weight)
        domain_w = torch.clamp(self.domain_weight, self.min_weight, self.max_weight)

        weighted_cls = cls_w * cls_loss
        weighted_domain = domain_w * domain_loss
        total = weighted_cls + weighted_domain

        return weighted_cls, weighted_domain, total

    def get_weights(self) -> tuple:
        """获取当前权重"""
        cls_w = torch.clamp(self.cls_weight, self.min_weight, self.max_weight)
        domain_w = torch.clamp(self.domain_weight, self.min_weight, self.max_weight)
        return cls_w.item(), domain_w.item()


if __name__ == "__main__":
    """测试损失函数"""
    print("=" * 60)
    print("损失函数模块测试")
    print("=" * 60)

    # 模拟数据（严重不平衡）
    np.random.seed(42)
    train_labels = np.concatenate([
        np.zeros(100),      # No_Fault
        np.ones(500),       # Motor (大量)
        np.full(50, 2),     # Accelerometer (少)
        np.full(50, 3),     # Gyroscope (少)
        np.full(50, 4),     # Magnetometer (少)
        np.full(50, 5),     # Barometer (少)
        np.full(50, 6),     # GPS (少)
    ])

    # 测试类别权重计算
    weights = compute_class_weights(train_labels, num_classes=7)
    print(f"\n类别权重张量: {weights}")

    # 测试Focal Loss
    batch_size = 32
    num_classes = 7
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    focal_loss = FocalLoss(alpha=weights, gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"\nFocal Loss: {loss.item():.4f}")

    # 测试Label Smoothing
    ls_loss = LabelSmoothingCrossEntropy(num_classes=7, epsilon=0.1, weight=weights)
    loss = ls_loss(logits, targets)
    print(f"Label Smoothing Loss: {loss.item():.4f}")

    print("\n测试完成！")
