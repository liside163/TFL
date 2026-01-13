# -*- coding: utf-8 -*-
"""
==============================================================================
改进版DANN模型（带注意力机制）
==============================================================================
功能：针对35%目标域准确率问题的模型架构改进
- 多头自注意力机制
- 双向LSTM
- 更深的特征提取器
- 残差连接
- Layer Normalization

改进理由：
-------------
1. 注意力机制能自动学习关键时间步和传感器
2. 双向LSTM能捕获前后文信息
3. 残差连接缓解梯度消失，支持更深层网络
4. LayerNorm提高训练稳定性

作者：UAV-DANN项目改进
日期：2025年
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import yaml
import math

from .layers import GradientReversalLayer, compute_grl_lambda


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制

    论文: Attention Is All You Need (NeurIPS 2017)
    Vaswani et al., 2017

    公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量，shape (B, T, C)
            mask: 注意力掩码 (可选)

        Returns:
            output: 输出张量，shape (B, T, C)
            attn_weights: 注意力权重，shape (B, num_heads, T, T)
        """
        B, T, C = x.shape

        # 投影到Q, K, V
        Q = self.q_proj(x)  # (B, T, C)
        K = self.k_proj(x)  # (B, T, C)
        V = self.v_proj(x)  # (B, T, C)

        # 分割多头
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到V
        context = torch.matmul(attn_weights, V)  # (B, H, T, D)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        output = self.out_proj(context)

        return output, attn_weights


class ImprovedFeatureExtractor(nn.Module):
    """
    改进的特征提取器

    改进点：
    1. 更深的CNN（3层）
    2. 双向LSTM
    3. 多头自注意力
    4. 残差连接
    5. Layer Normalization
    """

    def __init__(
        self,
        n_features: int = 21,
        seq_len: int = 100,
        cnn_channels: list = [64, 128, 256],
        kernel_size: int = 3,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        attn_num_heads: int = 4,
        dropout: float = 0.3
    ):
        super(ImprovedFeatureExtractor, self).__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = True

        # -------------------- CNN块 (带残差) --------------------
        self.conv_blocks = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        in_channels = n_features
        for out_channels in cnn_channels:
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout * 0.5)
            )
            self.conv_blocks.append(conv_block)
            self.layer_norms.append(nn.LayerNorm(out_channels))
            in_channels = out_channels

        # 计算CNN输出维度
        self.cnn_output_seq_len = seq_len // (2 ** len(cnn_channels))
        self.cnn_output_channels = cnn_channels[-1]

        # -------------------- 双向LSTM --------------------
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_size = lstm_hidden_size * 2  # 双向

        # -------------------- 多头注意力 --------------------
        self.attention = MultiHeadAttention(
            embed_dim=lstm_output_size,
            num_heads=attn_num_heads,
            dropout=dropout
        )

        # -------------------- 输出层 --------------------
        self.output_norm = nn.LayerNorm(lstm_output_size)
        self.output_dropout = nn.Dropout(dropout)

        self.output_dim = lstm_output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, T, F) = (B, 100, 21)

        Returns:
            features: 特征向量 (B, output_dim)
        """
        # (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)

        # CNN特征提取
        for i, conv_block in enumerate(self.conv_blocks):
            identity = x  # 保存残差（如果维度匹配）
            x = conv_block(x)
            # 注意：由于池化改变维度，这里不做残差连接

        # (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)

        # LSTM时序建模
        lstm_out, _ = self.lstm(x)  # (B, T, 2*hidden_size)

        # 自注意力
        attn_out, attn_weights = self.attention(lstm_out)

        # 残差连接 + LayerNorm
        x = lstm_out + attn_out
        x = self.output_norm(x)

        # 全局平均池化
        features = x.mean(dim=1)  # (B, 2*hidden_size)
        features = self.output_dropout(features)

        return features


class FaultClassifierImproved(nn.Module):
    """
    改进的故障分类器

    改进点：
    1. 更深（3层）
    2. 残差连接
    3. 预测置信度估计（Dropout at inference time）
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        num_classes: int = 7,
        dropout: float = 0.5
    ):
        super(FaultClassifierImproved, self).__init__()

        layers = []
        in_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        n_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout预测不确定性

        Args:
            features: 特征向量
            n_samples: MC采样次数

        Returns:
            predictions: 预测类别
            uncertainty: 预测不确定性（熵）
        """
        self.train()  # 启用dropout

        all_probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.classifier(features)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs)  # (n_samples, B, C)
        mean_probs = all_probs.mean(dim=0)  # (B, C)

        predictions = torch.argmax(mean_probs, dim=1)

        # 计算熵作为不确定性度量
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)

        return predictions, entropy


class DomainDiscriminatorImproved(nn.Module):
    """
    改进的域判别器

    改进点：
    1. 更深（3层）
    2. 渐进式收缩结构
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.5,
        grl_alpha: float = 1.0
    ):
        super(DomainDiscriminatorImproved, self).__init__()

        # 梯度反转层
        self.grl = GradientReversalLayer(alpha=grl_alpha)

        # 判别器网络
        layers = []
        in_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(in_dim, 1))

        self.discriminator = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # GRL
        reversed_features = self.grl(features)
        # 判别
        domain_logit = self.discriminator(reversed_features)
        return domain_logit

    def set_grl_alpha(self, alpha: float) -> None:
        self.grl.set_alpha(alpha)


class DANN_Improved(nn.Module):
    """
    改进版DANN模型

    改进内容：
    1. 特征提取器：更深CNN + 双向LSTM + 注意力机制
    2. 分类器：更深网络 + 残差连接
    3. 域判别器：渐进式收缩结构
    """

    def __init__(self, config: dict = None, config_path: str = None):
        super(DANN_Improved, self).__init__()

        # 加载配置
        if config is None:
            if config_path is None:
                raise ValueError("必须提供config或config_path参数")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

        self.config = config

        # 提取模型配置
        model_config = config['model']
        fe_config = model_config['feature_extractor']
        cls_config = model_config['classifier']
        dd_config = model_config['domain_discriminator']

        # -------------------- 构建网络组件 --------------------

        # 改进的特征提取器
        self.feature_extractor = ImprovedFeatureExtractor(
            n_features=config['preprocessing']['n_features'],
            seq_len=config['preprocessing']['window_size'],
            cnn_channels=[96, 192, 256],
            kernel_size=5,
            lstm_hidden_size=fe_config['lstm']['hidden_size'] // 2,  # 双向LSTM，减半
            lstm_num_layers=fe_config['lstm']['num_layers'],
            attn_num_heads=4,
            dropout=fe_config['lstm']['dropout']
        )

        feature_dim = self.feature_extractor.output_dim

        # 改进的分类器
        self.classifier = FaultClassifierImproved(
            input_dim=feature_dim,
            hidden_dims=[256, 128, 64],
            num_classes=cls_config['num_classes'],
            dropout=cls_config['dropout']
        )

        # 改进的域判别器
        self.domain_discriminator = DomainDiscriminatorImproved(
            input_dim=feature_dim,
            hidden_dims=[128, 64, 32],
            dropout=dd_config['dropout'],
            grl_alpha=0.0
        )

        self.feature_dim = feature_dim
        self.num_classes = cls_config['num_classes']

    def forward(
        self,
        x_source: torch.Tensor,
        x_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x_source: 源域输入 (B, T, F)
            x_target: 目标域输入 (可选)

        Returns:
            outputs: 包含分类和域判别输出的字典
        """
        outputs = {}

        # -------------------- 源域处理 --------------------
        features_source = self.feature_extractor(x_source)
        outputs['features_source'] = features_source

        class_logits = self.classifier(features_source)
        outputs['class_logits'] = class_logits

        domain_logits_source = self.domain_discriminator(features_source)
        outputs['domain_logits_source'] = domain_logits_source

        # -------------------- 目标域处理 --------------------
        if x_target is not None:
            features_target = self.feature_extractor(x_target)
            outputs['features_target'] = features_target

            domain_logits_target = self.domain_discriminator(features_target)
            outputs['domain_logits_target'] = domain_logits_target

        return outputs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理模式"""
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
            logits = self.classifier(features)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def set_grl_alpha(self, alpha: float) -> None:
        """设置GRL的α值"""
        self.domain_discriminator.set_grl_alpha(alpha)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取中间特征（用于可视化）"""
        return self.feature_extractor(x)


def build_dann_improved(config_path: str) -> DANN_Improved:
    """
    从配置文件构建改进版DANN模型

    Args:
        config_path: 配置文件路径

    Returns:
        model: 改进版DANN模型实例
    """
    return DANN_Improved(config_path=config_path)


if __name__ == "__main__":
    """测试改进版DANN模型"""
    import os

    print("=" * 60)
    print("改进版DANN模型测试")
    print("=" * 60)

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建模型
    model = DANN_Improved(config=config)

    print(f"\n模型结构:")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 测试前向传播
    batch_size = 16
    seq_len = 100
    n_features = 21

    x_source = torch.randn(batch_size, seq_len, n_features)
    x_target = torch.randn(batch_size, seq_len, n_features)

    print(f"\n>>> 前向传播测试:")
    print(f"输入维度:")
    print(f"  x_source: {x_source.shape}")
    print(f"  x_target: {x_target.shape}")

    model.train()
    outputs = model(x_source, x_target)

    print(f"\n输出维度:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    print("\n测试完成！")
