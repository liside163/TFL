# -*- coding: utf-8 -*-
"""
==============================================================================
DANN (Domain-Adversarial Neural Network) 模型
==============================================================================
功能：实现完整的DANN模型架构
- 特征提取器 (1D-CNN + LSTM)
- 故障分类器
- 域判别器 (带GRL)

作者：UAV-DANN项目
日期：2025年

架构概述：
---------
                                    ┌─────────────────┐
                                    │   故障分类器     │
                                    │ (Fault Classifier)│
                                    │   输出: 11类     │
                                    └────────▲────────┘
                                             │
                                             │ 特征
            ┌─────────────────────────────────┴─────────────────────────────────┐
            │                           特征提取器                              │
输入数据 ──►│  1D-CNN (Conv1→Pool→Conv2→Pool) → LSTM → 特征向量 (128维)       │
(B,100,21)  │        Feature Extractor                                        │
            └─────────────────────────────────┬─────────────────────────────────┘
                                             │ 特征
                                             ▼
                                    ┌─────────────────┐
                                    │      GRL        │
                                    │  (梯度反转层)    │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   域判别器      │
                                    │ Domain Discriminator │
                                    │   输出: 0/1     │
                                    │  (HIL vs Real)  │
                                    └─────────────────┘
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import yaml

from .layers import GradientReversalLayer, compute_grl_lambda


class FeatureExtractor(nn.Module):
    """
    特征提取器 (1D-CNN + LSTM)
    
    使用1D卷积网络提取局部时序特征，然后通过LSTM捕获长期依赖关系
    
    网络结构：
    ---------
    输入: (B, T, F) = (batch_size, seq_len, n_features) = (32, 100, 21)
    
    1. 转置: (B, T, F) → (B, F, T) = (32, 21, 100)  # Conv1d需要(B, C_in, L)格式
    
    2. Conv1d层1:
       - 输入: (B, 21, 100)
       - 卷积: kernel=5, stride=1, padding=2
       - 输出: (B, 64, 100)  # 64个卷积核，保持时间维度
       - BatchNorm + ReLU
    
    3. MaxPool1d层1:
       - 输入: (B, 64, 100)
       - 池化: kernel=2, stride=2
       - 输出: (B, 64, 50)  # 时间维度减半
    
    4. Conv1d层2:
       - 输入: (B, 64, 50)
       - 卷积: kernel=5, stride=1, padding=2
       - 输出: (B, 128, 50)  # 通道数翻倍
       - BatchNorm + ReLU
    
    5. MaxPool1d层2:
       - 输入: (B, 128, 50)
       - 池化: kernel=2, stride=2
       - 输出: (B, 128, 25)  # 时间维度再减半
    
    6. 转置: (B, 128, 25) → (B, 25, 128)  # LSTM需要(B, T, F)格式
    
    7. LSTM:
       - 输入: (B, 25, 128)
       - hidden_size=128, num_layers=2
       - 输出: 取最后时刻的隐藏状态
       - 输出: (B, 128)  # 最终特征向量
    
    最终输出: (B, hidden_dim) = (32, 128)
    """
    
    def __init__(
        self,
        n_features: int = 21,
        seq_len: int = 100,
        conv1_out_channels: int = 64,
        conv1_kernel_size: int = 5,
        conv2_out_channels: int = 128,
        conv2_kernel_size: int = 5,
        pool_kernel_size: int = 2,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3
    ):
        """
        初始化特征提取器
        
        Args:
            n_features: 输入特征维度 (默认21)
            seq_len: 序列长度 (默认100)
            conv1_out_channels: 第一层卷积输出通道数 (默认64)
            conv1_kernel_size: 第一层卷积核大小 (默认5)
            conv2_out_channels: 第二层卷积输出通道数 (默认128)
            conv2_kernel_size: 第二层卷积核大小 (默认5)
            pool_kernel_size: 池化核大小 (默认2)
            lstm_hidden_size: LSTM隐藏层大小 (默认128)
            lstm_num_layers: LSTM层数 (默认2)
            lstm_dropout: LSTM Dropout概率 (默认0.3)
        """
        super(FeatureExtractor, self).__init__()
        
        # 保存参数
        self.n_features = n_features
        self.seq_len = seq_len
        self.lstm_hidden_size = lstm_hidden_size
        
        # -------------------- 1D-CNN 部分 --------------------
        # 第一层卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,      # 输入通道 = 特征维度
                out_channels=conv1_out_channels,
                kernel_size=conv1_kernel_size,
                stride=1,
                padding=conv1_kernel_size // 2  # 保持时间维度
            ),
            nn.BatchNorm1d(conv1_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
        )
        
        # 第二层卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=conv1_out_channels,
                out_channels=conv2_out_channels,
                kernel_size=conv2_kernel_size,
                stride=1,
                padding=conv2_kernel_size // 2
            ),
            nn.BatchNorm1d(conv2_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
        )
        
        # 计算CNN输出的时间维度
        # 原始: seq_len=100 → Pool1: 50 → Pool2: 25
        self.cnn_out_seq_len = seq_len // (pool_kernel_size ** 2)
        
        # -------------------- LSTM 部分 --------------------
        self.lstm = nn.LSTM(
            input_size=conv2_out_channels,   # CNN输出通道作为LSTM输入特征
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 输出特征维度
        self.output_dim = lstm_hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，shape = (B, T, F) = (batch_size, seq_len, n_features)
               例如: (32, 100, 21)
        
        Returns:
            features: 特征向量，shape = (B, hidden_dim) = (32, 128)
        
        维度变化详解：
        -------------
        1. 输入: (32, 100, 21)
        2. 转置 (B, T, F) → (B, F, T): (32, 21, 100)
        3. Conv1 + Pool1: (32, 64, 50)
        4. Conv2 + Pool2: (32, 128, 25)
        5. 转置 (B, C, T) → (B, T, C): (32, 25, 128)
        6. LSTM: 取最后时刻隐藏状态 → (32, 128)
        """
        batch_size = x.size(0)
        
        # -------------------- Step 1: 转置 --------------------
        # (B, T, F) → (B, F, T): 适配Conv1d的输入格式
        x = x.permute(0, 2, 1)  # (32, 21, 100)
        
        # -------------------- Step 2: 1D-CNN --------------------
        x = self.conv1(x)  # (32, 64, 50)
        x = self.conv2(x)  # (32, 128, 25)
        
        # -------------------- Step 3: 转置回LSTM格式 --------------------
        # (B, C, T) → (B, T, C): 适配LSTM的输入格式
        x = x.permute(0, 2, 1)  # (32, 25, 128)
        
        # -------------------- Step 4: LSTM --------------------
        # LSTM输出: output, (h_n, c_n)
        # output: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size) - 最后一个时刻的隐藏状态
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一层的最后时刻隐藏状态作为特征
        # h_n[-1]: (batch, hidden_size) = (32, 128)
        features = h_n[-1]
        
        return features


class FaultClassifier(nn.Module):
    """
    故障分类器
    
    将特征向量映射到11种故障类别
    
    网络结构：
    ---------
    输入: (B, hidden_dim) = (32, 128)
    
    1. FC1: 
       - 输入: (B, 128)
       - 线性变换: 128 → 64
       - BatchNorm + ReLU + Dropout
       - 输出: (B, 64)
    
    2. FC2 (输出层):
       - 输入: (B, 64)
       - 线性变换: 64 → 11
       - 输出: (B, 11) → logits (未归一化的类别得分)
    
    最终输出: (B, num_classes) = (32, 11)
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 11,
        dropout: float = 0.5
    ):
        """
        初始化故障分类器
        
        Args:
            input_dim: 输入特征维度 (默认128)
            hidden_dim: 隐藏层维度 (默认64)
            num_classes: 输出类别数 (默认11)
            dropout: Dropout概率 (默认0.5)
        """
        super(FaultClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            # FC1: 128 → 64
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            # FC2 (输出层): 64 → 11
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.output_dim = num_classes
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 特征向量，shape = (B, input_dim) = (32, 128)
        
        Returns:
            logits: 类别logits，shape = (B, num_classes) = (32, 11)
        
        维度变化：
            输入: (32, 128)
            FC1: (32, 64)
            FC2: (32, 11)
        """
        logits = self.classifier(features)
        return logits


class DomainDiscriminator(nn.Module):
    """
    域判别器 (Domain Discriminator)
    
    判别输入特征来自源域(HIL=0)还是目标域(Real=1)
    包含梯度反转层(GRL)，实现域对抗训练
    
    网络结构：
    ---------
    输入: (B, hidden_dim) = (32, 128)
    
    0. GRL: 梯度反转（仅在反向传播时生效）
       - 输入/输出: (B, 128)
       - 前向: 不变
       - 反向: 梯度 × (-α)
    
    1. FC1:
       - 输入: (B, 128)
       - 线性变换: 128 → 64
       - BatchNorm + ReLU + Dropout
       - 输出: (B, 64)
    
    2. FC2 (输出层):
       - 输入: (B, 64)
       - 线性变换: 64 → 1
       - 输出: (B, 1) → 域判别logit
    
    最终输出: (B, 1) = (32, 1)
    
    使用方式：
    ---------
    domain_pred = domain_discriminator(features)
    # domain_pred: (B, 1)
    # 0 = HIL (源域), 1 = Real (目标域)
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        grl_alpha: float = 1.0
    ):
        """
        初始化域判别器
        
        Args:
            input_dim: 输入特征维度 (默认128)
            hidden_dim: 隐藏层维度 (默认64)
            dropout: Dropout概率 (默认0.5)
            grl_alpha: GRL初始α值 (默认1.0)
        """
        super(DomainDiscriminator, self).__init__()
        
        # 梯度反转层
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        
        # 判别器网络
        self.discriminator = nn.Sequential(
            # FC1: 128 → 64
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            # FC2 (输出层): 64 → 1
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 特征向量，shape = (B, input_dim) = (32, 128)
        
        Returns:
            domain_logit: 域判别logit，shape = (B, 1) = (32, 1)
        
        维度变化：
            输入: (32, 128)
            GRL: (32, 128) [维度不变，梯度反转]
            FC1: (32, 64)
            FC2: (32, 1)
        """
        # 通过GRL（反向传播时梯度反转）
        reversed_features = self.grl(features)
        
        # 通过判别器
        domain_logit = self.discriminator(reversed_features)
        
        return domain_logit
    
    def set_grl_alpha(self, alpha: float) -> None:
        """
        设置GRL的α值
        
        Args:
            alpha: 新的梯度反转系数
        """
        self.grl.set_alpha(alpha)


class DANN(nn.Module):
    """
    DANN (Domain-Adversarial Neural Network) 完整模型
    
    整合特征提取器、故障分类器和域判别器
    
    完整数据流：
    -----------
    输入: x_source, x_target
        shape = (B, T, F) = (32, 100, 21)
    
    1. 特征提取:
        features_s = feature_extractor(x_source)  # (32, 128)
        features_t = feature_extractor(x_target)  # (32, 128)
    
    2. 故障分类 (仅源域):
        class_logits = classifier(features_s)  # (32, 11)
    
    3. 域判别 (源域 + 目标域):
        domain_logits_s = domain_discriminator(features_s)  # (32, 1)
        domain_logits_t = domain_discriminator(features_t)  # (32, 1)
    
    训练损失：
    ---------
    L_total = L_cls + λ * L_domain
    
    其中:
    - L_cls: 交叉熵损失 (仅源域，有标签)
    - L_domain: 二元交叉熵损失 (源域+目标域)
    - λ: 域适应权重，随训练进度从0增加到1
    """
    
    def __init__(self, config: dict = None, config_path: str = None):
        """
        初始化DANN模型
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        super(DANN, self).__init__()
        
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
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            n_features=config['preprocessing']['n_features'],
            seq_len=config['preprocessing']['window_size'],
            conv1_out_channels=fe_config['cnn']['conv1_out_channels'],
            conv1_kernel_size=fe_config['cnn']['conv1_kernel_size'],
            conv2_out_channels=fe_config['cnn']['conv2_out_channels'],
            conv2_kernel_size=fe_config['cnn']['conv2_kernel_size'],
            pool_kernel_size=fe_config['cnn']['pool_kernel_size'],
            lstm_hidden_size=fe_config['lstm']['hidden_size'],
            lstm_num_layers=fe_config['lstm']['num_layers'],
            lstm_dropout=fe_config['lstm']['dropout']
        )
        
        # 获取特征维度
        feature_dim = self.feature_extractor.output_dim
        
        # 故障分类器
        self.classifier = FaultClassifier(
            input_dim=feature_dim,
            hidden_dim=cls_config['hidden_dim'],
            num_classes=cls_config['num_classes'],
            dropout=cls_config['dropout']
        )
        
        # 域判别器
        self.domain_discriminator = DomainDiscriminator(
            input_dim=feature_dim,
            hidden_dim=dd_config['hidden_dim'],
            dropout=dd_config['dropout'],
            grl_alpha=0.0  # 初始α=0，训练时逐渐增加
        )
        
        # 保存关键维度信息
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
            x_source: 源域输入，shape = (B, T, F) = (32, 100, 21)
            x_target: 目标域输入 (可选)，shape = (B, T, F) = (32, 100, 21)
        
        Returns:
            outputs: 包含以下键的字典
                - 'class_logits': 故障分类logits, shape = (B, 11)
                - 'domain_logits_source': 源域域判别logits, shape = (B, 1)
                - 'domain_logits_target': 目标域域判别logits, shape = (B, 1) (如果提供x_target)
                - 'features_source': 源域特征, shape = (B, 128)
                - 'features_target': 目标域特征, shape = (B, 128) (如果提供x_target)
        
        维度变化总结：
        -------------
        x_source: (32, 100, 21) → features_source: (32, 128) → class_logits: (32, 11)
                                                             → domain_logits_source: (32, 1)
        
        x_target: (32, 100, 21) → features_target: (32, 128) → domain_logits_target: (32, 1)
        """
        outputs = {}
        
        # -------------------- 源域处理 --------------------
        # 特征提取
        features_source = self.feature_extractor(x_source)  # (B, 128)
        outputs['features_source'] = features_source
        
        # 故障分类
        class_logits = self.classifier(features_source)  # (B, 11)
        outputs['class_logits'] = class_logits
        
        # 域判别
        domain_logits_source = self.domain_discriminator(features_source)  # (B, 1)
        outputs['domain_logits_source'] = domain_logits_source
        
        # -------------------- 目标域处理 (如果提供) --------------------
        if x_target is not None:
            # 特征提取
            features_target = self.feature_extractor(x_target)  # (B, 128)
            outputs['features_target'] = features_target
            
            # 域判别
            domain_logits_target = self.domain_discriminator(features_target)  # (B, 1)
            outputs['domain_logits_target'] = domain_logits_target
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理模式：仅返回故障分类预测
        
        Args:
            x: 输入数据，shape = (B, T, F)
        
        Returns:
            predictions: 预测的故障类别，shape = (B,)
        """
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
            logits = self.classifier(features)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def set_grl_alpha(self, alpha: float) -> None:
        """
        设置GRL的α值
        
        Args:
            alpha: 梯度反转系数
        """
        self.domain_discriminator.set_grl_alpha(alpha)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取中间特征（用于可视化）
        
        Args:
            x: 输入数据，shape = (B, T, F)
        
        Returns:
            features: 特征向量，shape = (B, feature_dim)
        """
        return self.feature_extractor(x)


def build_dann_from_config(config_path: str) -> DANN:
    """
    从配置文件构建DANN模型
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        model: DANN模型实例
    """
    return DANN(config_path=config_path)


if __name__ == "__main__":
    """
    测试DANN模型
    """
    import os
    
    print("=" * 60)
    print("DANN 模型测试")
    print("=" * 60)
    
    # 获取配置文件路径
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    model = DANN(config=config)
    
    print(f"\n模型结构:")
    print(model)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    print(f"\n>>> 前向传播测试:")
    batch_size = 32
    seq_len = config['preprocessing']['window_size']
    n_features = config['preprocessing']['n_features']
    
    x_source = torch.randn(batch_size, seq_len, n_features)
    x_target = torch.randn(batch_size, seq_len, n_features)
    
    print(f"输入维度:")
    print(f"  x_source: {x_source.shape}")
    print(f"  x_target: {x_target.shape}")
    
    # 前向传播
    model.train()
    outputs = model(x_source, x_target)
    
    print(f"\n输出维度:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # 测试GRL alpha调度
    print(f"\n>>> GRL α调度测试:")
    for epoch in [0, 25, 50, 75, 100]:
        alpha = compute_grl_lambda(epoch, 100)
        model.set_grl_alpha(alpha)
        print(f"  epoch={epoch}, α={alpha:.4f}")
    
    print("\n测试完成！")
