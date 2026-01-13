# -*- coding: utf-8 -*-
"""
==============================================================================
深度可调的DANN模型 (增强版)
==============================================================================
特性：
- 动态 CNN/LSTM/Classifier/Discriminator 层数
- 双向 LSTM 支持
- 时序注意力机制 (Temporal Attention)
- 残差连接 (Residual Blocks)
- BatchNorm 支持

维度变化 (假设输入 (B, 100, 21), cnn_channels=[48, 96], lstm_hidden=128, bidirectional=True):
---------------------------------------------------------------------------
输入: (B, 100, 21)
  ↓ permute
(B, 21, 100)
  ↓ ResidualCNNBlock × 2
(B, 96, 25)  [经过2次MaxPool，时间维度100→50→25]
  ↓ permute
(B, 25, 96)
  ↓ BiLSTM (hidden=128, bidirectional=True)
(B, 25, 256)  [双向输出维度 = 128 × 2 = 256]
  ↓ TemporalAttention
(B, 256)  [加权聚合所有时间步]
  ↓ Classifier
(B, 7)  [7个故障类别]
---------------------------------------------------------------------------

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    """梯度反转层 - DANN 的核心组件"""
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class TemporalAttention(nn.Module):
    """
    时序注意力机制
    
    对 LSTM 的所有时间步输出进行加权聚合，让模型关注最重要的时刻。
    
    维度变化：
        输入: (B, T, H) - T 个时间步，每步 H 维特征
        输出: (B, H) - 加权聚合后的特征向量
    
    注意力计算:
        score_t = v^T * tanh(W * h_t + b)
        α = softmax(scores)
        output = Σ α_t * h_t
    """
    def __init__(self, hidden_size: int):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: LSTM 输出, shape = (B, T, H)
        Returns:
            context: 注意力加权后的特征, shape = (B, H)
        """
        # (B, T, H) -> (B, T, 1)
        attention_scores = self.attention(lstm_output)
        # (B, T, 1) -> (B, T, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        # (B, T, H) * (B, T, 1) -> (B, T, H) -> sum -> (B, H)
        context = torch.sum(lstm_output * attention_weights, dim=1)
        
        return context


class ResidualBlock1D(nn.Module):
    """
    一维残差块 (用于 CNN)
    
    结构:
        x -> Conv1d -> BN -> ReLU -> Conv1d -> BN -> (+x) -> ReLU
    
    如果输入输出通道数不同，使用 1x1 卷积进行投影。
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, use_batchnorm: bool = True, use_layernorm: bool = False):
        super(ResidualBlock1D, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        if use_layernorm:
            self.bn1 = nn.GroupNorm(1, out_channels)
        else:
            self.bn1 = nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
            
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        if use_layernorm:
            self.bn2 = nn.GroupNorm(1, out_channels)
        else:
            self.bn2 = nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
        
        # 如果通道数改变，需要投影
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            norm_layer = nn.Identity()
            if use_layernorm:
                norm_layer = nn.GroupNorm(1, out_channels)
            elif use_batchnorm:
                norm_layer = nn.BatchNorm1d(out_channels)
                
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                norm_layer
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        维度变化: (B, C_in, T) -> (B, C_out, T)
        """
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual  # 残差连接
        out = self.relu(out)
        
        return out


class DANNDeep(nn.Module):
    """
    深度可调的DANN模型 (增强版)
    
    新增特性:
    - use_attention: 是否使用时序注意力
    - use_residual: 是否使用残差连接
    """
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        num_classes: int,
        # CNN配置
        cnn_layers: int = 2,
        cnn_channels: list = [64, 128],
        cnn_kernel_size: int = 5,
        # LSTM配置
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        lstm_bidirectional: bool = False,
        # 分类器配置
        classifier_layers: int = 2,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.5,
        # 域判别器配置
        discriminator_hidden: int = 64,
        discriminator_layers: int = 2,
        # 增强选项
        use_batchnorm: bool = False,
        use_layernorm: bool = True,     # 新增：LayerNorm (GroupNorm for CNN)
        use_attention: bool = True,      # 新增：时序注意力
        use_residual: bool = True         # 新增：残差连接
    ):
        super(DANNDeep, self).__init__()
        
        self.lstm_bidirectional = lstm_bidirectional
        self.use_attention = use_attention
        self.use_residual = use_residual
        lstm_output_size = lstm_hidden * (2 if lstm_bidirectional else 1)
        
        # ========== 动态CNN层 (支持残差连接) ==========
        cnn_modules = []
        in_channels = n_features
        current_seq_len = seq_len
        
        for i, out_channels in enumerate(cnn_channels[:cnn_layers]):
            if use_residual:
                # 使用残差块
                cnn_modules.append(ResidualBlock1D(in_channels, out_channels, 
                                                   cnn_kernel_size, use_batchnorm, use_layernorm))
            else:
                # 普通卷积
                cnn_modules.append(nn.Conv1d(in_channels, out_channels, 
                                             kernel_size=cnn_kernel_size, 
                                             padding=cnn_kernel_size // 2))
                if use_layernorm:
                    cnn_modules.append(nn.GroupNorm(1, out_channels))
                elif use_batchnorm:
                    cnn_modules.append(nn.BatchNorm1d(out_channels))
                cnn_modules.append(nn.ReLU(inplace=True))
            
            cnn_modules.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels
            current_seq_len = current_seq_len // 2
        
        self.cnn = nn.Sequential(*cnn_modules)
        self.cnn_output_channels = in_channels
        self.cnn_output_len = current_seq_len
        
        # ========== LSTM ==========
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=lstm_bidirectional
        )
        
        # ========== 时序注意力 (可选) ==========
        if use_attention:
            self.attention = TemporalAttention(lstm_output_size)
        
        self.feature_dim = lstm_output_size
        
        # ========== 动态分类器 ==========
        classifier_modules = []
        in_dim = lstm_output_size
        for i in range(classifier_layers - 1):
            classifier_modules.append(nn.Linear(in_dim, classifier_hidden))
            if use_layernorm:
                classifier_modules.append(nn.LayerNorm(classifier_hidden))
            elif use_batchnorm:
                classifier_modules.append(nn.BatchNorm1d(classifier_hidden))
            classifier_modules.append(nn.ReLU(inplace=True))
            classifier_modules.append(nn.Dropout(p=classifier_dropout))
            in_dim = classifier_hidden
        classifier_modules.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_modules)
        
        # ========== 动态域判别器 ==========
        discriminator_modules = []
        in_dim = lstm_output_size
        for i in range(discriminator_layers - 1):
            discriminator_modules.append(nn.Linear(in_dim, discriminator_hidden))
            if use_layernorm:
                discriminator_modules.append(nn.LayerNorm(discriminator_hidden))
            elif use_batchnorm:
                discriminator_modules.append(nn.BatchNorm1d(discriminator_hidden))
            discriminator_modules.append(nn.ReLU(inplace=True))
            discriminator_modules.append(nn.Dropout(p=classifier_dropout))
            in_dim = discriminator_hidden
        discriminator_modules.append(nn.Linear(in_dim, 1))
        self.discriminator = nn.Sequential(*discriminator_modules)
        
        self.grl_lambda = 0.0
        
        # ========== 权重初始化 (修复类别偏置问题) ==========
        self._init_weights()
    
    def set_grl_alpha(self, alpha):
        """兼容旧接口 set_grl_alpha"""
        self.grl_lambda = alpha

    def set_grl_lambda(self, lambda_val: float):
        self.grl_lambda = lambda_val
    
    def _init_weights(self):
        """
        正确的权重初始化 - 修复类别偏置问题
        
        问题：PyTorch默认初始化可能导致某些类别的输出偏大，
        造成模型一开始就偏向预测某个类别。
        
        解决方案：
        - 使用Xavier初始化线性层权重
        - 将分类器输出层的bias初始化为0
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化权重
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                # Kaiming初始化卷积层
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                # LayerNorm/GroupNorm初始化
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                # LSTM权重初始化
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
    
    def _extract_features(self, x):
        """
        特征提取流程
        
        维度变化 (假设 B=32, T=100, F=21):
            输入: (32, 100, 21)
            permute: (32, 21, 100)
            CNN: (32, 96, 25)
            permute: (32, 25, 96)
            LSTM: (32, 25, 256) [如果双向]
            Attention: (32, 256)
        """
        # (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # (B, C, T') -> (B, T', C)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_output, (h_n, _) = self.lstm(x)
        
        # 特征聚合
        if self.use_attention:
            # 使用注意力机制聚合所有时间步
            features = self.attention(lstm_output)
        else:
            # 仅使用最后时刻
            if self.lstm_bidirectional:
                features = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                features = h_n[-1]
        
        return features
    
    def forward(self, x_source, x_target=None):
        outputs = {}
        features_s = self._extract_features(x_source)
        outputs['features'] = features_s  # 用于 t-SNE 可视化
        outputs['features_source'] = features_s
        outputs['class_logits'] = self.classifier(features_s)
        
        reversed_s = GradientReversal.apply(features_s, self.grl_lambda)
        outputs['domain_logits_source'] = self.discriminator(reversed_s)
        
        if x_target is not None:
            features_t = self._extract_features(x_target)
            outputs['features_target'] = features_t
            reversed_t = GradientReversal.apply(features_t, self.grl_lambda)
            outputs['domain_logits_target'] = self.discriminator(reversed_t)
        
        return outputs
