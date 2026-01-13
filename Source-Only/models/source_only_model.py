# =====================================================================
# Source-Only 完整模型
# 整合特征提取器和分类器，用于无领域自适应的基线实验
# =====================================================================
#
# 模型架构与维度变换全流程:
#
# ┌─────────────────────────────────────────────────────────────────┐
# │                    Source-Only Model                             │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                  │
# │  输入: [B, 1000, 30]                                             │
# │    │                                                             │
# │    │  转置 (为CNN准备)                                           │
# │    ▼                                                             │
# │  [B, 30, 1000]                                                   │
# │    │                                                             │
# │    │  Conv1D Block 1: Conv(30→64) + BN + ReLU + MaxPool(2)      │
# │    ▼                                                             │
# │  [B, 64, 500]                                                    │
# │    │                                                             │
# │    │  Conv1D Block 2: Conv(64→128) + BN + ReLU + MaxPool(2)     │
# │    ▼                                                             │
# │  [B, 128, 250]                                                   │
# │    │                                                             │
# │    │  Conv1D Block 3: Conv(128→256) + BN + ReLU + MaxPool(2)    │
# │    ▼                                                             │
# │  [B, 256, 125]                                                   │
# │    │                                                             │
# │    │  转置 (为LSTM准备)                                          │
# │    ▼                                                             │
# │  [B, 125, 256]                                                   │
# │    │                                                             │
# │    │  Bi-LSTM(256→128*2)                                        │
# │    ▼                                                             │
# │  [B, 125, 256]  (LSTM输出)                                      │
# │    │                                                             │
# │    │  取最后时刻隐藏状态                                         │
# │    ▼                                                             │
# │  [B, 256]  ← 特征向量                                           │
# │    │                                                             │
# │    │  FC(256→128) + ReLU + Dropout                              │
# │    ▼                                                             │
# │  [B, 128]                                                        │
# │    │                                                             │
# │    │  FC(128→7)                                                  │
# │    ▼                                                             │
# │  [B, 7]  ← 7类故障概率输出                                       │
# │                                                                  │
# └─────────────────────────────────────────────────────────────────┘
# =====================================================================

import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.feature_extractor import FeatureExtractor
from models.classifier import FaultClassifier
from config import CNN_CONFIG, LSTM_CONFIG, CLASSIFIER_CONFIG


class SourceOnlyModel(nn.Module):
    """
    Source-Only 模型
    
    用于迁移学习的基线实验，仅使用源域数据训练，
    直接在目标域上评估，不进行任何领域自适应。
    
    Attributes:
        feature_extractor: CNN+LSTM特征提取器
        classifier: 故障分类器
    """
    
    def __init__(
        self,
        cnn_config: dict = None,
        lstm_config: dict = None,
        classifier_config: dict = None
    ):
        super().__init__()
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(cnn_config, lstm_config)
        
        # 更新分类器输入维度以匹配特征提取器输出
        if classifier_config is None:
            classifier_config = CLASSIFIER_CONFIG.copy()
        classifier_config["input_size"] = self.feature_extractor.output_size
        
        # 分类器
        self.classifier = FaultClassifier(classifier_config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，形状 [B, seq_len, features] = [B, 1000, 30]
        
        Returns:
            logits: 模型输出，形状 [B, num_classes] = [B, 7]
        """
        # 特征提取
        # [B, 1000, 30] -> [B, 256]
        features = self.feature_extractor(x)
        
        # 分类
        # [B, 256] -> [B, 7]
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征向量（用于可视化或其他分析）
        
        Args:
            x: 输入数据，形状 [B, 1000, 30]
        
        Returns:
            features: 特征向量，形状 [B, 256]
        """
        with torch.no_grad():
            return self.feature_extractor(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            x: 输入数据，形状 [B, 1000, 30]
        
        Returns:
            predictions: 预测类别索引，形状 [B]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别概率
        
        Args:
            x: 输入数据，形状 [B, 1000, 30]
        
        Returns:
            probabilities: 各类别概率，形状 [B, 7]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


def build_model(device: torch.device = None) -> SourceOnlyModel:
    """
    构建模型并移动到指定设备
    
    Args:
        device: 目标设备，默认使用配置中的DEVICE
    
    Returns:
        model: 构建好的模型
    """
    from config import DEVICE
    
    if device is None:
        device = DEVICE
    
    model = SourceOnlyModel()
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*50}")
    print(f"模型构建完成")
    print(f"{'='*50}")
    print(f"设备: {device}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"{'='*50}\n")
    
    return model


if __name__ == "__main__":
    # 测试代码
    print("测试 Source-Only 模型...")
    
    # 创建模拟输入
    batch_size = 4
    seq_len = 1000
    features = 30
    
    x = torch.randn(batch_size, seq_len, features)
    print(f"输入形状: {x.shape}")
    
    # 构建模型
    model = SourceOnlyModel()
    
    # 前向传播测试
    logits = model(x)
    print(f"Logits形状: {logits.shape}")
    
    # 特征提取测试
    feats = model.get_features(x)
    print(f"特征形状: {feats.shape}")
    
    # 预测测试
    preds = model.predict(x)
    print(f"预测形状: {preds.shape}")
    
    # 概率预测测试
    probs = model.predict_proba(x)
    print(f"概率形状: {probs.shape}")
    print(f"概率和: {probs.sum(dim=1)}")  # 应该都是1
    
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
