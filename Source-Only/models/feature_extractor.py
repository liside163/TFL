# =====================================================================
# 特征提取器模块
# 使用 1D CNN + Bi-LSTM 从时序数据中提取特征
# =====================================================================
#
# 维度变换流程:
# 输入: [B, seq_len, features] = [B, 1000, 30]
# 转置: [B, features, seq_len] = [B, 30, 1000]
# CNN处理: [B, 256, seq_len/8] = [B, 256, 125]
# 转置: [B, seq_len/8, 256] = [B, 125, 256]
# LSTM处理: [B, seq_len/8, hidden*2] = [B, 125, 256]
# 取最后时刻: [B, hidden*2] = [B, 256]
# =====================================================================

import torch
import torch.nn as nn
from typing import Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import CNN_CONFIG, LSTM_CONFIG


class CNN1DBlock(nn.Module):
    """
    1D卷积块: Conv1D + BatchNorm + ReLU + MaxPool
    
    维度变换:
    输入: [B, in_channels, seq_len]
    输出: [B, out_channels, seq_len / pool_size]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 卷积层，使用padding='same'保持序列长度
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # same padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, in_channels, seq_len]
        
        Returns:
            [B, out_channels, seq_len / pool_size]
        """
        x = self.conv(x)       # [B, out_channels, seq_len]
        x = self.bn(x)         # [B, out_channels, seq_len]
        x = self.relu(x)       # [B, out_channels, seq_len]
        x = self.pool(x)       # [B, out_channels, seq_len / pool_size]
        x = self.dropout(x)    # [B, out_channels, seq_len / pool_size]
        return x


class CNNFeatureExtractor(nn.Module):
    """
    多层1D CNN特征提取器
    
    维度变换:
    输入: [B, 30, 1000] (转置后的时序数据)
    经过3层CNN块后: [B, 256, 125]
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        if config is None:
            config = CNN_CONFIG
        
        # 三层CNN块
        # 第一层: [B, 30, 1000] -> [B, 64, 500]
        self.block1 = CNN1DBlock(
            in_channels=config["in_channels"],
            out_channels=config["layer1_channels"],
            kernel_size=config["kernel_size"],
            pool_size=config["pool_size"],
            dropout=config["dropout"]
        )
        
        # 第二层: [B, 64, 500] -> [B, 128, 250]
        self.block2 = CNN1DBlock(
            in_channels=config["layer1_channels"],
            out_channels=config["layer2_channels"],
            kernel_size=config["kernel_size"],
            pool_size=config["pool_size"],
            dropout=config["dropout"]
        )
        
        # 第三层: [B, 128, 250] -> [B, 256, 125]
        self.block3 = CNN1DBlock(
            in_channels=config["layer2_channels"],
            out_channels=config["layer3_channels"],
            kernel_size=config["kernel_size"],
            pool_size=config["pool_size"],
            dropout=config["dropout"]
        )
        
        self.output_channels = config["layer3_channels"]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, features, seq_len] = [B, 30, 1000]
        
        Returns:
            [B, 256, 125]
        """
        x = self.block1(x)  # [B, 64, 500]
        x = self.block2(x)  # [B, 128, 250]
        x = self.block3(x)  # [B, 256, 125]
        return x


class LSTMFeatureExtractor(nn.Module):
    """
    双向LSTM特征提取器
    
    维度变换:
    输入: [B, seq_len/8, 256] = [B, 125, 256]
    LSTM输出: [B, 125, hidden*2] = [B, 125, 256]
    取最后时刻: [B, hidden*2] = [B, 256]
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        if config is None:
            config = LSTM_CONFIG
        
        self.hidden_size = config["hidden_size"]
        self.bidirectional = config["bidirectional"]
        self.num_directions = 2 if self.bidirectional else 1
        
        # Bi-LSTM层
        self.lstm = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=True,  # 输入格式 [B, seq_len, features]
            bidirectional=config["bidirectional"],
            dropout=config["dropout"] if config["num_layers"] > 1 else 0
        )
        
        self.output_size = config["hidden_size"] * self.num_directions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, seq_len, input_size] = [B, 125, 256]
        
        Returns:
            [B, hidden*2] = [B, 256] (取最后时刻的隐藏状态)
        """
        # LSTM前向
        # output: [B, seq_len, hidden*num_directions] = [B, 125, 256]
        # h_n: [num_layers*num_directions, B, hidden] = [4, B, 128]
        output, (h_n, c_n) = self.lstm(x)
        
        # 取最后时刻的隐藏状态
        # 对于双向LSTM，拼接正向最后时刻和反向最后时刻的隐藏状态
        if self.bidirectional:
            # h_n[-2]: 正向最后一层最后时刻 [B, hidden]
            # h_n[-1]: 反向最后一层最后时刻 [B, hidden]
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, hidden*2]
        else:
            hidden = h_n[-1]  # [B, hidden]
        
        return hidden


class FeatureExtractor(nn.Module):
    """
    完整的特征提取器: CNN + LSTM
    
    维度变换总览:
    输入: [B, seq_len, features] = [B, 1000, 30]
    转置: [B, features, seq_len] = [B, 30, 1000]
    CNN: [B, 256, seq_len/8] = [B, 256, 125]
    转置: [B, seq_len/8, 256] = [B, 125, 256]
    LSTM: [B, hidden*2] = [B, 256]
    """
    
    def __init__(self, cnn_config: dict = None, lstm_config: dict = None):
        super().__init__()
        
        # CNN特征提取
        self.cnn = CNNFeatureExtractor(cnn_config)
        
        # 更新LSTM输入维度以匹配CNN输出
        if lstm_config is None:
            lstm_config = LSTM_CONFIG.copy()
        lstm_config["input_size"] = self.cnn.output_channels
        
        # LSTM时序特征提取
        self.lstm = LSTMFeatureExtractor(lstm_config)
        
        self.output_size = self.lstm.output_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, seq_len, features] = [B, 1000, 30]
        
        Returns:
            [B, 256] 特征向量
        """
        # 转置以适配Conv1D输入格式
        # [B, 1000, 30] -> [B, 30, 1000]
        x = x.transpose(1, 2)
        
        # CNN特征提取
        # [B, 30, 1000] -> [B, 256, 125]
        x = self.cnn(x)
        
        # 转置以适配LSTM输入格式
        # [B, 256, 125] -> [B, 125, 256]
        x = x.transpose(1, 2)
        
        # LSTM时序特征提取
        # [B, 125, 256] -> [B, 256]
        x = self.lstm(x)
        
        return x


if __name__ == "__main__":
    # 测试代码
    print("测试特征提取器...")
    
    # 创建模拟输入
    batch_size = 4
    seq_len = 1000
    features = 30
    
    x = torch.randn(batch_size, seq_len, features)
    print(f"输入形状: {x.shape}")
    
    # 测试特征提取器
    extractor = FeatureExtractor()
    output = extractor(x)
    print(f"输出形状: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in extractor.parameters())
    trainable_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
