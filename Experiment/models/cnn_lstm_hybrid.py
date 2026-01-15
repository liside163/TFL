"""
CNN-LSTM混合模型
并行提取局部时序模式和全局依赖
"""

import torch
import torch.nn as nn
from config import Config


class CNNLSTMHybrid(nn.Module):
    """
    CNN-LSTM混合模型

    架构:
        CNN分支: 2层Conv1D -> GlobalAvgPool -> [batch, 128]
        LSTM分支: 双向LSTM -> 最后时间步 -> [batch, 256]
        拼接: [batch, 384]
        FC: 384 -> 256 -> 11
    """

    def __init__(self, input_dim=27, num_classes=11, config=None):
        super(CNNLSTMHybrid, self).__init__()

        self.config = config or Config()

        # CNN分支 - 提取局部时序模式
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [batch, 128, 1]

        # LSTM分支 - 建模长程依赖
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )  # 输出256维

        # 融合层
        fusion_dim = 128 + 256  # CNN(128) + LSTM(256) = 384
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        """
        前向传播

        参数:
            x: [batch, time_steps=100, features=27]
            return_features: bool

        返回:
            logits: [batch, num_classes]
            features (可选): [batch, fusion_dim]
        """
        # CNN分支 - 局部特征提取
        # 输入: [batch, 100, 27]
        x_cnn = x.transpose(1, 2)  # [batch, 27, 100]
        x_cnn = self.conv1(x_cnn)  # [batch, 64, 100]
        x_cnn = self.relu1(x_cnn)
        x_cnn = self.conv2(x_cnn)  # [batch, 128, 50]
        x_cnn = self.relu2(x_cnn)
        x_cnn = self.global_pool(x_cnn)  # [batch, 128, 1]
        cnn_features = x_cnn.squeeze(-1)  # [batch, 128]

        # LSTM分支 - 全局时序建模
        # 输入: [batch, 100, 27]
        x_lstm, _ = self.lstm(x)  # [batch, 100, 256]
        # 取最后时间步
        lstm_features = x_lstm[:, -1, :]  # [batch, 256]

        # 融合特征
        features = torch.cat([cnn_features, lstm_features], dim=1)  # [batch, 384]

        # 分类
        # 输入: [batch, 384]
        x = self.fc1(features)  # [batch, 256]
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # [batch, 11]

        if return_features:
            return logits, features
        return logits


# 测试代码
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMHybrid().to(device)

    print("模型架构:")
    print(model)

    print("\n参数量:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")

    x = torch.randn(32, 100, 27).to(device)
    output, features = model(x, return_features=True)
    print(f"\n前向传播:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {output.shape}")
    print(f"  特征: {features.shape}")
