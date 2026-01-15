"""
LSTM模型
用于时序故障检测的长程依赖建模
"""

import torch
import torch.nn as nn
from config import Config


class LSTMModel(nn.Module):
    """
    LSTM模型用于时间序列分类

    架构:
        Input: [batch, time_steps=100, features=27]
        Bidirectional LSTM (hidden=128, layers=2)
        Last timestep output
        FC(256 -> 128 -> 11)
    """

    def __init__(self, input_dim=27, num_classes=11, config=None):
        super(LSTMModel, self).__init__()

        self.config = config or Config()

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.config.LSTM_HIDDEN,
            num_layers=self.config.LSTM_LAYERS,
            batch_first=True,
            dropout=self.config.LSTM_DROPOUT if self.config.LSTM_LAYERS > 1 else 0,
            bidirectional=self.config.LSTM_BIDIRECTIONAL
        )

        # 计算LSTM输出维度
        lstm_output_dim = self.config.LSTM_HIDDEN * 2 if self.config.LSTM_BIDIRECTIONAL else self.config.LSTM_HIDDEN

        # 分类头
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        """
        前向传播

        参数:
            x: [batch, time_steps, features]
            return_features: bool, 是否返回中间特征

        返回:
            logits: [batch, num_classes]
            features (可选): [batch, lstm_output_dim]
        """
        # LSTM: [batch, time, input] -> [batch, time, hidden*2]
        # 输入: [batch, 100, 27]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, 100, 256] (双向128*2)

        # 使用最后一个时间步的输出
        # 对于双向LSTM，需要拼接前向和后向的最后隐藏状态
        if self.config.LSTM_BIDIRECTIONAL:
            # h_n shape: [num_layers*2, batch, hidden]
            # 拼接前向最后一层和后向最后一层
            h_forward = h_n[-2]  # 前向最后一层: [batch, 128]
            h_backward = h_n[-1]  # 后向最后一层: [batch, 128]
            features = torch.cat([h_forward, h_backward], dim=1)  # [batch, 256]
        else:
            features = h_n[-1]  # [batch, hidden]

        # 分类头
        # 输入: [batch, 256]
        x = self.fc1(features)  # 输出: [batch, 128]
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # 输出: [batch, 11]

        if return_features:
            return logits, features
        return logits


# 测试代码
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel().to(device)

    print("模型架构:")
    print(model)

    print("\n参数量:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")

    print("\n前向传播测试:")
    x = torch.randn(32, 100, 27).to(device)
    output, features = model(x, return_features=True)
    print(f"  输入shape: {x.shape}")
    print(f"  输出shape: {output.shape}")
    print(f"  特征shape: {features.shape}")
