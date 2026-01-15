"""
1D CNN模型
用于时序故障检测的特征提取
"""

import torch
import torch.nn as nn
from config import Config


class CNN1D(nn.Module):
    """
    1D卷积神经网络用于时间序列分类

    架构:
        Input: [batch, time_steps=100, features=27]
        Conv1D(64) -> ReLU -> Dropout
        Conv1D(128) -> ReLU -> Dropout
        Conv1D(256) -> ReLU -> Dropout
        GlobalMaxPool
        FC(256 -> 128 -> 11)
    """

    def __init__(self, input_dim=27, num_classes=11, config=None):
        super(CNN1D, self).__init__()

        self.config = config or Config()

        # 第一层卷积: 保持时间维度
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.config.CNN_FILTERS[0],
            kernel_size=self.config.CNN_KERNELS[0],
            stride=self.config.CNN_STRIDES[0],
            padding=2  # 保持时间长度
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.config.CNN_DROPOUT[0])

        # 第二层卷积: 下采样时间维度
        self.conv2 = nn.Conv1d(
            in_channels=self.config.CNN_FILTERS[0],
            out_channels=self.config.CNN_FILTERS[1],
            kernel_size=self.config.CNN_KERNELS[1],
            stride=self.config.CNN_STRIDES[1],
            padding=2
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.config.CNN_DROPOUT[1])

        # 第三层卷积: 进一步下采样
        self.conv3 = nn.Conv1d(
            in_channels=self.config.CNN_FILTERS[1],
            out_channels=self.config.CNN_FILTERS[2],
            kernel_size=self.config.CNN_KERNELS[2],
            stride=self.config.CNN_STRIDES[2],
            padding=1
        )
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(self.config.CNN_DROPOUT[2])

        # 全局池化
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # [batch, channels, 1]

        # 分类头
        self.fc1 = nn.Linear(self.config.CNN_FILTERS[2], 128)
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        """
        前向传播

        参数:
            x: [batch, time_steps, features]
            return_features: bool, 是否返回中间特征

        返回:
            logits: [batch, num_classes]
            features (可选): [batch, 256]
        """
        # 转换维度: [batch, time, features] -> [batch, features, time]
        x = x.transpose(1, 2)  # [batch, 27, 100]

        # 卷积层1
        # 输入: [batch, 27, 100]
        x = self.conv1(x)  # 输出: [batch, 64, 100]
        x = self.relu1(x)
        x = self.dropout1(x)

        # 卷积层2
        # 输入: [batch, 64, 100]
        x = self.conv2(x)  # 输出: [batch, 128, 50]
        x = self.relu2(x)
        x = self.dropout2(x)

        # 卷积层3
        # 输入: [batch, 128, 50]
        x = self.conv3(x)  # 输出: [batch, 256, 25]
        x = self.relu3(x)
        x = self.dropout3(x)

        # 全局池化
        # 输入: [batch, 256, 25]
        x = self.global_pool(x)  # 输出: [batch, 256, 1]
        x = x.squeeze(-1)  # 输出: [batch, 256]

        # 保存特征 (用于迁移学习)
        features = x

        # 分类头
        # 输入: [batch, 256]
        x = self.fc1(x)  # 输出: [batch, 128]
        x = self.relu_fc(x)
        x = self.dropout_fc(x)
        logits = self.fc2(x)  # 输出: [batch, 11]

        if return_features:
            return logits, features
        return logits

    def get_feature_extractor(self):
        """返回特征提取器部分 (用于迁移学习)"""
        return nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2,
            self.conv3, self.relu3, self.dropout3,
            self.global_pool,
            nn.Flatten()
        )


# 测试代码
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN1D().to(device)

    print("模型架构:")
    print(model)

    print("\n参数量:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    print("\n前向传播测试:")
    x = torch.randn(32, 100, 27).to(device)
    output, features = model(x, return_features=True)
    print(f"  输入shape: {x.shape}")
    print(f"  输出shape: {output.shape}")
    print(f"  特征shape: {features.shape}")
