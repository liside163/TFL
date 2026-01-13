from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: Tuple[int, int], stride: int = 1):
        super().__init__()
        pad = (k[0] // 2, k[1] // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = _ConvBNReLU(ch, ch, (3, 3))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return F.relu(x + y, inplace=True)


class MSCNNFeatureExtractor(nn.Module):
    def __init__(self, in_ch: int = 6, base_channels: int = 32, feat_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        # 为什么：多尺度卷积能同时捕获不同时间/频率分辨率的故障纹理特征
        self.branch3 = nn.Sequential(_ConvBNReLU(in_ch, base_channels, (3, 3)), nn.MaxPool2d(2))
        self.branch5 = nn.Sequential(_ConvBNReLU(in_ch, base_channels, (5, 5)), nn.MaxPool2d(2))
        self.branch7 = nn.Sequential(_ConvBNReLU(in_ch, base_channels, (7, 7)), nn.MaxPool2d(2))

        merged_ch = base_channels * 3
        self.backbone = nn.Sequential(
            _ConvBNReLU(merged_ch, merged_ch, (3, 3)),
            nn.MaxPool2d(2),
            _ResBlock(merged_ch),
            _ResBlock(merged_ch),
            _ConvBNReLU(merged_ch, merged_ch * 2, (3, 3)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(merged_ch * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        z = torch.cat([b3, b5, b7], dim=1)
        z = self.backbone(z)
        z = self.proj(z)
        return z


class FaultClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        # 为什么：分类器保持简单，主要让特征提取器与 MMD 去承擔“跨域泛化”
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DomainAdaptNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, feat_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.feature_extractor = MSCNNFeatureExtractor(
            in_ch=6,
            base_channels=base_channels,
            feat_dim=feat_dim,
            dropout=dropout,
        )
        self.classifier = FaultClassifier(feat_dim=feat_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        z = self.feature_extractor(x)
        logits = self.classifier(z)
        if return_feat:
            return logits, z
        return logits
