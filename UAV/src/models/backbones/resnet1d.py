"""
1D ResNet Backbone 实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import BaseBackbone, register_backbone

__all__ = ["ResNet1D"]


class BasicBlock1D(nn.Module):
    """
    1D ResNet 基础残差块。
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


@register_backbone("resnet1d_18")
class ResNet1D(BaseBackbone):
    """
    ResNet-18 for 1D signal.
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 64, out_dim: int = 512):
        """
        Args:
            in_channels (int): 输入通道数。
            base_filters (int): 基础滤波器数量。
            out_dim (int): 输出特征维度（Global Average Pooling 后的维度）。
        """
        super().__init__(out_dim=out_dim)
        self.in_planes = base_filters

        self.conv1 = nn.Conv1d(
            in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock1D, base_filters, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, base_filters * 2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, base_filters * 4, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, base_filters * 8, 2, stride=2)
        
        # 最终输出层通常会经过一个 FC 调整维度，或者直接是 GAP 的输出
        # ResNet18 标准实现最后是 GAP -> 512
        # 如果 out_dim 不等于 512，可以加一个线性层映射
        self.final_dim = base_filters * 8 * BasicBlock1D.expansion
        
        self.fc = None
        if out_dim != self.final_dim:
            self.fc = nn.Linear(self.final_dim, out_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, L]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Average Pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)

        if self.fc:
            x = self.fc(x)

        return x
