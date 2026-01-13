"""
分类头实现。
"""

import torch.nn as nn

from src.models.heads import register_head

__all__ = ["ClassifierHead"]


@register_head("classifier")
class ClassifierHead(nn.Module):
    """
    通用分类头。
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 0):
        """
        Args:
            in_dim (int): 输入特征维度。
            num_classes (int): 类别数量。
            hidden_dim (int): 隐藏层维度。如果 > 0，则增加一层 MLP。
        """
        super().__init__()
        
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5), # 常用配置，也可参数化
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入特征 [B, in_dim]

        Returns:
            dict: {"logits": Tensor [B, num_classes]}
        """
        logits = self.net(x)
        return {"logits": logits}
