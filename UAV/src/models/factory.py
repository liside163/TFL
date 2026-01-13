"""
模型工厂模块。

定义通用迁移学习模型结构，并提供从配置构建模型的工厂函数。
"""

from typing import Any, Dict, Optional

import torch.nn as nn

from src.models.backbones import create_backbone
from src.models.heads import create_head

__all__ = ["TransferModel", "build_model"]


class TransferModel(nn.Module):
    """
    通用迁移学习模型。

    结构:
        Input -> Backbone -> Features -> Head -> Logits/Outputs

    支持冻结 Backbone 参数进行微调或 Linear Probing。
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone (nn.Module): 特征提取器。
            head (nn.Module): 任务头。
            freeze_backbone (bool): 是否冻结 backbone 参数。
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            self._freeze_module(self.backbone)

    def _freeze_module(self, module: nn.Module):
        """冻结模块的所有参数。"""
        module.eval() # 设置为 eval 模式 (影响 Dropout/BN)
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """
        重写 train 方法。
        如果冻结了 backbone，即使在 model.train() 时，backbone 也应保持 eval 模式
        (特别是 BatchNorm 的统计量不应更新)。
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x, **kwargs) -> Dict[str, Any]:
        """
        前向传播。

        Args:
            x (Tensor): 输入数据。

        Returns:
            Dict[str, Any]: 模型输出，至少包含 "logits"。
                也可包含 "features" 以便于 t-SNE 可视化或正则化。
        """
        features = self.backbone(x)
        outputs = self.head(features)
        
        # 自动注入 features 到输出字典中，便于后续使用
        if isinstance(outputs, dict):
            if "features" not in outputs:
                outputs["features"] = features
        else:
            # 如果 head 返回的不是字典（不太可能，根据规范），尝试包装
            outputs = {"logits": outputs, "features": features}
            
        return outputs


def build_model(model_cfg: Dict[str, Any]) -> TransferModel:
    """
    根据配置构建 TransferModel。

    配置结构示例:
    {
        "backbone": {
            "name": "resnet1d_18",
            "params": {"in_channels": 1, "out_dim": 512}
        },
        "head": {
            "name": "classifier",
            "params": {"in_dim": 512, "num_classes": 10}
        },
        "freeze_backbone": false
    }

    Args:
        model_cfg (Dict[str, Any]): 模型配置字典。

    Returns:
        TransferModel: 构建好的模型实例。
    """
    # 1. 构建 Backbone
    backbone_cfg = model_cfg.get("backbone", {})
    backbone_name = backbone_cfg.get("name")
    if not backbone_name:
        raise ValueError("模型配置中必须包含 'backbone.name'")
    
    # 确保导入了具体的 backbone 模块以触发注册
    # (在实际项目中，建议在 main 或 __init__ 统一导入，这里为了稳健尝试动态导入)
    import src.models.backbones.resnet1d
    
    backbone = create_backbone(backbone_name, **backbone_cfg.get("params", {}))

    # 2. 构建 Head
    head_cfg = model_cfg.get("head", {})
    head_name = head_cfg.get("name")
    if not head_name:
        raise ValueError("模型配置中必须包含 'head.name'")
    
    # 确保导入了具体的 head 模块
    import src.models.heads.classifier
    
    head = create_head(head_name, **head_cfg.get("params", {}))

    # 3. 组装 TransferModel
    freeze_backbone = model_cfg.get("freeze_backbone", False)
    
    model = TransferModel(backbone, head, freeze_backbone=freeze_backbone)
    
    return model
