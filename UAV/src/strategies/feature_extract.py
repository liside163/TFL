"""
Feature-Extract 策略实现。
"""

from typing import Any, Dict

import torch.nn.functional as F

from src.strategies.transfer_base import BaseTransferStrategy, register_strategy

__all__ = ["FeatureExtractStrategy"]


@register_strategy("feature_extract")
class FeatureExtractStrategy(BaseTransferStrategy):
    """
    特征提取策略 (Feature Extraction / Linear Probing)。

    假定 Backbone 已经被冻结（在模型构建阶段或初始化阶段），
    本策略只负责常规的训练步骤，不涉及动态解冻参数。
    适用于只训练分类头的情况。
    """

    def __init__(self, model, config):
        super().__init__(model, config)
        
        # 强制确保 backbone 冻结（双重保险）
        if hasattr(self.model, "backbone"):
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        self.model.backbone.eval()

    def training_step(self, batch: Dict[str, Any], batch_idx: int, current_epoch: int = 0) -> Dict[str, Any]:
        """
        计算训练损失。
        """
        x = batch["input"].to(self.device)
        y = batch["label"].to(self.device)

        # 确保 backbone 在训练过程中保持 eval 模式
        if hasattr(self.model, "backbone"):
            self.model.backbone.eval()

        # 前向传播
        outputs = self.model(x)
        logits = outputs["logits"]

        # 计算损失
        loss = F.cross_entropy(logits, y)

        return {
            "loss": loss,
            "logits": logits.detach(),
            "labels": y,
        }

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        计算验证损失。
        """
        x = batch["input"].to(self.device)
        y = batch["label"].to(self.device)

        outputs = self.model(x)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, y)

        return {
            "loss": loss,
            "logits": logits.detach(),
            "labels": y,
        }
