"""
Fine-Tune 策略实现。
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F

from src.strategies.transfer_base import BaseTransferStrategy, register_strategy

__all__ = ["FineTuneStrategy"]


@register_strategy("finetune")
class FineTuneStrategy(BaseTransferStrategy):
    """
    微调策略 (Fine-Tuning)。

    支持 Warmup 机制：
    - 在前 `warmup_epochs` 轮，冻结 backbone，只训练 head。
    - 之后解冻 backbone，进行全模型微调。
    
    配置要求 (config.strategy.params):
        warmup_epochs (int): 预热轮数，默认为 0。
    """

    def __init__(self, model, config):
        super().__init__(model, config)
        
        # 解析策略参数
        strategy_cfg = config.get("strategy", {})
        self.params = strategy_cfg.get("params", {})
        self.warmup_epochs = self.params.get("warmup_epochs", 0)
        
        # 初始状态：如果需要 warmup，先冻结 backbone
        # 注意：这里假设 model 具有 backbone 属性 (如 TransferModel)
        # 如果模型结构不同，需要调整
        if self.warmup_epochs > 0 and hasattr(self.model, "backbone"):
            self._freeze_backbone(True)

    def _freeze_backbone(self, freeze: bool):
        """冻结或解冻 backbone。"""
        if not hasattr(self.model, "backbone"):
            return
            
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze
            
        # 注意：BatchNorm 的 running_stats 在 freeze 时通常也应该停止更新 (eval模式)
        # 或者保持 train 模式但参数不更新。
        # 这里仅控制 requires_grad，模式切换在 training_step 或 Trainer 中控制。
        # 严格的 Fine-tuning 通常在 freeze 时将模块设为 eval。
        if freeze:
            self.model.backbone.eval()
        else:
            self.model.backbone.train()

    def on_epoch_start(self, current_epoch: int):
        """
        在每个 Epoch 开始时检查是否需要切换冻结状态。
        """
        # 如果刚好结束 warmup，解冻 backbone
        if self.warmup_epochs > 0 and current_epoch == self.warmup_epochs:
            print(f"[Strategy] Warmup finished at epoch {current_epoch}. Unfreezing backbone.")
            self._freeze_backbone(False)
        
        # 如果处于 warmup 阶段，确保 backbone 是 eval 模式
        if current_epoch < self.warmup_epochs and hasattr(self.model, "backbone"):
            self.model.backbone.eval()

    def training_step(self, batch: Dict[str, Any], batch_idx: int, current_epoch: int = 0) -> Dict[str, Any]:
        """
        计算训练损失。
        """
        x = batch["input"].to(self.device)
        y = batch["label"].to(self.device)

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
