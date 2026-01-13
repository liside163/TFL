"""
基础训练策略模块。

定义所有训练策略的基类，以及策略注册表。
"""

from typing import Any, Callable, Dict, Type

import torch
import torch.nn as nn

__all__ = ["BaseTransferStrategy", "STRATEGY_REGISTRY", "register_strategy", "create_strategy"]

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, Type["BaseTransferStrategy"]] = {}


def register_strategy(name: str) -> Callable:
    """
    策略类装饰器，用于注册训练策略。

    Args:
        name (str): 注册名称。
    """
    def decorator(cls: Type["BaseTransferStrategy"]) -> Type["BaseTransferStrategy"]:
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"训练策略 '{name}' 已被注册: {STRATEGY_REGISTRY[name]}")
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def create_strategy(name: str, model: nn.Module, config: Dict[str, Any]) -> "BaseTransferStrategy":
    """
    创建策略实例。

    Args:
        name (str): 策略注册名称。
        model (nn.Module): 待训练的模型。
        config (Dict[str, Any]): 全局配置或策略相关配置。

    Returns:
        BaseTransferStrategy: 实例化的策略对象。
    """
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"训练策略 '{name}' 未找到。可用策略: {available}")
    return STRATEGY_REGISTRY[name](model, config)


# -----------------------------------------------------------------------------
# Base Strategy
# -----------------------------------------------------------------------------

class BaseTransferStrategy:
    """
    训练策略基类。

    封装了一次训练 / 验证 step 的计算逻辑。
    Trainer 只负责循环和日志，本类负责：
      - 前向传播
      - 计算损失
      - 返回日志字典
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        初始化策略。

        Args:
            model (nn.Module): 待训练的模型 (通常是 TransferModel)。
            config (Dict[str, Any]): 配置字典。
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    def training_step(self, batch: Dict[str, Any], batch_idx: int, current_epoch: int = 0) -> Dict[str, Any]:
        """
        执行单个训练步。

        Args:
            batch (Dict[str, Any]): 从 DataLoader 获取的一个 batch 数据。
                通常包含: {"input": Tensor, "label": Tensor, ...}
            batch_idx (int): 当前 Batch 索引。
            current_epoch (int): 当前 Epoch 索引 (从 0 开始)。

        Returns:
            Dict[str, Any]: 包含损失和日志信息的字典。
                必须包含 "loss" 键用于反向传播。
                建议包含 "logits", "labels" 用于计算指标。
        """
        raise NotImplementedError

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        执行单个验证步。

        Args:
            batch (Dict[str, Any]): 验证集数据。
            batch_idx (int): Batch 索引。

        Returns:
            Dict[str, Any]: 包含评估结果的字典。
                通常包含 "loss", "logits", "labels"。
        """
        raise NotImplementedError

    def on_epoch_start(self, current_epoch: int):
        """
        每个 Epoch 开始时的钩子（可选）。
        """
        pass
    
    def on_epoch_end(self, current_epoch: int):
        """
        每个 Epoch 结束时的钩子（可选）。
        """
        pass
