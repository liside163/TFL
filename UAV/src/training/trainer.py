"""
通用训练器模块。

负责管理训练循环、验证、模型保存和日志记录。
"""

import os
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.strategies.transfer_base import BaseTransferStrategy

__all__ = ["Trainer"]


class Trainer:
    """
    通用 Trainer 类。

    负责执行训练和验证循环，管理优化器、调度器和模型保存。
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: BaseTransferStrategy,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: str,
        config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        """
        初始化 Trainer。

        Args:
            model (nn.Module): 待训练模型。
            strategy (BaseTransferStrategy): 训练策略。
            optimizer (Optimizer): 优化器。
            scheduler (Optional[Any]): 学习率调度器。
            device (str): 运行设备 ('cpu', 'cuda' 等)。
            config (Dict[str, Any]): 训练配置。
                期望包含:
                - train.epochs (int)
                - train.output_dir (str)
                - train.save_best (bool, default=True)
                - train.tqdm (bool, default=True)
            logger (Optional[Any]): 日志记录器，需实现 log(dict) 方法。
        """
        self.model = model
        self.strategy = strategy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger

        # 移动模型到设备
        self.model.to(self.device)

        # 解析配置
        self.train_cfg = config.get("train", {})
        self.epochs = self.train_cfg.get("epochs", 10)
        self.output_dir = self.train_cfg.get("output_dir", "checkpoints")
        self.save_best = self.train_cfg.get("save_best", True)
        self.use_tqdm = self.train_cfg.get("tqdm", True)
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        self.best_val_acc = 0.0

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        执行完整的训练和验证流程。

        Args:
            train_loader (DataLoader): 训练集加载器。
            val_loader (Optional[DataLoader]): 验证集加载器。
        """
        print(f"Start training for {self.epochs} epochs on {self.device}...")

        for epoch in range(self.epochs):
            # 策略钩子
            if hasattr(self.strategy, "on_epoch_start"):
                self.strategy.on_epoch_start(epoch)

            # 1. 训练一个 Epoch
            train_metrics = self._train_one_epoch(train_loader, epoch)
            
            # 2. 验证
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate(val_loader, epoch)
            
            # 3. 学习率调度
            if self.scheduler:
                # 兼容 ReduceLROnPlateau 和 StepLR
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    val_loss = val_metrics.get("val_loss", train_metrics.get("train_loss"))
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                # 记录 LR
                current_lr = self.optimizer.param_groups[0]["lr"]
                train_metrics["lr"] = current_lr

            # 策略钩子
            if hasattr(self.strategy, "on_epoch_end"):
                self.strategy.on_epoch_end(epoch)

            # 4. 日志记录
            log_dict = {"epoch": epoch + 1, **train_metrics, **val_metrics}
            if self.logger:
                self.logger.log(log_dict)
            
            # 打印摘要
            msg = f"Epoch [{epoch+1}/{self.epochs}] "
            msg += " ".join([f"{k}: {v:.4f}" for k, v in log_dict.items() if isinstance(v, (int, float))])
            print(msg)

            # 5. 模型保存
            if self.save_best and val_loader:
                current_acc = val_metrics.get("val_acc", 0.0)
                if current_acc > self.best_val_acc:
                    self.best_val_acc = current_acc
                    self._save_checkpoint("best_model.pt")
            
            # 也可以保存 last
            self._save_checkpoint("last_model.pt")

    def _train_one_epoch(self, train_loader: DataLoader, epoch_idx: int) -> Dict[str, float]:
        """
        训练单个 Epoch。

        Args:
            train_loader (DataLoader): 训练数据加载器。
            epoch_idx (int): 当前 Epoch 索引。

        Returns:
            Dict[str, float]: 训练指标（平均 loss 等）。
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # 准备进度条
        iterator = train_loader
        if self.use_tqdm:
            iterator = tqdm(train_loader, desc=f"Train Ep {epoch_idx+1}", leave=False)

        for batch_idx, batch in enumerate(iterator):
            # 移动数据到设备 (部分策略可能在内部处理，但通常在此处理更通用)
            # 为了兼容性，这里尽量让 strategy 处理 device 或者假设 strategy.training_step 能处理
            # 我们的策略实现中假设了 strategy 内部处理 .to(device)，或者在这里统一处理
            # 根据之前策略实现，策略内部做了 batch["input"].to(device)。
            # 为了确保万无一失，我们可以先不处理，或者让策略处理。
            # 查看 finetune.py 实现：x = batch["input"].to(self.device)
            # 所以这里不需要手动 .to(device) 整个 batch，直接传给 strategy
            
            self.optimizer.zero_grad()
            
            # 策略执行前向和损失计算
            step_output = self.strategy.training_step(batch, batch_idx, current_epoch=epoch_idx)
            
            loss = step_output["loss"]
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.use_tqdm:
                iterator.set_postfix(loss=loss.item())

        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}

    def _validate(self, val_loader: DataLoader, epoch_idx: int) -> Dict[str, float]:
        """
        验证过程。

        Args:
            val_loader (DataLoader): 验证数据加载器。
            epoch_idx (int): 当前 Epoch 索引。

        Returns:
            Dict[str, float]: 验证指标（loss, acc 等）。
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        iterator = val_loader
        if self.use_tqdm:
            iterator = tqdm(val_loader, desc=f"Val Ep {epoch_idx+1}", leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                step_output = self.strategy.validation_step(batch, batch_idx)
                
                loss = step_output["loss"]
                logits = step_output["logits"]
                labels = step_output["labels"]
                
                total_loss += loss.item()
                
                # 计算准确率 (假设单标签分类)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        acc = correct / total_samples if total_samples > 0 else 0.0
        
        return {"val_loss": avg_loss, "val_acc": acc}

    def _save_checkpoint(self, filename: str):
        """保存模型权重。"""
        save_path = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), save_path)
