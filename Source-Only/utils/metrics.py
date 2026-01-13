# =====================================================================
# 评估指标模块
# 计算分类任务的各种评估指标
# =====================================================================

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import FAULT_IDX_TO_LABEL, NUM_CLASSES


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    
    Returns:
        准确率 (0-1)
    """
    return accuracy_score(y_true, y_pred)


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    average: str = "weighted"
) -> Dict[str, float]:
    """
    计算多种分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        average: 多分类平均方式 ("micro", "macro", "weighted")
    
    Returns:
        包含各评估指标的字典
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        normalize: 归一化方式 ("true", "pred", "all", None)
    
    Returns:
        混淆矩阵 [num_classes, num_classes]
    """
    return confusion_matrix(y_true, y_pred, normalize=normalize)


def get_classification_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    output_dict: bool = True
) -> Dict:
    """
    获取详细的分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        output_dict: 是否输出字典格式
    
    Returns:
        分类报告（字典或字符串）
    """
    # 获取实际存在的类别
    labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = [FAULT_IDX_TO_LABEL.get(i, f"Class_{i}") for i in labels]
    
    return classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """
    打印分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    """
    labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = [FAULT_IDX_TO_LABEL.get(i, f"Class_{i}") for i in labels]
    
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )
    
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(report)


class MetricTracker:
    """
    训练过程中的指标跟踪器
    """
    
    def __init__(self):
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "target_acc": [],
            "learning_rate": []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        target_acc: Optional[float] = None,
        lr: Optional[float] = None
    ):
        """
        更新跟踪指标
        """
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        
        if target_acc is not None:
            self.history["target_acc"].append(target_acc)
        
        if lr is not None:
            self.history["learning_rate"].append(lr)
        
        # 更新最佳记录
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
    
    def get_history(self) -> Dict[str, List[float]]:
        """获取完整历史记录"""
        return self.history
    
    def get_best(self) -> Tuple[int, float]:
        """获取最佳epoch和验证准确率"""
        return self.best_epoch, self.best_val_acc


if __name__ == "__main__":
    # 测试代码
    print("测试评估指标...")
    
    # 模拟数据
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1, 2, 3])
    
    # 测试各指标
    print(f"准确率: {compute_accuracy(y_true, y_pred):.4f}")
    
    metrics = compute_metrics(y_true, y_pred)
    print(f"评估指标: {metrics}")
    
    cm = compute_confusion_matrix(y_true, y_pred)
    print(f"混淆矩阵:\n{cm}")
    
    print_classification_report(y_true, y_pred)
