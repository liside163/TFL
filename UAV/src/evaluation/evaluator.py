"""模型评估工具。

提供在给定数据加载器上对分类模型进行推理并计算指标的函数。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

__all__ = ["evaluate_model"]


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将张量转换为 NumPy 数组。

    Args:
        tensor (torch.Tensor): 输入张量。

    Returns:
        np.ndarray: 转换后的数组。
    """

    return tensor.detach().cpu().numpy()


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
) -> Dict[str, Any]:
    """在给定数据加载器上评估分类模型。

    Args:
        model (nn.Module): 已训练好的 PyTorch 模型。
        data_loader (torch.utils.data.DataLoader): 评估数据加载器，batch 中需包含
            "input" (Tensor) 和 "label" (Tensor 或可转换为 Tensor)。
        device (str): 运行设备标识，例如 "cpu" 或 "cuda"。

    Returns:
        Dict[str, Any]: 评估结果字典，包括:
            - ``"accuracy"``: 准确率。
            - ``"macro_f1"``: 宏平均 F1 分数。
            - ``"per_class_precision"``: 每个类别的精确率字典。
            - ``"per_class_recall"``: 每个类别的召回率字典。
            - ``"confusion_matrix"``: 混淆矩阵 (``numpy.ndarray``)。
    """

    model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch.get("input")
            labels = batch.get("label") or batch.get("labels")

            if inputs is None or labels is None:
                raise ValueError("batch 中缺少 'input' 或 'label/labels' 键")

            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            else:
                labels = torch.tensor(labels, device=device)

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
            else:
                inputs = torch.tensor(inputs, device=device)

            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(_to_numpy(preds))
            all_labels.append(_to_numpy(labels))

    if not all_labels:
        raise ValueError("数据加载器为空，无法评估")

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    conf_mat = confusion_matrix(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    per_class_precision = {}
    per_class_recall = {}
    for label, metrics in report.items():
        if label not in {"accuracy", "macro avg", "weighted avg"}:
            per_class_precision[label] = metrics.get("precision", 0.0)
            per_class_recall[label] = metrics.get("recall", 0.0)

    accuracy = accuracy_score(y_true, y_pred) if y_true.size > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": conf_mat,
    }
