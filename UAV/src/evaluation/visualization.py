"""评估结果可视化工具。"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.manifold import TSNE

__all__ = [
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_feature_tsne",
]


def _ensure_numpy(array: Iterable) -> np.ndarray:
    """确保输入被转换为 ``numpy.ndarray``。

    Args:
        array (Iterable): 任意可迭代或张量。

    Returns:
        np.ndarray: 转换后的数组。
    """

    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    if hasattr(array, "cpu"):
        return array.cpu().numpy()
    return np.asarray(array)


def plot_confusion_matrix(conf_mat: np.ndarray, class_names: List[str], save_path: str) -> None:
    """绘制并保存混淆矩阵。

    Args:
        conf_mat (np.ndarray): 形状为 (num_classes, num_classes) 的混淆矩阵。
        class_names (List[str]): 类别名称列表，长度应与 ``conf_mat`` 维度一致。
        save_path (str): 图像保存路径。
    """

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap=cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(conf_mat.shape[1]),
        yticks=np.arange(conf_mat.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 添加数值标注
    thresh = conf_mat.max() / 2.0 if conf_mat.size else 0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(
                j,
                i,
                format(conf_mat[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_mat[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """绘制训练过程中的 loss 与 accuracy 曲线。

    Args:
        history (Dict[str, List[float]]): 包含训练/验证 loss 与 acc 的历史记录，
            例如 ``{"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...]}``。
        save_path (str): 图像保存路径。
    """

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    epochs = range(1, len(next(iter(history.values()), [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss 曲线
    axes[0].plot(epochs, history.get("train_loss", []), label="Train Loss")
    axes[0].plot(epochs, history.get("val_loss", []), label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Accuracy 曲线
    axes[1].plot(epochs, history.get("train_acc", []), label="Train Acc")
    axes[1].plot(epochs, history.get("val_acc", []), label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_tsne(
    features: np.ndarray,
    labels: Iterable,
    domain_labels: Optional[Iterable] = None,
    save_path: str = "tsne.png",
) -> None:
    """使用 t-SNE 将高维特征可视化到 2D 平面。

    Args:
        features (np.ndarray): 特征矩阵，形状为 (num_samples, feature_dim)。
        labels (Iterable): 样本标签，可转换为 ``numpy.ndarray``。
        domain_labels (Optional[Iterable], optional): 域标签（如源/目标域）。提供时将使用不同标记样式。
        save_path (str, optional): 图像保存路径。默认 ``"tsne.png"``。
    """

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    feats_np = _ensure_numpy(features)
    label_np = _ensure_numpy(labels)
    domain_np = _ensure_numpy(domain_labels) if domain_labels is not None else None

    tsne = TSNE(n_components=2, init="pca", random_state=42)
    reduced = tsne.fit_transform(feats_np)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(label_np)
    markers = ["o", "s", "^", "D", "P", "X", "*", "v"]

    if domain_np is None:
        for idx, label in enumerate(unique_labels):
            mask = label_np == label
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                label=f"Class {label}",
                alpha=0.7,
                s=30,
            )
    else:
        unique_domains = np.unique(domain_np)
        for domain_idx, domain_value in enumerate(unique_domains):
            domain_mask = domain_np == domain_value
            marker = markers[domain_idx % len(markers)]
            for label in unique_labels:
                mask = (label_np == label) & domain_mask
                ax.scatter(
                    reduced[mask, 0],
                    reduced[mask, 1],
                    label=f"Class {label} | Domain {domain_value}",
                    alpha=0.7,
                    s=30,
                    marker=marker,
                )

    ax.set_title("t-SNE of Features")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize="small", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
