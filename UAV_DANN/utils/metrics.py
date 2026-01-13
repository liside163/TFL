# -*- coding: utf-8 -*-
"""
==============================================================================
评估指标模块
==============================================================================
功能：计算模型评估指标
- 准确率 (Accuracy)
- F1分数 (F1-Score)
- 精确率 (Precision)
- 召回率 (Recall)
- 混淆矩阵 (Confusion Matrix)

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted',
    labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签，shape = (N,)
        y_pred: 预测标签，shape = (N,)
        average: F1/Precision/Recall的平均方式
                 'micro': 全局统计
                 'macro': 各类别平均
                 'weighted': 按类别样本数加权平均
        labels: 类别名称列表 (可选)
    
    Returns:
        metrics: 包含各项指标的字典
            - 'accuracy': 准确率
            - 'f1_score': F1分数
            - 'precision': 精确率
            - 'recall': 召回率
    
    指标说明：
    ----------
    Accuracy = 正确预测数 / 总样本数
    
    Precision = TP / (TP + FP) → 预测为正的样本中，实际为正的比例
    
    Recall = TP / (TP + FN) → 实际为正的样本中，被正确预测的比例
    
    F1 = 2 * Precision * Recall / (Precision + Recall) → 精确率和召回率的调和平均
    """
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 计算各项指标
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 11
) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        y_true: 真实标签，shape = (N,)
        y_pred: 预测标签，shape = (N,)
        num_classes: 类别数量
    
    Returns:
        cm: 混淆矩阵，shape = (num_classes, num_classes)
            cm[i, j] = 真实类别为i，预测为j的样本数
    
    混淆矩阵解读：
    -------------
    对角线元素: 正确分类的样本数
    非对角线元素: 错误分类的样本数
    
    行(i)表示真实类别，列(j)表示预测类别
    """
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = 'Confusion Matrix',
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    绘制混淆矩阵热力图
    
    Args:
        cm: 混淆矩阵，shape = (num_classes, num_classes)
        class_names: 类别名称列表
        save_path: 保存路径 (可选)
        title: 图标题
        normalize: 是否归一化 (显示百分比)
        figsize: 图像尺寸
    
    Returns:
        fig: matplotlib Figure对象
    """
    if normalize:
        # 按行归一化 (每个真实类别的总样本数)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        cm_display = cm_normalized
        fmt = '.2f'
        vmax = 1.0
    else:
        cm_display = cm
        fmt = 'd'
        vmax = cm.max()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=vmax,
        square=True,
        cbar_kws={'label': 'Ratio' if normalize else 'Count'}
    )
    
    # 设置标签
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[信息] 混淆矩阵已保存至: {save_path}")
    
    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    num_classes: int = None
) -> str:
    """
    打印详细的分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        num_classes: 类别数量（用于处理某些类别缺失的情况）
    
    Returns:
        report: 分类报告字符串
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 确定labels参数，确保包含所有可能的类别
    if num_classes is None:
        num_classes = len(class_names)
    
    labels = list(range(num_classes))
    
    report = classification_report(
        y_true, y_pred,
        labels=labels,  # 明确指定所有可能的标签
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    print("\n" + "=" * 60)
    print("分类报告 (Classification Report)")
    print("=" * 60)
    print(report)
    
    return report


def compute_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 11
) -> Dict[int, float]:
    """
    计算每个类别的准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数量
    
    Returns:
        class_acc: 每个类别的准确率字典
            key: 类别索引
            value: 该类别的准确率
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    class_acc = {}
    
    for c in range(num_classes):
        mask = (y_true == c)
        if mask.sum() > 0:
            class_acc[c] = (y_pred[mask] == c).mean()
        else:
            class_acc[c] = 0.0
    
    return class_acc


class MetricsTracker:
    """
    指标追踪器
    
    用于在训练过程中追踪和记录各项指标
    
    Attributes:
        history: 历史指标记录
    """
    
    def __init__(self):
        """初始化追踪器"""
        self.history = {
            'train_loss': [],
            'train_cls_loss': [],
            'train_domain_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'target_accuracy': [],
            'target_f1': []
        }
        
        self.best_metrics = {
            'best_val_accuracy': 0.0,
            'best_target_accuracy': 0.0,
            'best_target_f1': 0.0,
            'best_epoch': 0
        }
    
    def update(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        更新指标记录
        
        Args:
            epoch: 当前轮数
            metrics: 指标字典
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        # 更新最佳指标
        if 'val_accuracy' in metrics:
            if metrics['val_accuracy'] > self.best_metrics['best_val_accuracy']:
                self.best_metrics['best_val_accuracy'] = metrics['val_accuracy']
                self.best_metrics['best_epoch'] = epoch
        
        if 'target_accuracy' in metrics:
            if metrics['target_accuracy'] > self.best_metrics['best_target_accuracy']:
                self.best_metrics['best_target_accuracy'] = metrics['target_accuracy']
        
        if 'target_f1' in metrics:
            if metrics['target_f1'] > self.best_metrics['best_target_f1']:
                self.best_metrics['best_target_f1'] = metrics['target_f1']
    
    def plot_training_curves(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        绘制训练曲线
        
        Args:
            save_path: 保存路径 (可选)
            figsize: 图像尺寸
        
        Returns:
            fig: matplotlib Figure对象
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. 损失曲线
        ax = axes[0, 0]
        if self.history['train_loss']:
            ax.plot(epochs, self.history['train_loss'], 'b-', label='Total Loss')
        if self.history['train_cls_loss']:
            ax.plot(epochs, self.history['train_cls_loss'], 'g--', label='Classification Loss')
        if self.history['train_domain_loss']:
            ax.plot(epochs, self.history['train_domain_loss'], 'r--', label='Domain Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 源域准确率
        ax = axes[0, 1]
        if self.history['train_accuracy']:
            ax.plot(epochs, self.history['train_accuracy'], 'b-', label='Train')
        if self.history['val_accuracy']:
            ax.plot(epochs, self.history['val_accuracy'], 'g-', label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Source Domain Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 目标域准确率
        ax = axes[1, 0]
        if self.history['target_accuracy']:
            ax.plot(epochs, self.history['target_accuracy'], 'r-', label='Target Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Target Domain Accuracy (Real)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. F1分数
        ax = axes[1, 1]
        if self.history['val_f1']:
            ax.plot(epochs, self.history['val_f1'], 'g-', label='Val F1')
        if self.history['target_f1']:
            ax.plot(epochs, self.history['target_f1'], 'r-', label='Target F1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[信息] 训练曲线已保存至: {save_path}")
        
        return fig
    
    def get_summary(self) -> str:
        """
        获取指标摘要
        
        Returns:
            summary: 摘要字符串
        """
        summary = "\n" + "=" * 60 + "\n"
        summary += "训练指标摘要\n"
        summary += "=" * 60 + "\n"
        summary += f"最佳验证准确率: {self.best_metrics['best_val_accuracy']:.4f}\n"
        summary += f"最佳目标域准确率: {self.best_metrics['best_target_accuracy']:.4f}\n"
        summary += f"最佳目标域F1: {self.best_metrics['best_target_f1']:.4f}\n"
        summary += f"最佳轮次: {self.best_metrics['best_epoch']}\n"
        summary += "=" * 60 + "\n"
        
        return summary


if __name__ == "__main__":
    """
    测试评估指标模块
    """
    print("=" * 60)
    print("评估指标模块测试")
    print("=" * 60)
    
    # 模拟预测结果
    np.random.seed(42)
    num_samples = 100
    num_classes = 11
    
    y_true = np.random.randint(0, num_classes, size=num_samples)
    # 模拟约70%准确率
    y_pred = y_true.copy()
    noise_idx = np.random.choice(num_samples, size=int(num_samples * 0.3), replace=False)
    y_pred[noise_idx] = np.random.randint(0, num_classes, size=len(noise_idx))
    
    # 测试指标计算
    print("\n>>> 测试指标计算:")
    metrics = calculate_metrics(y_true, y_pred)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试混淆矩阵
    print("\n>>> 测试混淆矩阵:")
    cm = get_confusion_matrix(y_true, y_pred, num_classes)
    print(f"  混淆矩阵 shape: {cm.shape}")
    print(f"  对角线元素 (正确分类): {np.diag(cm)}")
    
    # 测试类别准确率
    print("\n>>> 测试类别准确率:")
    class_acc = compute_class_accuracy(y_true, y_pred, num_classes)
    for c, acc in class_acc.items():
        print(f"  Class {c}: {acc:.4f}")
    
    # 类别名称
    class_names = [
        'No_Fault', 'Motor', 'Propeller', 'Low_Voltage', 
        'Wind_Affect', 'Load_Lose', 'Accelerometer', 
        'Gyroscope', 'Magnetometer', 'Barometer', 'GPS'
    ]
    
    # 打印分类报告
    print_classification_report(y_true, y_pred, class_names)
    
    print("\n测试完成！")
