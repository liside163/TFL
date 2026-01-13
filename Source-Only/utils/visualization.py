# =====================================================================
# 可视化工具模块
# 生成训练曲线、混淆矩阵热力图、t-SNE特征可视化等
# =====================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import FAULT_IDX_TO_LABEL, RESULT_DIR

# 设置字体支持 - 使用fallback机制避免缺失中文字体的警告
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用默认字体
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass  # 忽略字体配置错误


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "训练曲线"
):
    """
    绘制训练曲线（损失和准确率）
    
    Args:
        history: 包含训练历史的字典
        save_path: 保存路径，如果为None则显示
        title: 图表标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # 损失曲线
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, history["val_loss"], 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('损失曲线', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2 = axes[1]
    ax2.plot(epochs, history["train_acc"], 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, history["val_acc"], 'r-', label='验证准确率', linewidth=2)
    if "target_acc" in history and len(history["target_acc"]) > 0:
        ax2.plot(epochs[:len(history["target_acc"])], history["target_acc"], 
                 'g--', label='目标域准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('准确率曲线', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[Path] = None,
    title: str = "混淆矩阵",
    normalize: bool = True
):
    """
    绘制混淆矩阵热力图
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图表标题
        normalize: 是否归一化
    """
    if class_names is None:
        class_names = [FAULT_IDX_TO_LABEL.get(i, f"Class_{i}") for i in range(cm.shape[0])]
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零
    else:
        cm_normalized = cm
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用seaborn绘制热力图
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': '比例' if normalize else '数量'}
    )
    
    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    domain_labels: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "t-SNE 特征可视化"
):
    """
    使用t-SNE进行特征降维可视化
    
    Args:
        features: 特征矩阵 [N, feature_dim]
        labels: 类别标签 [N]
        domain_labels: 域标签 [N]（用于区分源域和目标域）
        save_path: 保存路径
        title: 图表标题
    """
    from sklearn.manifold import TSNE
    
    print("正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # 根据是否有域标签决定绘图方式
    if domain_labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 按类别着色
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=labels, cmap='tab10', alpha=0.7, s=30
        )
        ax1.set_title('按故障类型着色', fontsize=12)
        ax1.set_xlabel('t-SNE 1', fontsize=10)
        ax1.set_ylabel('t-SNE 2', fontsize=10)
        
        # 添加图例
        unique_labels = np.unique(labels)
        handles1 = [plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor=plt.cm.tab10(i/10), markersize=8,
                    label=FAULT_IDX_TO_LABEL.get(i, f'Class_{i}'))
                    for i in unique_labels]
        ax1.legend(handles=handles1, loc='best', fontsize=8)
        
        # 按域着色
        ax2 = axes[1]
        colors = ['blue' if d == 0 else 'red' for d in domain_labels]
        ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.5, s=30)
        ax2.set_title('按域着色 (蓝=源域, 红=目标域)', fontsize=12)
        ax2.set_xlabel('t-SNE 1', fontsize=10)
        ax2.set_ylabel('t-SNE 2', fontsize=10)
        
        # 添加图例
        handles2 = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor='blue', markersize=8, label='源域 (HIL)'),
                   plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor='red', markersize=8, label='目标域 (Real)')]
        ax2.legend(handles=handles2, loc='best', fontsize=10)
        
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=labels, cmap='tab10', alpha=0.7, s=30
        )
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        
        # 添加图例
        unique_labels = np.unique(labels)
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=plt.cm.tab10(i/10), markersize=8,
                   label=FAULT_IDX_TO_LABEL.get(i, f'Class_{i}'))
                   for i in unique_labels]
        ax.legend(handles=handles, loc='best', fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE可视化已保存到: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_class_distribution(
    labels: np.ndarray,
    title: str = "类别分布",
    save_path: Optional[Path] = None
):
    """
    绘制类别分布柱状图
    
    Args:
        labels: 标签数组
        title: 图表标题
        save_path: 保存路径
    """
    unique, counts = np.unique(labels, return_counts=True)
    class_names = [FAULT_IDX_TO_LABEL.get(i, f"Class_{i}") for i in unique]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(class_names, counts, color='steelblue', edgecolor='black')
    
    # 在柱子上添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('故障类型', fontsize=12)
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"类别分布图已保存到: {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # 测试代码
    print("测试可视化工具...")
    
    # 模拟训练历史
    history = {
        "train_loss": [1.5, 1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.3],
        "val_loss": [1.6, 1.3, 1.0, 0.85, 0.75, 0.7, 0.68, 0.65],
        "train_acc": [0.3, 0.45, 0.6, 0.7, 0.78, 0.83, 0.86, 0.88],
        "val_acc": [0.25, 0.4, 0.55, 0.65, 0.7, 0.73, 0.75, 0.76],
        "target_acc": [0.2, 0.3, 0.35, 0.4, 0.42, 0.45, 0.46, 0.47]
    }
    
    # 测试训练曲线绘制
    plot_training_curves(history, title="测试 - 训练曲线")
    
    # 模拟混淆矩阵
    cm = np.array([
        [45, 3, 1, 0, 1, 0, 0],
        [2, 35, 2, 1, 0, 0, 0],
        [1, 3, 30, 2, 1, 2, 1],
        [0, 1, 2, 28, 1, 2, 1],
        [1, 0, 1, 1, 32, 2, 3],
        [0, 0, 2, 2, 2, 29, 1],
        [0, 0, 1, 1, 2, 1, 40]
    ])
    
    # 测试混淆矩阵绘制
    class_names = ["Motor", "Accelerometer", "Gyroscope", "Magnetometer", 
                   "Barometer", "GPS", "No Fault"]
    plot_confusion_matrix(cm, class_names=class_names, title="测试 - 混淆矩阵")
