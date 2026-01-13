# -*- coding: utf-8 -*-
"""
==============================================================================
单工况 DANN 训练脚本
==============================================================================
功能：针对单一飞行状态（工况）进行 HIL→REAL 域适应训练

工况 (飞行状态) 共6种:
  0=hover, 1=waypoint, 2=velocity, 3=circling, 4=acce, 5=dece

使用方式：
---------
# 训练 hover 工况
python train_single_condition.py --condition 0

# 训练 waypoint 工况
python train_single_condition.py --condition 1

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import yaml
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.preprocess_single_condition import SingleConditionPreprocessor, load_single_condition_data
from data.dataloader import UAVDataset, DANNDataLoader
from models.dann import build_dann_from_config
from utils.metrics import calculate_metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_dual_confusion_matrix(source_labels, source_preds, target_labels, target_preds,
                                class_names, save_path, condition_name):
    """
    绘制源域和目标域的双混淆矩阵
    
    Args:
        source_labels: 源域真实标签
        source_preds: 源域预测标签
        target_labels: 目标域真实标签
        target_preds: 目标域预测标签
        class_names: 类别名称列表
        save_path: 保存路径
        condition_name: 工况名称
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 源域混淆矩阵
    cm_source = confusion_matrix(source_labels, source_preds)
    sns.heatmap(cm_source, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f'Source Domain (HIL) - {condition_name}\nAccuracy: {(source_preds == source_labels).mean():.2%}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    # 目标域混淆矩阵
    cm_target = confusion_matrix(target_labels, target_preds)
    sns.heatmap(cm_target, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_title(f'Target Domain (Real) - {condition_name}\nAccuracy: {(target_preds == target_labels).mean():.2%}',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[信息] 双混淆矩阵已保存: {save_path}")


class FocalLoss(nn.Module):
    """
    Focal Loss - 解决类别不平衡问题
    
    论文: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    公式: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        gamma: 聚焦参数，越大越关注难分类样本 (默认2.0)
        alpha: 类别权重张量，用于处理类别不平衡
        reduction: 'mean' 或 'sum'
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 模型输出 logits, shape = (N, C)
            targets: 真实标签, shape = (N,)
        Returns:
            focal loss 值
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    计算类别权重（平方根逆频率加权 - 更保守）
    
    使用平方根缓解极端权重：
    - 原始逆频率: weight = 1/count → 极端不平衡时差异巨大 (如0.08 vs 1.36)
    - 平方根逆频率: weight = 1/sqrt(count) → 差异更温和 (如0.28 vs 1.16)
    
    Args:
        y: 标签数组
        num_classes: 类别数
        device: 设备
        
    Returns:
        权重张量, shape = (num_classes,)
    """
    class_counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    # 避免除零
    class_counts = np.maximum(class_counts, 1.0)
    
    # ========== 方案2：平方根逆频率权重（更保守）==========
    # 使用平方根缓解极端权重，避免过度惩罚/奖励某些类别
    weights = 1.0 / np.sqrt(class_counts)
    
    # 归一化使权重和为 num_classes
    weights = weights / weights.sum() * num_classes
    
    print(f"[信息] 类别分布: {class_counts.astype(int)}")
    print(f"[信息] 平方根权重: {np.round(weights, 3)} (原逆频率权重会导致过度不平衡)")
    
    return torch.FloatTensor(weights).to(device)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing 交叉熵损失
    
    论文: "Rethinking the Inception Architecture for Computer Vision" (Szegedy et al., 2016)
    
    原理: 将硬标签 [0, 0, 1, 0] 软化为 [0.025, 0.025, 0.925, 0.025]
    优势: 防止过拟合，提高泛化能力
    
    Args:
        num_classes: 类别数
        epsilon: 平滑系数 (推荐 0.05-0.2)
        weight: 类别权重（可选）
    """
    def __init__(self, num_classes: int, epsilon: float = 0.1, weight: torch.Tensor = None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.weight = weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits, shape = (B, C)
            targets: labels, shape = (B,)
        Returns:
            loss: 标量
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        with torch.no_grad():
            # 创建平滑标签
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
            smooth_targets = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        
        # 如果有类别权重，应用权重
        if self.weight is not None:
            per_sample_weight = self.weight[targets]
            loss = -(smooth_targets * log_probs).sum(dim=-1) * per_sample_weight
            return loss.mean()
        else:
            loss = -(smooth_targets * log_probs).sum(dim=-1)
            return loss.mean()


def set_seed(seed: int) -> None:
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device(config: dict) -> torch.device:
    """获取计算设备"""
    if config['device']['use_gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        print(f"[信息] 使用GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("[信息] 使用CPU")
    return device


def get_dataloaders_single_condition(data_dict: Dict, config: dict, sc_config: dict) -> Dict:
    """为单工况数据创建数据加载器（可选加权采样）"""
    sc_training = sc_config.get('training', {})
    batch_size = sc_training.get('batch_size', config['training']['batch_size'])

    loss_config = sc_training.get('loss', {})
    loss_type = loss_config.get('classification', 'focal_loss')
    use_class_weights = loss_config.get('use_class_weights')
    if use_class_weights is None:
        use_class_weights = loss_type != 'focal_loss'

    dataloader_config = sc_training.get('dataloader', {})
    use_weighted_sampler = dataloader_config.get('use_weighted_sampler')
    if use_weighted_sampler is None:
        use_weighted_sampler = not (loss_type == 'focal_loss' and not use_class_weights)
    if use_weighted_sampler and use_class_weights:
        print("[警告] 加权采样 + 类别加权损失会双重惩罚大类样本，建议只启用一种。")
    
    source_train_dataset = UAVDataset(data_dict['X_source_train'], data_dict['y_source_train'], domain_label=0)
    source_val_dataset = UAVDataset(data_dict['X_source_val'], data_dict['y_source_val'], domain_label=0)
    target_train_dataset = UAVDataset(data_dict['X_target_train'], data_dict['y_target_train'], domain_label=1)
    target_test_dataset = UAVDataset(data_dict['X_target_test'], data_dict['y_target_test'], domain_label=1)
    
    # ========== 可选加权采样：让小类样本有更高的被采样概率 ==========
    y_source_train = data_dict['y_source_train']
    class_counts = np.bincount(y_source_train)
    sampler = None

    if use_weighted_sampler:
        # 计算每个类别的权重（逆频率）
        class_weights = 1.0 / class_counts
        # 归一化权重，使得平均权重为1
        class_weights = class_weights / class_weights.mean()

        # 为每个样本分配权重
        sample_weights = class_weights[y_source_train]

        # 创建加权采样器
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # 允许重复采样
        )

        print(f"[信息] 使用加权采样 - 类别权重: {class_weights}")
    else:
        print("[信息] 未启用加权采样，使用随机打乱")
    print(f"[信息] 类别分布: {class_counts}")
    
    # 使用sampler时不能设置shuffle=True
    if sampler is not None:
        source_train_loader = DataLoader(
            source_train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # 使用加权采样器
            drop_last=True,
            num_workers=0
        )
    else:
        source_train_loader = DataLoader(
            source_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
    source_val_loader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    dann_loader = DANNDataLoader(source_train_loader, target_train_loader)
    
    return {
        'source_train': source_train_loader,
        'source_val': source_val_loader,
        'target_train': target_train_loader,
        'target_test': target_test_loader,
        'dann_train': dann_loader
    }


def train_one_epoch(model, dann_loader, optimizer, cls_criterion, domain_criterion,
                    device, epoch, total_epochs, config, phase_info=None) -> Dict:
    """
    训练一个epoch - 支持两阶段训练策略
    
    两阶段训练策略：
    ==================== 
    阶段1 (Epochs 1-50)：纯分类训练
      - gamma_grl = 0
      - 只优化分类损失
      - 目标：确保源域准确率 > 90%
    
    阶段2 (Epochs 51-150)：域适应训练
      - gamma_grl 线性增长到 max_gamma
      - 监控目标域准确率
      - 如果源域准确率下降 > 10%，降低 domain_loss_weight
    ====================
    """
    model.train()

    total_cls_loss = 0.0
    total_domain_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    # ==================== 两阶段训练参数 ====================
    da_config = config['training']['domain_adaptation']
    
    # 阶段1参数
    phase1_epochs = da_config.get('phase1_epochs', 50)  # 纯分类阶段
    
    # 阶段2参数
    max_gamma_grl = da_config.get('gamma_grl', 2.0)  # 最终的gamma值
    domain_loss_weight = da_config.get('domain_loss_weight', 0.3)
    
    # ==================== 两阶段GRL调度 ====================
    if epoch <= phase1_epochs:
        # 阶段1：纯分类，不进行域适应
        grl_lambda = 0.0
        phase = 1
    else:
        # 阶段2：域适应，GRL线性增长
        phase = 2
        phase2_progress = (epoch - phase1_epochs) / (total_epochs - phase1_epochs)
        # 线性增长到 max_gamma_grl
        grl_lambda = phase2_progress * max_gamma_grl
        
        # 如果有phase_info，检查源域准确率保护
        if phase_info is not None:
            source_acc_drop = phase_info.get('source_acc_drop', 0.0)
            if source_acc_drop > 0.10:  # 源域准确率下降超过10%
                # 降低域损失权重
                domain_loss_weight = domain_loss_weight * 0.5
    
    model.set_grl_alpha(grl_lambda)  # 设置梯度反转层的 alpha 值
    
    for source_batch, target_batch in dann_loader:
        x_source, y_source, _ = source_batch
        x_target, _, _ = target_batch
        
        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_source, x_target)
        
        cls_loss = cls_criterion(outputs['class_logits'], y_source)
        
        # 阶段1：只优化分类损失
        if phase == 1:
            loss = cls_loss
            domain_loss = torch.tensor(0.0).to(device)
        else:
            # 阶段2：加入域适应损失
            domain_source = torch.zeros(x_source.size(0), 1).to(device)
            domain_target = torch.ones(x_target.size(0), 1).to(device)
            domain_logits = torch.cat([outputs['domain_logits_source'], outputs['domain_logits_target']], dim=0)
            domain_labels = torch.cat([domain_source, domain_target], dim=0)
            domain_loss = domain_criterion(domain_logits, domain_labels)
            
            loss = cls_loss + domain_loss_weight * grl_lambda * domain_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_cls_loss += cls_loss.item()
        total_domain_loss += domain_loss.item() if isinstance(domain_loss, torch.Tensor) else domain_loss
        total_loss += loss.item()
        num_batches += 1
    
    return {
        'cls_loss': total_cls_loss / num_batches,
        'domain_loss': total_domain_loss / num_batches,
        'total_loss': total_loss / num_batches,
        'grl_lambda': grl_lambda,
        'phase': phase
    }


def evaluate(model, dataloader, cls_criterion, device, prefix='val') -> Dict:
    """评估模型性能"""
    model.eval()
    
    all_preds, all_labels = [], []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            
            outputs = model(x)
            loss = cls_criterion(outputs['class_logits'], y)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs['class_logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            num_batches += 1
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 诊断：检查预测类别分布
    pred_dist = np.bincount(all_preds, minlength=7)
    label_dist = np.bincount(all_labels, minlength=7)
    unique_preds = len(np.unique(all_preds))
    if unique_preds <= 2:
        print(f"  ⚠️ [{prefix}] 类别坍塌! 只预测{unique_preds}类: pred={pred_dist}, label={label_dist}")
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics[f'{prefix}_loss'] = total_loss / num_batches
    
    return metrics


def extract_features(model, dataloader, device) -> tuple:
    """
    提取模型特征用于 t-SNE 可视化

    Returns:
        features: 特征向量 (N, feature_dim)
        labels: 标签 (N,)
        domain_labels: 域标签 (N,) - 0=source, 1=target
    """
    model.eval()
    all_features, all_labels, all_domains = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x, y, d = batch
            x = x.to(device)

            # ==================== 修复：正确提取特征 ====================
            # 直接调用特征提取器，而不是使用 model(x) 的输出
            # 因为 model(x) 只返回 class_logits 和 domain_logits，不返回特征
            try:
                # 尝试直接调用特征提取器
                features = model.feature_extractor(x)
            except AttributeError:
                # 如果没有 feature_extractor 属性，使用其他方式
                outputs = model(x)
                if 'features_source' in outputs:
                    features = outputs['features_source']
                elif 'features' in outputs:
                    features = outputs['features']
                else:
                    # 最后的备选方案：使用 class_logits 作为特征（虽然不理想）
                    features = outputs['class_logits']

            all_features.append(features.cpu().numpy())
            all_labels.extend(y.numpy())
            all_domains.extend(d.numpy())

    return np.concatenate(all_features, axis=0), np.array(all_labels), np.array(all_domains)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: list, save_path: str, 
                          condition_name: str = "") -> None:
    """
    绘制并保存混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
        condition_name: 工况名称
    """
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互模式
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # 标签
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names[:cm.shape[1]],
               yticklabels=class_names[:cm.shape[0]],
               title=f'混淆矩阵 - {condition_name}',
               ylabel='真实标签',
               xlabel='预测标签')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 在每个格子中显示数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[信息] 混淆矩阵已保存: {save_path}")
        
    except Exception as e:
        print(f"[警告] 混淆矩阵绘制失败: {e}")


def plot_tsne_visualization(source_features: np.ndarray, source_labels: np.ndarray,
                            target_features: np.ndarray, target_labels: np.ndarray,
                            class_names: list, save_path: str,
                            condition_name: str = "") -> None:
    """
    绘制 t-SNE 特征可视化图
    
    Args:
        source_features: 源域特征
        source_labels: 源域标签
        target_features: 目标域特征
        target_labels: 目标域标签
        class_names: 类别名称
        save_path: 保存路径
        condition_name: 工况名称
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 合并源域和目标域特征
        all_features = np.vstack([source_features, target_features])
        all_labels = np.concatenate([source_labels, target_labels])
        domain_labels = np.array([0] * len(source_labels) + [1] * len(target_labels))
        
        # 如果样本太多，随机采样
        max_samples = 2000
        if len(all_features) > max_samples:
            idx = np.random.choice(len(all_features), max_samples, replace=False)
            all_features = all_features[idx]
            all_labels = all_labels[idx]
            domain_labels = domain_labels[idx]
        
        print(f"[信息] 正在计算 t-SNE (样本数: {len(all_features)})...")
        
        # 兼容新版scikit-learn: n_iter改为max_iter
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        features_2d = tsne.fit_transform(all_features)
        
        # 绘制两个子图
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 子图1: 按域区分
        colors_domain = ['#1f77b4', '#ff7f0e']
        for d, (color, label) in enumerate(zip(colors_domain, ['HIL (Source)', 'Real (Target)'])):
            mask = domain_labels == d
            axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=color, label=label, alpha=0.6, s=15)
        axes[0].set_title(f't-SNE Domain Distribution - {condition_name}')
        axes[0].legend()
        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')
        
        # 子图2: 按类别区分
        num_classes = len(class_names)
        cmap = plt.cm.get_cmap('tab10', num_classes)
        for c in range(num_classes):
            mask = all_labels == c
            if mask.sum() > 0:
                axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                               c=[cmap(c)], label=class_names[c], alpha=0.6, s=15)
        axes[1].set_title(f't-SNE Class Distribution - {condition_name}')
        axes[1].legend(loc='best', fontsize=8)
        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[信息] t-SNE 可视化已保存: {save_path}")
        
    except Exception as e:
        print(f"[警告] t-SNE 可视化失败: {e}")


def final_evaluation_with_visualization(
    model, source_loader, target_loader, cls_criterion, device,
    condition: int, condition_name: str, results_dir: str, config: dict
) -> Dict:
    """
    最终评估并生成可视化
    
    Args:
        model: 训练好的模型
        source_loader: 源域验证数据加载器
        target_loader: 目标域测试数据加载器
        cls_criterion: 分类损失函数
        device: 设备
        condition: 工况代码
        condition_name: 工况名称
        results_dir: 结果保存目录
        config: 配置
        
    Returns:
        评估指标字典
    """
    print("\n>>> 最终评估与可视化...")
    
    # ========== 修复：使用正确的类别名称（与标签0-6严格对应）==========
    class_names = config['fault_types'].get('names', 
                  ['No_Fault', 'Motor', 'Accelerometer', 'Gyroscope', 
                   'Magnetometer', 'Barometer', 'GPS'])
    
    # 评估目标域
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in target_loader:
            x, y, _ = batch
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs['class_logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # ========== 新增：评估源域并生成预测 ==========
    source_preds_cm, source_labels_cm = [], []
    with torch.no_grad():
        for batch in source_loader:
            x, y, _ = batch
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs['class_logits'], dim=1)
            source_preds_cm.extend(preds.cpu().numpy())
            source_labels_cm.extend(y.numpy())
    
    source_preds_cm = np.array(source_preds_cm)
    source_labels_cm = np.array(source_labels_cm)
    
    # 创建可视化目录
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. 生成双混淆矩阵（源域 + 目标域）
    cm_path = os.path.join(vis_dir, f'confusion_matrix_condition_{condition}.png')
    plot_dual_confusion_matrix(
        source_labels_cm, source_preds_cm,  # 源域
        all_labels, all_preds,              # 目标域
        class_names, cm_path, condition_name
    )
    
    # 2. t-SNE 可视化
    try:
        source_features, source_labels, _ = extract_features(model, source_loader, device)
        target_features, target_labels, _ = extract_features(model, target_loader, device)
        
        tsne_path = os.path.join(vis_dir, f'tsne_condition_{condition}.png')
        plot_tsne_visualization(source_features, source_labels, 
                               target_features, target_labels,
                               class_names, tsne_path, condition_name)
    except Exception as e:
        print(f"[警告] t-SNE 特征提取失败: {e}")
    
    # ========== 3. 域对齐分析 ==========
    # 评估源域准确率
    source_preds, source_labels_eval = [], []
    with torch.no_grad():
        for batch in source_loader:
            x, y, _ = batch
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs['class_logits'], dim=1)
            source_preds.extend(preds.cpu().numpy())
            source_labels_eval.extend(y.numpy())
    
    source_metrics = calculate_metrics(np.array(source_labels_eval), np.array(source_preds))
    target_metrics_final = calculate_metrics(all_labels, all_preds)
    
    # 计算域差距
    domain_gap = source_metrics['accuracy'] - target_metrics_final['accuracy']
    
    # 打印域对齐分析报告
    print("\n" + "=" * 60)
    print("域对齐分析报告")
    print("=" * 60)
    print(f"源域 (HIL) 准确率:   {source_metrics['accuracy']:.2%}")
    print(f"源域 (HIL) F1分数:   {source_metrics['f1_score']:.4f}")
    print(f"目标域 (Real) 准确率: {target_metrics_final['accuracy']:.2%}")
    print(f"目标域 (Real) F1分数: {target_metrics_final['f1_score']:.4f}")
    print("-" * 60)
    print(f"域差距 (Source - Target): {domain_gap:+.2%}")
    
    if abs(domain_gap) < 0.1:
        alignment_status = "✅ 优秀 - 域特征对齐良好"
    elif abs(domain_gap) < 0.2:
        alignment_status = "⚠️ 一般 - 存在一定域偏移"
    else:
        alignment_status = "❌ 较差 - 域偏移严重，需优化"
    print(f"对齐状态: {alignment_status}")
    print("=" * 60)
    
    # 返回增强的指标
    final_metrics = target_metrics_final.copy()
    final_metrics['source_accuracy'] = source_metrics['accuracy']
    final_metrics['source_f1_score'] = source_metrics['f1_score']
    final_metrics['domain_gap'] = domain_gap
    
    return final_metrics


def train_single_condition(config_path: str, sc_config_path: str, condition: int, resume_path: Optional[str] = None) -> Dict:
    """
    单工况迁移训练主函数
    
    Args:
        config_path: 主配置文件路径
        sc_config_path: 单工况配置文件路径
        condition: 飞行状态代码 (0-5)
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    with open(sc_config_path, 'r', encoding='utf-8') as f:
        sc_config = yaml.safe_load(f)
    
    condition_name = sc_config['conditions']['names'].get(condition, f"Condition_{condition}")
    
    print("=" * 70)
    print(f"UAV-DANN 单工况迁移训练")
    print(f"飞行状态: {condition} ({condition_name})")
    print("=" * 70)
    
    set_seed(config['reproducibility']['seed'])
    device = get_device(config)
    
    # 加载或预处理数据
    results_dir = sc_config['output']['results_dir']
    processed_path = os.path.join(results_dir, f'processed_data_condition_{condition}.pkl')
    
    if os.path.exists(processed_path):
        print("\n>>> 加载已处理的数据...")
        data_dict = load_single_condition_data(condition, results_dir)
    else:
        print("\n>>> 开始数据预处理...")
        preprocessor = SingleConditionPreprocessor(
            config_path=config_path,
            sc_config_path=sc_config_path,
            condition=condition
        )
        data_dict = preprocessor.process(save_processed=True)
    
    # 创建数据加载器
    print("\n>>> 创建数据加载器...")
    loaders = get_dataloaders_single_condition(data_dict, config, sc_config)
    
    from models.dann_deep import DANNDeep

    # 创建模型
    print("\n>>> 创建模型...")
    
    # 检查是否使用深度配置
    mh = sc_config.get('training', {}).get('model_hyperparameters')
    
    if mh:
        print("[信息] 检测到深度模型超参数，使用动态 DANNDeep...")
        n_features = config['preprocessing']['n_features']
        seq_len = config['preprocessing']['window_size']
        num_classes = config['fault_types']['num_classes']
        
        # 提取参数，提供默认值
        cnn_conf = mh.get('cnn', {})
        lstm_conf = mh.get('lstm', {})
        clf_conf = mh.get('classifier', {})
        disc_conf = mh.get('discriminator', {})
        
        model = DANNDeep(
            n_features=n_features,
            seq_len=seq_len,
            num_classes=num_classes,
            cnn_layers=cnn_conf.get('num_layers', 2),
            cnn_channels=cnn_conf.get('channels', [64, 128]),
            lstm_hidden=lstm_conf.get('hidden_size', 128),
            lstm_layers=lstm_conf.get('num_layers', 2),
            lstm_dropout=lstm_conf.get('dropout', 0.5),
            lstm_bidirectional=lstm_conf.get('bidirectional', False),
            classifier_layers=clf_conf.get('num_layers', 2),
            classifier_hidden=clf_conf.get('hidden_dim', 64),
            classifier_dropout=clf_conf.get('dropout', 0.5),
            discriminator_layers=disc_conf.get('num_layers', 2),
            discriminator_hidden=disc_conf.get('hidden_dim', 64),
            # 模型架构开关
            use_layernorm=mh.get('use_layernorm', True),  # 新增
            use_batchnorm=mh.get('use_batchnorm', False),
            use_attention=mh.get('use_attention', True),
            use_residual=mh.get('use_residual', True)
        )
    else:
        print("[信息] 使用标准配置构建模型...")
        model = build_dann_from_config(config_path)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[信息] 模型参数: {total_params:,}")
    
    # 优化器 (优先使用工况配置，否则使用主配置)
    sc_training = sc_config.get('training', {})
    sc_optimizer = sc_training.get('optimizer', {})
    lr = float(sc_optimizer.get('learning_rate', config['training']['optimizer']['learning_rate']))
    weight_decay = float(sc_optimizer.get('weight_decay', config['training']['optimizer']['weight_decay']))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器 (优先使用工况配置)
    sc_scheduler = sc_training.get('scheduler', config['training'].get('scheduler', {}))
    scheduler_name = sc_scheduler.get('name', 'step')
    
    if scheduler_name == 'warmup_cosine':
        warmup_epochs = sc_scheduler.get('warmup_epochs', 5)
        num_epochs = sc_training.get('num_epochs', config['training']['num_epochs'])
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # ========== 计算类别权重（可选用于损失） ==========
    num_classes = config['fault_types']['num_classes']
    class_weights = compute_class_weights(data_dict['y_source_train'], num_classes, device)
    
    # 从配置中读取损失函数类型
    loss_config = sc_training.get('loss', {})
    loss_type = loss_config.get('classification', 'focal_loss')
    use_class_weights = loss_config.get('use_class_weights')
    if use_class_weights is None:
        use_class_weights = loss_type != 'focal_loss'
    
    if loss_type == 'label_smoothing':
        epsilon = loss_config.get('label_smoothing_epsilon', 0.1)
        cls_criterion = LabelSmoothingCrossEntropy(
            num_classes=num_classes,
            epsilon=epsilon,
            weight=class_weights if use_class_weights else None
        )
        weight_note = " + 类别加权" if use_class_weights else ""
        print(f"[信息] 使用 LabelSmoothingCE (epsilon={epsilon}){weight_note}")
    elif loss_type == 'cross_entropy':
        cls_criterion = nn.CrossEntropyLoss(weight=class_weights if use_class_weights else None)
        weight_note = " + 类别加权" if use_class_weights else ""
        print(f"[信息] 使用 CrossEntropyLoss{weight_note}")
    else:
        focal_gamma = loss_config.get('focal_gamma', 2.0)
        cls_criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights if use_class_weights else None)
        weight_note = " + 类别加权" if use_class_weights else ""
        print(f"[信息] 使用 FocalLoss (gamma={focal_gamma}){weight_note}")
    
    domain_criterion = nn.BCEWithLogitsLoss()
    
    num_epochs = sc_training.get('num_epochs', config['training']['num_epochs'])
    early_stopping_patience = sc_training.get('early_stopping_patience', 15)
    
    checkpoint_dir = sc_config['output']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练
    print("\n>>> 开始训练...")
    print(f"两阶段训练策略: 阶段1(纯分类, 1-50), 阶段2(域适应, 51-{num_epochs})")
    best_target_acc = -float('inf')  # 初始化为负无穷，确保能保存第一个满足条件的模型
    best_epoch = 0
    patience_counter = 0
    history = []
    
    # 两阶段训练：记录阶段1的最佳源域准确率
    phase1_best_source_acc = 0.0
    phase_info = None
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # 计算phase_info用于源域准确率保护
        if epoch > 50 and phase1_best_source_acc > 0:
            current_source_acc = history[-1]['source_val_acc'] if history else 0.0
            source_acc_drop = phase1_best_source_acc - current_source_acc
            phase_info = {'source_acc_drop': source_acc_drop}
        
        # 🔧 修复：传入sc_config和phase_info
        train_metrics = train_one_epoch(model, loaders['dann_train'], optimizer, cls_criterion, domain_criterion, device, epoch, num_epochs, sc_config, phase_info)
        val_metrics = evaluate(model, loaders['source_val'], cls_criterion, device, 'val')
        target_metrics = evaluate(model, loaders['target_test'], cls_criterion, device, 'target')
        
        # 阶段1：记录最佳源域准确率
        if epoch <= 50:
            phase1_best_source_acc = max(phase1_best_source_acc, val_metrics['accuracy'])
        
        if scheduler is not None:
            scheduler.step()
        
        # ========== 调试：验证数据加载是否正确 ==========
        if epoch == 1:
            print("\n=== 数据验证 (Epoch 1) ===")
            print(f"Source train size: {len(loaders['source_train'].dataset)}")
            print(f"Source val size: {len(loaders['source_val'].dataset)}")
            print(f"Target test size: {len(loaders['target_test'].dataset)}")
            
            # 检查一个batch的数据
            test_batch = next(iter(loaders['source_train']))
            test_x, test_y, test_d = test_batch
            print(f"\n样本维度检查:")
            print(f"  X shape: {test_x.shape}")  # 应该是 (batch, seq_len, n_features)
            print(f"  Y shape: {test_y.shape}, 值范围: [{test_y.min()}, {test_y.max()}]")
            print(f"  Y分布: {torch.bincount(test_y)}")
            
            # 检查模型输出
            model.eval()
            with torch.no_grad():
                test_out = model(test_x.to(device))
                test_logits = test_out['class_logits']
                test_preds = torch.argmax(test_logits, dim=1)
                print(f"\n模型输出检查:")
                print(f"  Logits shape: {test_logits.shape}")
                print(f"  Logits范围: [{test_logits.min():.3f}, {test_logits.max():.3f}]")
                print(f"  预测分布: {torch.bincount(test_preds.cpu())}")
            model.train()
            print("=" * 60 + "\n")
        
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_metrics['total_loss'],
            'source_val_acc': val_metrics['accuracy'],              # 源域验证准确率
            'target_acc': target_metrics['accuracy'],
            'target_f1': target_metrics['f1_score'],
            'grl_lambda': train_metrics['grl_lambda'],
            # 域差距分析 (使用验证集而不是训练集)
            'domain_gap': val_metrics['accuracy'] - target_metrics['accuracy']
        }
        history.append(epoch_record)
        
        # 增强打印：显示验证集准确率和当前阶段
        domain_gap = val_metrics['accuracy'] - target_metrics['accuracy']
        phase_str = f"P{train_metrics.get('phase', 1)}"
        print(f"[{phase_str}] Epoch {epoch:3d}/{num_epochs} | Loss: {train_metrics['total_loss']:.4f} | "
              f"Source: {val_metrics['accuracy']:.2%} | Target: {target_metrics['accuracy']:.2%} | "
              f"Gap: {domain_gap:+.2%} | F1: {target_metrics['f1_score']:.3f} | λ: {train_metrics['grl_lambda']:.3f}")
        
        # ========== 改进3：模型选择逻辑（源域约束） ==========
        # 只有当源域准确率>50%时，才考虑目标域表现
        source_acc = val_metrics['accuracy']
        target_acc = target_metrics['accuracy']
        
        # 综合评分：源域>50%才有资格作为最佳模型
        if source_acc >= 0.5:
            # 综合得分: 70%目标域 + 30%源域
            current_score = 0.7 * target_acc + 0.3 * source_acc
        else:
            # 源域崩溃：给负分，不考虑作为最佳模型
            current_score = -1.0
        
        is_best = current_score > best_target_acc and source_acc >= 0.5
        if is_best:
            best_target_acc = current_score  # 实际存储综合得分
            best_epoch = epoch
            patience_counter = 0
            
            # ========== 改进4：完善checkpoint保存 ==========
            save_path = os.path.join(checkpoint_dir, f'condition_{condition}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': {
                    'source_acc': source_acc,
                    'target_acc': target_acc,
                    'target_f1': target_metrics['f1_score'],
                    'score': current_score
                },
                'condition': condition,
                'config': {
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'loss_type': loss_type
                }
            }, save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"\n[信息] 早停! Best epoch: {best_epoch}, Best score: {best_target_acc:.4f}")
            break
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"训练完成! 工况: {condition} ({condition_name})")
    print(f"最佳Epoch: {best_epoch}, 综合得分: {best_target_acc:.4f} (0.7×Target + 0.3×Source)")
    print(f"训练时间: {training_time/60:.2f} 分钟")
    print("=" * 70)
    
    # ========== 生成可视化 ==========
    # 加载最佳模型进行可视化
    best_model_path = os.path.join(checkpoint_dir, f'condition_{condition}_best.pth')
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[信息] 已加载最佳模型 (Epoch {checkpoint['epoch']})")
        except RuntimeError as e:
            print(f"[警告] 无法加载checkpoint（可能模型架构不匹配）: {e}")
            print(f"[信息] 使用当前训练最后的模型进行评估")
    else:
        if best_epoch == 0:
            print(f"[警告] 整个训练过程源域准确率未超过50%，无最佳模型保存")
            print(f"[信息] 使用当前训练最后的模型进行评估")
    
    # 执行最终评估与可视化
    final_metrics = final_evaluation_with_visualization(
        model=model,
        source_loader=loaders['source_val'],
        target_loader=loaders['target_test'],
        cls_criterion=cls_criterion,
        device=device,
        condition=condition,
        condition_name=condition_name,
        results_dir=results_dir,
        config=config
    )
    
    # 保存结果
    results = {
        'condition': condition,
        'condition_name': condition_name,
        'best_epoch': best_epoch,
        'best_target_acc': best_target_acc,
        'final_metrics': final_metrics,
        'training_time': training_time,
        'history': history
    }
    
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'training_results_condition_{condition}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[信息] 结果已保存至: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='UAV-DANN 单工况迁移训练')
    
    parser.add_argument('--condition', type=int, default=0,
                        help='飞行状态代码 (0-5): 0=hover, 1=waypoint, 2=velocity, 3=circling, 4=acce, 5=dece')
    parser.add_argument('--config', type=str, default='./config/config.yaml')
    parser.add_argument('--sc_config', type=str, default=None,
                        help='单工况配置文件路径 (默认自动选择: config/condition_{N}_{name}.yaml)')
    
    args = parser.parse_args()
    
    if args.condition not in range(6):
        print(f"[错误] 飞行状态代码必须在0-5之间")
        return
    
    # 工况名称映射
    condition_names = {0: 'hover', 1: 'waypoint', 2: 'velocity', 3: 'circling', 4: 'acce', 5: 'dece'}
    
    # 自动选择对应工况的配置文件
    if args.sc_config is None:
        cond_name = condition_names[args.condition]
        args.sc_config = f'./config/condition_{args.condition}_{cond_name}.yaml'
        print(f"[信息] 使用工况专属配置: {args.sc_config}")
    
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    if not os.path.isabs(args.sc_config):
        args.sc_config = os.path.join(project_root, args.sc_config)
    
    # 检查配置文件是否存在
    if not os.path.exists(args.sc_config):
        print(f"[错误] 配置文件不存在: {args.sc_config}")
        return
    
    train_single_condition(args.config, args.sc_config, args.condition)


if __name__ == "__main__":
    main()

