# -*- coding: utf-8 -*-
"""
==============================================================================
DANN训练脚本
==============================================================================
功能：完整的DANN域对抗训练流程
- 数据加载
- 模型训练（分类损失 + 域判别损失）
- GRL系数调度
- 验证与测试
- 模型保存

作者：UAV-DANN项目
日期：2025年

训练流程概述：
-------------
1. 加载预处理数据（源域HIL + 目标域Real）
2. 初始化DANN模型
3. 每个epoch:
   a. 遍历DANN联合数据加载器（同时获取源域和目标域数据）
   b. 源域：计算分类损失（交叉熵）
   c. 源域+目标域：计算域判别损失（二元交叉熵）
   d. 总损失 = 分类损失 + λ * 域损失
   e. 更新GRL的λ系数
4. 验证与测试
5. 保存最优模型
==============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import yaml
from typing import Dict, Tuple, Optional
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.dataloader import get_dataloaders, UAVDataset
from data.preprocess import DataPreprocessor, load_processed_data
from models.dann import DANN, build_dann_from_config
from models.layers import compute_grl_lambda
from utils.metrics import calculate_metrics, MetricsTracker, get_confusion_matrix, plot_confusion_matrix
from utils.logger import setup_logger, TensorBoardLogger, print_epoch_summary


def set_seed(seed: int) -> None:
    """
    设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 启用确定性模式（可能略微降低性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> torch.device:
    """
    获取计算设备
    
    Args:
        config: 配置字典
        
    Returns:
        device: torch设备对象
    """
    if config['device']['use_gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        print(f"[设备] 使用GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("[设备] 使用CPU")
    return device


def train_one_epoch(
    model: DANN,
    dann_loader,
    optimizer: optim.Optimizer,
    cls_criterion: nn.Module,
    domain_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    config: dict
) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        model: DANN模型
        dann_loader: DANN联合数据加载器
        optimizer: 优化器
        cls_criterion: 分类损失函数
        domain_criterion: 域判别损失函数
        device: 计算设备
        epoch: 当前epoch
        total_epochs: 总epoch数
        config: 配置字典
    
    Returns:
        metrics: 训练指标字典
    
    训练维度变化：
    -------------
    输入批次 (从DANN加载器):
        x_source: (B, T, F) = (32, 100, 21)
        y_source: (B,) = (32,)
        d_source: (B,) = (32,) 全为0
        
        x_target: (B, T, F) = (32, 100, 21)
        d_target: (B,) = (32,) 全为1
    
    前向传播:
        outputs = model(x_source, x_target)
        - class_logits: (B, 11) = (32, 11)
        - domain_logits_source: (B, 1) = (32, 1)
        - domain_logits_target: (B, 1) = (32, 1)
    
    损失计算:
        L_cls = CrossEntropy(class_logits, y_source)
        L_domain = BCE(cat(domain_logits_s, domain_logits_t), cat(0s, 1s))
        L_total = L_cls + λ * L_domain
    """
    model.train()
    
    # 记录指标
    total_loss = 0.0
    total_cls_loss = 0.0
    total_domain_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0
    
    # 计算当前epoch的GRL系数λ
    warmup_epochs = config['training']['domain_adaptation']['warmup_epochs']
    if epoch < warmup_epochs:
        # 预热阶段：不进行域适应
        grl_lambda = 0.0
    else:
        # 正式训练：λ从0逐渐增加到1
        gamma_grl = config['training']['domain_adaptation']['gamma_grl']
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        grl_lambda = compute_grl_lambda(
            epoch=int(progress * 100),
            total_epochs=100,
            gamma=gamma_grl
        )
    
    # 更新模型的GRL系数
    model.set_grl_alpha(grl_lambda)
    
    # 创建进度条
    pbar = tqdm(dann_loader, desc=f'Epoch {epoch}/{total_epochs} [Train]', ncols=100)
    
    for source_batch, target_batch in pbar:
        # ==================== 数据准备 ====================
        # 源域数据
        x_source, y_source, d_source = source_batch
        x_source = x_source.to(device)  # (B, T, F)
        y_source = y_source.to(device)  # (B,)
        
        # 目标域数据
        x_target, _, d_target = target_batch
        x_target = x_target.to(device)  # (B, T, F)
        
        # 域标签
        batch_size = x_source.size(0)
        domain_source = torch.zeros(batch_size, 1).to(device)  # 源域=0
        domain_target = torch.ones(x_target.size(0), 1).to(device)  # 目标域=1
        
        # ==================== 前向传播 ====================
        optimizer.zero_grad()
        
        outputs = model(x_source, x_target)
        
        # 提取输出
        class_logits = outputs['class_logits']  # (B, 11)
        domain_logits_source = outputs['domain_logits_source']  # (B, 1)
        domain_logits_target = outputs['domain_logits_target']  # (B, 1)
        
        # ==================== 损失计算 ====================
        # 1. 分类损失（仅源域）
        cls_loss = cls_criterion(class_logits, y_source)
        
        # 2. 域判别损失（源域 + 目标域）
        # 合并域判别输出和标签
        domain_logits = torch.cat([domain_logits_source, domain_logits_target], dim=0)
        domain_labels = torch.cat([domain_source, domain_target], dim=0)
        domain_loss = domain_criterion(torch.sigmoid(domain_logits), domain_labels)
        
        # 3. 总损失
        domain_weight = config['training']['domain_adaptation']['domain_loss_weight']
        total_loss_batch = cls_loss + grl_lambda * domain_weight * domain_loss
        
        # ==================== 反向传播 ====================
        total_loss_batch.backward()
        optimizer.step()
        
        # ==================== 记录指标 ====================
        total_loss += total_loss_batch.item()
        total_cls_loss += cls_loss.item()
        total_domain_loss += domain_loss.item()
        
        # 记录预测结果
        preds = torch.argmax(class_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_source.cpu().numpy())
        
        n_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'dom': f'{domain_loss.item():.4f}',
            'λ': f'{grl_lambda:.3f}'
        })
    
    # 计算平均指标
    avg_loss = total_loss / n_batches
    avg_cls_loss = total_cls_loss / n_batches
    avg_domain_loss = total_domain_loss / n_batches
    
    # 计算准确率
    train_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    metrics = {
        'train_loss': avg_loss,
        'train_cls_loss': avg_cls_loss,
        'train_domain_loss': avg_domain_loss,
        'train_accuracy': train_metrics['accuracy'],
        'grl_lambda': grl_lambda
    }
    
    return metrics


@torch.no_grad()
def evaluate(
    model: DANN,
    dataloader,
    cls_criterion: nn.Module,
    device: torch.device,
    prefix: str = 'val'
) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        model: DANN模型
        dataloader: 数据加载器
        cls_criterion: 分类损失函数
        device: 计算设备
        prefix: 指标前缀 ('val' 或 'target')
    
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0
    
    for batch in dataloader:
        x, y, d = batch
        x = x.to(device)
        y = y.to(device)
        
        # 前向传播（仅特征提取和分类）
        features = model.feature_extractor(x)
        logits = model.classifier(features)
        
        # 计算损失
        loss = cls_criterion(logits, y)
        total_loss += loss.item()
        
        # 记录预测
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        n_batches += 1
    
    # 计算指标
    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    eval_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    metrics = {
        f'{prefix}_loss': avg_loss,
        f'{prefix}_accuracy': eval_metrics['accuracy'],
        f'{prefix}_f1': eval_metrics['f1_score'],
        f'{prefix}_precision': eval_metrics['precision'],
        f'{prefix}_recall': eval_metrics['recall']
    }
    
    # 保存原始预测和标签用于混淆矩阵
    metrics['_preds'] = np.array(all_preds)
    metrics['_labels'] = np.array(all_labels)
    
    return metrics


def save_checkpoint(
    model: DANN,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_path: str,
    is_best: bool = False
) -> None:
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        metrics: 指标
        save_path: 保存路径
        is_best: 是否是最优模型
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"[保存] 最优模型已保存: {best_path}")


def load_checkpoint(
    model: DANN,
    optimizer: Optional[optim.Optimizer],
    checkpoint_path: str,
    device: torch.device
) -> int:
    """
    加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器 (可选)
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        epoch: 恢复的epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"[加载] 从epoch {epoch} 恢复模型: {checkpoint_path}")
    
    return epoch


def train(config_path: str, resume_path: Optional[str] = None) -> None:
    """
    完整训练流程
    
    Args:
        config_path: 配置文件路径
        resume_path: 恢复训练的检查点路径 (可选)
    """
    # ==================== 1. 加载配置 ====================
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("UAV-DANN 域对抗迁移学习训练")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    
    # ==================== 2. 设置随机种子 ====================
    seed = config['reproducibility']['seed']
    set_seed(seed)
    print(f"随机种子: {seed}")
    
    # ==================== 3. 设置设备 ====================
    device = get_device(config)
    
    # ==================== 4. 加载数据 ====================
    print("\n>>> 加载数据...")
    
    # 检查是否有预处理数据
    processed_path = os.path.join(config['data']['processed_dir'], 'processed_data.pkl')
    if os.path.exists(processed_path):
        print(f"发现预处理数据: {processed_path}")
        data_dict = load_processed_data(config['data']['processed_dir'])
    else:
        print("未找到预处理数据，开始预处理...")
        preprocessor = DataPreprocessor(config_path=config_path)
        data_dict = preprocessor.process(save_processed=True)
    
    # 创建数据加载器
    loaders = get_dataloaders(config=config, data_dict=data_dict)
    
    # ==================== 5. 初始化模型 ====================
    print("\n>>> 初始化模型...")
    model = DANN(config=config)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # ==================== 6. 初始化优化器和损失函数 ====================
    # 优化器
    optimizer_config = config['training']['optimizer']
    if optimizer_config['name'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay'],
            betas=tuple(optimizer_config['betas'])
        )
    elif optimizer_config['name'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay'],
            momentum=0.9
        )
    elif optimizer_config['name'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 学习率调度器
    scheduler_config = config['training']['scheduler']
    if scheduler_config['name'] == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
    elif scheduler_config['name'] == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 损失函数
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    
    # ==================== 7. 初始化日志 ====================
    log_dir = config['logging']['log_dir']
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = setup_logger('UAV-DANN', log_dir)
    
    tb_logger = TensorBoardLogger(
        log_dir=config['logging']['tensorboard_dir'],
        experiment_name=config['logging']['experiment_name'],
        enabled=config['logging']['use_tensorboard']
    )
    
    metrics_tracker = MetricsTracker()
    
    # ==================== 8. 恢复训练（如果指定） ====================
    start_epoch = 1
    if resume_path is not None and os.path.exists(resume_path):
        start_epoch = load_checkpoint(model, optimizer, resume_path, device) + 1
    
    # ==================== 9. 训练循环 ====================
    num_epochs = config['training']['num_epochs']
    best_target_accuracy = 0.0
    
    print("\n>>> 开始训练...")
    print(f"总轮数: {num_epochs}")
    print(f"批次大小: {config['training']['batch_size']}")
    print("-" * 70)
    
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        
        # -------------------- 训练 --------------------
        train_metrics = train_one_epoch(
            model=model,
            dann_loader=loaders['dann_train'],
            optimizer=optimizer,
            cls_criterion=cls_criterion,
            domain_criterion=domain_criterion,
            device=device,
            epoch=epoch,
            total_epochs=num_epochs,
            config=config
        )
        
        # -------------------- 验证（源域） --------------------
        val_metrics = evaluate(
            model=model,
            dataloader=loaders['source_val'],
            cls_criterion=cls_criterion,
            device=device,
            prefix='val'
        )
        
        # -------------------- 测试（目标域） --------------------
        target_metrics = evaluate(
            model=model,
            dataloader=loaders['target_test'],
            cls_criterion=cls_criterion,
            device=device,
            prefix='target'
        )
        
        # -------------------- 学习率调度 --------------------
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # -------------------- 合并指标 --------------------
        epoch_metrics = {
            **train_metrics,
            **{k: v for k, v in val_metrics.items() if not k.startswith('_')},
            **{k: v for k, v in target_metrics.items() if not k.startswith('_')},
            'learning_rate': current_lr
        }
        
        # -------------------- 更新追踪器 --------------------
        metrics_tracker.update(epoch, epoch_metrics)
        
        # -------------------- TensorBoard记录 --------------------
        tb_logger.log_scalars(epoch_metrics, epoch)
        
        # -------------------- 打印摘要 --------------------
        elapsed_time = time.time() - epoch_start_time
        print_epoch_summary(epoch, num_epochs, epoch_metrics, elapsed_time)
        
        # 日志记录
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['train_loss']:.4f}, "
            f"val_acc={val_metrics['val_accuracy']:.4f}, "
            f"target_acc={target_metrics['target_accuracy']:.4f}, "
            f"λ={train_metrics['grl_lambda']:.3f}"
        )
        
        # -------------------- 保存检查点 --------------------
        # 检查是否是最优模型（基于目标域准确率）
        is_best = target_metrics['target_accuracy'] > best_target_accuracy
        if is_best:
            best_target_accuracy = target_metrics['target_accuracy']
            print(f"[新最优] 目标域准确率: {best_target_accuracy:.4f}")
        
        # 定期保存
        if epoch % config['logging']['save_every_n_epochs'] == 0 or is_best:
            save_path = os.path.join(
                checkpoint_dir,
                f"{config['logging']['experiment_name']}_epoch{epoch}.pth"
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                save_path=save_path,
                is_best=is_best
            )
    
    # ==================== 10. 训练结束 ====================
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    
    # 打印最终结果
    print(metrics_tracker.get_summary())
    
    # 绘制训练曲线
    curve_path = os.path.join(log_dir, 'training_curves.png')
    metrics_tracker.plot_training_curves(save_path=curve_path)
    
    # 绘制混淆矩阵
    class_names = list(config['fault_types']['labels'].values())
    cm = get_confusion_matrix(target_metrics['_labels'], target_metrics['_preds'], num_classes=11)
    cm_path = os.path.join(log_dir, 'confusion_matrix_target.png')
    plot_confusion_matrix(cm, class_names, save_path=cm_path, title='Target Domain Confusion Matrix')
    
    # 关闭TensorBoard
    tb_logger.close()
    
    print(f"\n日志目录: {log_dir}")
    print(f"检查点目录: {checkpoint_dir}")
    print(f"最佳目标域准确率: {best_target_accuracy:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='UAV-DANN 训练脚本')
    parser.add_argument(
        '--config', 
        type=str, 
        default='./config/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的检查点路径'
    )
    
    args = parser.parse_args()
    
    # 获取绝对路径
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    
    train(config_path=args.config, resume_path=args.resume)


if __name__ == "__main__":
    main()
