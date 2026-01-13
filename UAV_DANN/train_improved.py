# -*- coding: utf-8 -*-
"""
==============================================================================
改进版DANN训练脚本
==============================================================================
功能：解决35%目标域准确率问题的改进训练流程
- 类别权重 + Focal Loss（处理类别不平衡）
- 保守的GRL调度（降低gamma_grl）
- 目标域验证集监控
- 早停机制
- 学习率Warmup

改进内容：
-------------
1. 添加类别权重计算（逆频率加权）
2. 使用Focal Loss替代标准CE Loss
3. 降低gamma_grl从17.7→3.0
4. 延长warmup从5→20 epochs
5. 添加目标域验证和早停
6. 降低学习率从0.0068→0.0001

作者：UAV-DANN项目改进
日期：2025年
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
from torch.optim.lr_scheduler import LambdaLR
import yaml
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from copy import deepcopy

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.dataloader import get_dataloaders, UAVDataset
from data.preprocess import DataPreprocessor, load_processed_data
from models.dann import DANN, build_dann_from_config
from models.layers import compute_grl_lambda
from utils.metrics import calculate_metrics, MetricsTracker, get_confusion_matrix, plot_confusion_matrix
from utils.logger import setup_logger, TensorBoardLogger, print_epoch_summary
from utils.losses import compute_class_weights, FocalLoss, LabelSmoothingCrossEntropy


def set_seed(seed: int) -> None:
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> torch.device:
    """获取计算设备"""
    if config['device']['use_gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        print(f"[设备] 使用GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("[设备] 使用CPU")
    return device


class EarlyStopping:
    """
    早停机制

    基于目标域准确率监控，若连续patience个epoch无改善则停止训练
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = 'max'
    ):
        """
        Args:
            patience: 容忍无改善的epoch数
            min_delta: 最小改善幅度
            mode: 'max' (越大越好) 或 'min' (越小越好)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Args:
            score: 当前评估分数
            model: 当前模型

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = deepcopy(model.state_dict())
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_model_state = deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def load_best_model(self, model: nn.Module) -> None:
        """加载最佳模型状态"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def get_lr_scheduler_with_warmup(
    optimizer: optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6
) -> LambdaLR:
    """
    创建带warmup的学习率调度器

    Args:
        optimizer: 优化器
        num_epochs: 总训练轮数
        warmup_epochs: warmup轮数
        min_lr: 最小学习率

    Returns:
        scheduler: LambdaLR调度器
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Warmup阶段：线性增长
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine退火
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return max(min_lr, 0.5 * (1 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


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
    训练一个epoch（改进版）

    改进：
    - 使用Focal Loss或加权CE Loss
    - 保守的GRL调度
    """
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_domain_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0

    # ==================== 改进的GRL调度 ====================
    # 使用更保守的参数
    improved_gamma = config['training'].get('domain_adaptation', {}).get('gamma_grl_improved', 3.0)
    warmup_epochs = config['training'].get('domain_adaptation', {}).get('warmup_epochs_improved', 20)

    if epoch < warmup_epochs:
        # 预热阶段：不进行域适应
        grl_lambda = 0.0
    else:
        # 正式训练：使用更保守的gamma
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        grl_lambda = compute_grl_lambda(
            epoch=int(progress * 100),
            total_epochs=100,
            gamma=improved_gamma  # 使用改进的gamma（默认3.0）
        )

    # 更新模型的GRL系数
    model.set_grl_alpha(grl_lambda)

    # 创建进度条
    pbar = tqdm(dann_loader, desc=f'Epoch {epoch}/{total_epochs} [Train]', ncols=120)

    for source_batch, target_batch in pbar:
        # ==================== 数据准备 ====================
        x_source, y_source, d_source = source_batch
        x_source = x_source.to(device)
        y_source = y_source.to(device)

        x_target, _, d_target = target_batch
        x_target = x_target.to(device)

        batch_size = x_source.size(0)
        domain_source = torch.zeros(batch_size, 1).to(device)
        domain_target = torch.ones(x_target.size(0), 1).to(device)

        # ==================== 前向传播 ====================
        optimizer.zero_grad()

        outputs = model(x_source, x_target)

        class_logits = outputs['class_logits']
        domain_logits_source = outputs['domain_logits_source']
        domain_logits_target = outputs['domain_logits_target']

        # ==================== 损失计算 ====================
        # 1. 分类损失（使用Focal Loss或加权CE）
        cls_loss = cls_criterion(class_logits, y_source)

        # 2. 域判别损失
        domain_logits = torch.cat([domain_logits_source, domain_logits_target], dim=0)
        domain_labels = torch.cat([domain_source, domain_target], dim=0)
        domain_loss = domain_criterion(torch.sigmoid(domain_logits), domain_labels)

        # 3. 总损失（使用改进的域损失权重）
        domain_weight = config['training'].get('domain_adaptation', {}).get('domain_loss_weight_improved', 0.5)
        total_loss_batch = cls_loss + grl_lambda * domain_weight * domain_loss

        # ==================== 反向传播 ====================
        total_loss_batch.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ==================== 记录指标 ====================
        total_loss += total_loss_batch.item()
        total_cls_loss += cls_loss.item()
        total_domain_loss += domain_loss.item()

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

    train_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))

    metrics = {
        'train_loss': avg_loss,
        'train_cls_loss': avg_cls_loss,
        'train_domain_loss': avg_domain_loss,
        'train_accuracy': train_metrics['accuracy'],
        'train_f1': train_metrics['f1_score'],
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
    """评估模型性能"""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0

    for batch in dataloader:
        x, y, d = batch
        x = x.to(device)
        y = y.to(device)

        features = model.feature_extractor(x)
        logits = model.classifier(features)

        loss = cls_criterion(logits, y)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    eval_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))

    metrics = {
        f'{prefix}_loss': avg_loss,
        f'{prefix}_accuracy': eval_metrics['accuracy'],
        f'{prefix}_f1': eval_metrics['f1_score'],
        f'{prefix}_precision': eval_metrics['precision'],
        f'{prefix}_recall': eval_metrics['recall']
    }

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
    """保存模型检查点"""
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


def train_improved(config_path: str, resume_path: Optional[str] = None) -> None:
    """
    改进的训练流程

    关键改进：
    1. 类别权重 + Focal Loss
    2. 保守的GRL调度
    3. 目标域验证 + 早停
    4. 学习率Warmup
    """
    # ==================== 1. 加载配置 ====================
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("UAV-DANN 改进版训练")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    print("\n改进内容:")
    print("  - 类别权重 + Focal Loss (处理类别不平衡)")
    print("  - 保守GRL调度 (gamma=3.0, warmup=20)")
    print("  - 目标域验证 + 早停")
    print("  - 学习率Warmup")
    print("=" * 70)

    # ==================== 2. 设置随机种子 ====================
    seed = config['reproducibility']['seed']
    set_seed(seed)

    # ==================== 3. 设置设备 ====================
    device = get_device(config)

    # ==================== 4. 加载数据 ====================
    print("\n>>> 加载数据...")

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

    # ==================== 5. 计算类别权重 ====================
    print("\n>>> 计算类别权重...")
    # 从训练集获取标签
    train_source_dataset = loaders['train_source_dataset']
    train_labels = train_source_dataset.y.numpy()
    num_classes = config['model']['classifier']['num_classes']

    # 计算类别权重
    class_weights = compute_class_weights(train_labels, num_classes=num_classes)
    class_weights = class_weights.to(device)

    # ==================== 6. 初始化模型 ====================
    print("\n>>> 初始化模型...")
    model = DANN(config=config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ==================== 7. 初始化优化器和损失函数 ====================

    # 使用更低的学习率
    improved_lr = config['training'].get('optimizer', {}).get('learning_rate_improved', 1e-4)
    print(f"使用改进学习率: {improved_lr}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=improved_lr,
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    # 学习率调度器（带Warmup）
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training'].get('domain_adaptation', {}).get('warmup_epochs_improved', 20)
    scheduler = get_lr_scheduler_with_warmup(
        optimizer,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs
    )

    # 损失函数 - 使用Focal Loss
    focal_gamma = config['training'].get('focal_gamma', 2.0)
    cls_criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    print(f"使用Focal Loss (gamma={focal_gamma})")

    domain_criterion = nn.BCELoss()

    # ==================== 8. 早停机制 ====================
    early_stopping = EarlyStopping(
        patience=20,
        min_delta=0.001,
        mode='max'
    )

    # ==================== 9. 初始化日志 ====================
    log_dir = config['logging']['log_dir']
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = setup_logger('UAV-DANN-Improved', log_dir)

    tb_logger = TensorBoardLogger(
        log_dir=config['logging']['tensorboard_dir'],
        experiment_name=config['logging']['experiment_name'] + '_improved',
        enabled=config['logging']['use_tensorboard']
    )

    metrics_tracker = MetricsTracker()

    # ==================== 10. 训练循环 ====================
    start_epoch = 1
    if resume_path is not None and os.path.exists(resume_path):
        # TODO: 实现恢复训练
        pass

    best_target_accuracy = 0.0
    best_source_accuracy = 0.0

    print("\n>>> 开始训练...")
    print(f"总轮数: {num_epochs}")
    print(f"批次大小: {config['training']['batch_size']}")
    print(f"Warmup轮数: {warmup_epochs}")
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

        # -------------------- 验证（目标域） - 关键改进 --------------------
        target_val_metrics = evaluate(
            model=model,
            dataloader=loaders['target_val'],
            cls_criterion=cls_criterion,
            device=device,
            prefix='target_val'
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
            **{k: v for k, v in target_val_metrics.items() if not k.startswith('_')},
            **{k: v for k, v in target_metrics.items() if not k.startswith('_')},
            'learning_rate': current_lr
        }

        # -------------------- TensorBoard记录 --------------------
        tb_logger.log_scalars(epoch_metrics, epoch)

        # -------------------- 打印摘要 --------------------
        elapsed_time = time.time() - epoch_start_time

        print(f"\nEpoch {epoch}/{num_epochs} ({elapsed_time:.1f}s)")
        print(f"  Train: Loss={train_metrics['train_loss']:.4f}, "
              f"Acc={train_metrics['train_accuracy']:.4f}, F1={train_metrics['train_f1']:.4f}")
        print(f"  源域验证: Loss={val_metrics['val_loss']:.4f}, "
              f"Acc={val_metrics['val_accuracy']:.4f}, F1={val_metrics['val_f1']:.4f}")
        print(f"  目标域验证: Loss={target_val_metrics['target_val_loss']:.4f}, "
              f"Acc={target_val_metrics['target_val_accuracy']:.4f}, F1={target_val_metrics['target_val_f1']:.4f}")
        print(f"  目标域测试: Acc={target_metrics['target_accuracy']:.4f}, "
              f"F1={target_metrics['target_f1']:.4f}")
        print(f"  λ={train_metrics['grl_lambda']:.3f}, LR={current_lr:.6f}")

        # -------------------- 早停检查（基于目标域验证准确率） --------------------
        target_val_acc = target_val_metrics['target_val_accuracy']
        is_best = target_val_acc > best_target_accuracy

        if is_best:
            best_target_accuracy = target_val_acc
            print(f"  [新最优] 目标域验证准确率: {best_target_accuracy:.4f}")

        if val_metrics['val_accuracy'] > best_source_accuracy:
            best_source_accuracy = val_metrics['val_accuracy']

        # 早停检查
        if early_stopping(target_val_acc, model):
            print(f"\n[早停] {early_stopping.patience}轮无改善，停止训练")
            early_stopping.load_best_model(model)
            break

        # -------------------- 保存检查点 --------------------
        if epoch % config['logging']['save_every_n_epochs'] == 0 or is_best:
            save_path = os.path.join(
                checkpoint_dir,
                f"{config['logging']['experiment_name']}_improved_epoch{epoch}.pth"
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                save_path=save_path,
                is_best=is_best
            )

    # ==================== 11. 训练结束 ====================
    print("\n" + "=" * 70)
    print("改进版训练完成!")
    print("=" * 70)
    print(f"\n最佳结果:")
    print(f"  源域验证准确率: {best_source_accuracy:.4f}")
    print(f"  目标域验证准确率: {best_target_accuracy:.4f}")

    # 最终评估
    print("\n>>> 最终评估...")
    final_target_metrics = evaluate(
        model=model,
        dataloader=loaders['target_test'],
        cls_criterion=cls_criterion,
        device=device,
        prefix='final_target'
    )
    print(f"最终目标域测试准确率: {final_target_metrics['final_target_accuracy']:.4f}")
    print(f"最终目标域测试F1分数: {final_target_metrics['final_target_f1']:.4f}")

    # 绘制混淆矩阵
    class_names = list(config['fault_types']['labels'].values())
    cm = get_confusion_matrix(
        final_target_metrics['_labels'],
        final_target_metrics['_preds'],
        num_classes=num_classes
    )
    cm_path = os.path.join(log_dir, 'confusion_matrix_target_improved.png')
    plot_confusion_matrix(cm, class_names, save_path=cm_path, title='Target Domain (Improved)')

    tb_logger.close()

    print(f"\n日志目录: {log_dir}")
    print(f"检查点目录: {checkpoint_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='UAV-DANN 改进版训练脚本')
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

    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)

    train_improved(config_path=args.config, resume_path=args.resume)


if __name__ == "__main__":
    main()
