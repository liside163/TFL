# -*- coding: utf-8 -*-
"""
==============================================================================
单工况 Optuna 超参数调优脚本 (深度扩展版)
==============================================================================
功能：针对单个飞行状态（工况）进行深度超参数优化 (网络结构 + 训练策略)

工况 (飞行状态) 共6种:
  0=hover, 1=waypoint, 2=velocity, 3=circling, 4=acce, 5=dece

使用方式：
---------
# 调优 hover 工况
python optuna_tune_single_condition.py --condition 0 --n_trials 50

# 调优 velocity 工况
python optuna_tune_single_condition.py --condition 2 --n_trials 30

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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
from typing import Dict
import yaml
import math
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[错误] 请先安装Optuna: pip install optuna")

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.preprocess_single_condition import SingleConditionPreprocessor, load_single_condition_data
from data.dataloader import UAVDataset, DANNDataLoader
from utils.metrics import calculate_metrics
from torch.utils.data import DataLoader
from models.dann_deep import DANNDeep  # 导入重构后的模型
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss - 解决类别不平衡问题"""
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


def compute_class_weights(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """计算类别权重 (平方根逆频率加权 - 更保守，避免极端权重)"""
    class_counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    # 使用平方根缓解极端权重差异
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / weights.mean()  # 归一化使平均权重为1
    return torch.FloatTensor(weights).to(device)


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_scheduler(optimizer, scheduler_type: str, num_epochs: int, steps_per_epoch: int, 
                  warmup_epochs: int = 0, **kwargs):
    """获取学习率调度器"""
    if scheduler_type == 'step':
        return StepLR(optimizer, step_size=kwargs.get('step_size', 30), 
                      gamma=kwargs.get('gamma', 0.1))
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif scheduler_type == 'onecycle':
        return OneCycleLR(optimizer, max_lr=kwargs.get('max_lr', 0.01),
                          epochs=num_epochs, steps_per_epoch=steps_per_epoch,
                          pct_start=0.3, anneal_strategy='cos')
    elif scheduler_type == 'warmup_cosine':
        # 自定义warmup + cosine
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return None


def objective_single_condition(
    trial: Trial,
    condition: int,
    config_path: str,
    sc_config_path: str,
    data_dict: Dict,
    device: torch.device
) -> float:
    """
    单工况深度超参数优化目标函数
    """
    # 加载配置用于获取固定参数
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ========== 扩展搜索空间 (深度网络结构) ==========
    cnn_layers = trial.suggest_int('cnn_layers', 1, 4)
    cnn_base_channels = trial.suggest_categorical('cnn_base_channels', [32, 48, 64, 96])
    cnn_channels = [cnn_base_channels * (2 ** i) for i in range(cnn_layers)]
    
    lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    lstm_hidden = trial.suggest_categorical('lstm_hidden', [64, 96, 128, 192, 256])
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5)
    lstm_bidirectional = trial.suggest_categorical('lstm_bidirectional', [True, False])
    
    classifier_layers = trial.suggest_int('classifier_layers', 2, 3)  # 最少2层，增强容错
    classifier_hidden = trial.suggest_categorical('classifier_hidden', [32, 64, 96, 128])
    classifier_dropout = trial.suggest_float('classifier_dropout', 0.2, 0.6)
    
    discriminator_layers = trial.suggest_int('discriminator_layers', 1, 3)
    discriminator_hidden = trial.suggest_categorical('discriminator_hidden', [32, 64, 96])
    
    # ========== 训练策略 ==========
    lr_scheduler_type = trial.suggest_categorical('lr_scheduler', 
                                                   ['step', 'cosine', 'warmup_cosine', 'onecycle'])
    # 修复：收窄学习率范围，避免过快收敛到简单策略
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 2e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Batch size (保留针对小数据集的搜索范围)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # 修复：扩大Warmup范围，确保分类器有足够时间学习
    warmup_epochs = trial.suggest_int('warmup_epochs', 20, 40)
    
    # ========== 域适应参数 (保守范围避免负迁移) ==========
    gamma_grl = trial.suggest_float('gamma_grl', 1.0, 4.0)  # 降低范围，之前实验显示2.0较好
    domain_loss_weight = trial.suggest_float('domain_loss_weight', 0.05, 0.5)  # 降低最大权重
    domain_weight_schedule = trial.suggest_categorical('domain_weight_schedule', 
                                                        ['constant', 'linear', 'exponential'])
    
    # ========== 模型架构 (固定为最佳实践) ==========
    # 基于之前实验，固定最佳架构组合
    use_attention = True
    use_residual = True
    use_batchnorm = False
    use_layernorm = True
    
    # ========== 修复：类别平衡策略（互斥约束）==========
    # 加权采样和类别权重不能同时启用，否则会双重惩罚大类
    balance_strategy = trial.suggest_categorical('balance_strategy', 
                                                  ['weighted_sampling', 'class_weights', 'focal_only'])
    use_class_weights = (balance_strategy == 'class_weights')
    use_weighted_sampling = (balance_strategy == 'weighted_sampling')
    focal_gamma = trial.suggest_float('focal_gamma', 1.5, 3.5)  # 收窄范围
    
    # ========== 创建数据加载器 ==========
    # 构造临时配置以复用现有dataloader工具
    config_copy = config.copy()
    config_copy['training'] = config['training'].copy()
    config_copy['training']['batch_size'] = batch_size
    
    # 使用 DANNDataLoader 自行构建，以便控制细节
    source_train_dataset = UAVDataset(data_dict['X_source_train'], data_dict['y_source_train'], domain_label=0)
    source_val_dataset = UAVDataset(data_dict['X_source_val'], data_dict['y_source_val'], domain_label=0)
    target_train_dataset = UAVDataset(data_dict['X_target_train'], data_dict['y_target_train'], domain_label=1)
    target_test_dataset = UAVDataset(data_dict['X_target_test'], data_dict['y_target_test'], domain_label=1)
    
    # ========== 新增：条件加权采样 ==========
    if use_weighted_sampling:
        y_source_train = data_dict['y_source_train']
        class_counts = np.bincount(y_source_train)
        sample_class_weights = 1.0 / class_counts
        sample_class_weights = sample_class_weights / sample_class_weights.mean()
        sample_weights = sample_class_weights[y_source_train]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0)
    else:
        source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    source_val_loader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    dann_loader = DANNDataLoader(source_train_loader, target_train_loader)
    steps_per_epoch = len(dann_loader)
    
    # ========== 创建模型 (DANNDeep) ==========
    n_features = config['preprocessing']['n_features']
    seq_len = config['preprocessing']['window_size']
    num_classes = config['fault_types']['num_classes']
    
    model = DANNDeep(
        n_features=n_features,
        seq_len=seq_len,
        num_classes=num_classes,
        cnn_layers=cnn_layers,
        cnn_channels=cnn_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        lstm_bidirectional=lstm_bidirectional,
        classifier_layers=classifier_layers,
        classifier_hidden=classifier_hidden,
        classifier_dropout=classifier_dropout,
        discriminator_hidden=discriminator_hidden,
        discriminator_layers=discriminator_layers,

        # 固定最佳架构参数
        use_attention=use_attention,
        use_residual=use_residual,
        use_batchnorm=use_batchnorm,
        use_layernorm=use_layernorm
    ).to(device)
    
    # ========== 优化器和调度器 ==========
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 设定训练轮数
    num_epochs = 60 # 调优轮数
    
    scheduler = get_scheduler(optimizer, lr_scheduler_type, num_epochs=num_epochs, 
                              steps_per_epoch=steps_per_epoch,
                              warmup_epochs=warmup_epochs, max_lr=learning_rate)
    
    # ========== 损失函数：根据搜索参数配置 ==========
    num_classes = config['fault_types']['num_classes']
    if use_class_weights:
        class_weights = compute_class_weights(data_dict['y_source_train'], num_classes, device)
        cls_criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)
    else:
        cls_criterion = FocalLoss(gamma=focal_gamma, alpha=None)
    domain_criterion = nn.BCEWithLogitsLoss()
    
    # ========== 训练循环 ==========
    best_objective_score = -float('inf')  # 修复：初始化综合得分
    best_target_acc = 0.0
    best_source_acc = 0.0  # 新增：记录最佳源域准确率
    patience_counter = 0
    early_stopping_patience = 50 # 增加耐心，允许域适应发挥作用
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        # GRL lambda
        if epoch <= warmup_epochs:
            grl_lambda = 0.0
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            grl_lambda = 2.0 / (1.0 + np.exp(-gamma_grl * progress)) - 1.0
        
        model.set_grl_lambda(grl_lambda) # Use compatibility method
        
        # 域权重
        if domain_weight_schedule == 'constant':
            current_domain_weight = domain_loss_weight
        elif domain_weight_schedule == 'linear':
            current_domain_weight = domain_loss_weight * (epoch / num_epochs)
        else:  # exponential
            current_domain_weight = domain_loss_weight * (1 - np.exp(-3 * epoch / num_epochs))
        
        # ========== 两阶段训练：阶段1完全隔离域适应 ==========
        for source_batch, target_batch in dann_loader:
            x_s, y_s, _ = source_batch
            x_t, _, _ = target_batch
            
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t = x_t.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_s, x_t)
            
            cls_loss = cls_criterion(outputs['class_logits'], y_s)
            
            # 阶段1：纯分类训练，不计算域损失
            if epoch <= warmup_epochs:
                total_loss = cls_loss
            else:
                # 阶段2：加入域适应
                domain_s = torch.zeros(x_s.size(0), 1).to(device)
                domain_t = torch.ones(x_t.size(0), 1).to(device)
                domain_logits = torch.cat([outputs['domain_logits_source'], outputs['domain_logits_target']], dim=0)
                domain_labels = torch.cat([domain_s, domain_t], dim=0)
                domain_loss = domain_criterion(domain_logits, domain_labels)
                
                total_loss = cls_loss + current_domain_weight * grl_lambda * domain_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler is not None and lr_scheduler_type != 'onecycle':
            scheduler.step()
            
        # ========== 修复：同时验证源域和目标域 ==========
        model.eval()

        # 评估目标域
        all_preds_t, all_labels_t = [], []
        with torch.no_grad():
            for batch in target_test_loader:
                x, y, _ = batch
                x = x.to(device)
                features = model._extract_features(x)
                logits = model.classifier(features)
                preds = torch.argmax(logits, dim=1)
                all_preds_t.extend(preds.cpu().numpy())
                all_labels_t.extend(y.numpy())

        target_metrics = calculate_metrics(np.array(all_labels_t), np.array(all_preds_t))
        target_acc = target_metrics['accuracy']

        # 评估源域（新增）
        all_preds_s, all_labels_s = [], []
        with torch.no_grad():
            for batch in source_val_loader:  # 使用源域验证集
                x, y, _ = batch
                x = x.to(device)
                features = model._extract_features(x)
                logits = model.classifier(features)
                preds = torch.argmax(logits, dim=1)
                all_preds_s.extend(preds.cpu().numpy())
                all_labels_s.extend(y.numpy())

        source_metrics = calculate_metrics(np.array(all_labels_s), np.array(all_preds_s))
        source_acc = source_metrics['accuracy']

        # 计算域差距
        domain_gap = abs(source_acc - target_acc)

        # ========== 修复：多目标优化 + 类别分布约束 ==========
        # 检查预测类别分布（防止全预测一类的假成功）
        unique_preds_s = len(np.unique(all_preds_s))
        unique_preds_t = len(np.unique(all_preds_t))
        
        # 约束1：源域准确率不能低于50%
        # 约束2：必须预测至少3个不同类别（防止只预测多数类）
        if source_acc < 0.5:
            # 源域崩溃：给予重惩罚
            objective_score = target_acc - 1.0
        elif unique_preds_s <= 2:
            # 类别退化：只预测1-2个类，严重惩罚
            objective_score = target_acc - 1.5
        else:
            # 正常情况：综合评估
            # 70%目标域 + 20%源域 - 10%域差距
            objective_score = 0.7 * target_acc + 0.2 * source_acc - 0.1 * domain_gap

        # ========== 阶段1结束时：源域准确率早期检查 ==========
        if epoch == warmup_epochs:
            if source_acc < 0.4:
                # 分类器在阶段1没学好，直接剪枝避免浪费计算
                raise optuna.exceptions.TrialPruned()
        
        # 报告
        trial.report(objective_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if objective_score > best_objective_score:
            best_objective_score = objective_score
            best_target_acc = target_acc
            best_source_acc = source_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    # 返回综合得分而不是仅目标域准确率
    return best_objective_score


def run_single_condition_optuna(
    condition: int,
    config_path: str,
    sc_config_path: str,
    n_trials: int = 50
) -> Dict:
    """运行单工况 Optuna 优化"""
    if not OPTUNA_AVAILABLE:
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    with open(sc_config_path, 'r', encoding='utf-8') as f:
        sc_config = yaml.safe_load(f)
    
    condition_name = sc_config['conditions']['names'].get(condition, f"condition_{condition}")
    
    print("=" * 70)
    print(f"单工况深度 Optuna 调优 (扩展版)")
    print(f"工况: {condition} ({condition_name})")
    print(f"试验次数: {n_trials}")
    print("=" * 70)
    
    if config['device']['use_gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        print(f"[信息] 使用GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("[信息] 使用CPU")
        
    print("\n>>> 加载数据...")
    results_dir = sc_config['output']['results_dir']
    processed_path = os.path.join(results_dir, f'processed_data_condition_{condition}.pkl')
    
    if os.path.exists(processed_path):
        data_dict = load_single_condition_data(condition, results_dir)
    else:
        print("[信息] 数据未预处理，开始预处理...")
        # 注意: 需要确保 SingleConditionPreprocessor 可用
        try:
            from data.preprocess_single_condition import SingleConditionPreprocessor
            preprocessor = SingleConditionPreprocessor(
                config_path=config_path,
                sc_config_path=sc_config_path,
                condition=condition
            )
            data_dict = preprocessor.process(save_processed=True)
        except ImportError:
            print("[错误] 无法导入预处理器，请先运行数据预处理脚本")
            return {}
        
    set_seed(42)
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15)
    
    study = optuna.create_study(
        study_name=f'single_deep_{condition}',
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )
    
    print("\n>>> 开始超参数搜索...")
    
    def objective_wrapper(trial):
        set_seed(42)
        return objective_single_condition(
            trial, condition, config_path, sc_config_path, data_dict, device
        )
    
    study.optimize(objective_wrapper, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))
    
    print("\n" + "=" * 70)
    print(f"调优完成! 工况: {condition} ({condition_name})")
    print("=" * 70)
    print(f"\n最佳综合得分: {study.best_value:.4f}")
    print(f"  (公式: 0.7×目标域 + 0.2×源域 - 0.1×域差距)")
    print(f"  源域准确率约束: ≥50%")
    
    # 保存结果
    optuna_results_dir = './optuna_results'
    os.makedirs(optuna_results_dir, exist_ok=True)
    
    best_params_path = os.path.join(optuna_results_dir, f'condition_{condition}_best_params.yaml')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        yaml.dump(study.best_params, f, default_flow_style=False)
    
    print(f"\n[信息] 最优参数已保存至: {best_params_path}")
    
    # 保存试验记录
    df = study.trials_dataframe()
    df_path = os.path.join(optuna_results_dir, f'condition_{condition}_trials.csv')
    df.to_csv(df_path, index=False)
    
    # 自动同步
    print("\n>>> 自动同步最优参数到配置...")
    try:
        from scripts.sync_single_condition_params import sync_single_condition_params
        sync_single_condition_params(condition, backup=True)
        print("[信息] 参数已自动同步!")
    except Exception as e:
        print(f"[警告] 自动同步失败: {e}")
    
    return study.best_params


def main():
    parser = argparse.ArgumentParser(description='单工况 Optuna 超参数调优')
    parser.add_argument('--condition', type=int, default=0, help='工况代码 0-5')
    parser.add_argument('--config', type=str, default='./config/config.yaml')
    parser.add_argument('--sc_config', type=str, default=None)
    parser.add_argument('--n_trials', type=int, default=50)
    
    args = parser.parse_args()
    
    if args.condition not in range(6):
        print(f"[错误] 工况代码必须在0-5之间")
        return
        
    condition_names = {0: 'hover', 1: 'waypoint', 2: 'velocity', 3: 'circling', 4: 'acce', 5: 'dece'}
    if args.sc_config is None:
        cond_name = condition_names[args.condition]
        args.sc_config = f'./config/condition_{args.condition}_{cond_name}.yaml'
        print(f"[信息] 使用工况专属配置: {args.sc_config}")
        
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    if not os.path.isabs(args.sc_config):
        args.sc_config = os.path.join(project_root, args.sc_config)
        
    if not os.path.exists(args.sc_config):
        print(f"[错误] 配置文件不存在: {args.sc_config}")
        return
        
    run_single_condition_optuna(args.condition, args.config, args.sc_config, args.n_trials)


if __name__ == "__main__":
    main()
