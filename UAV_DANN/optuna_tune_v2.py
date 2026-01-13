# -*- coding: utf-8 -*-
"""
==============================================================================
Optuna深度超参数优化脚本 (扩展版)
==============================================================================
功能：全面搜索DANN模型的最优超参数
- 网络深度 (CNN层数, LSTM层数)
- 学习率策略 (warmup, cosine annealing, step decay)
- 域适应权重动态调整

使用方式：
---------
python optuna_tune_v2.py --config ./config/config.yaml --n_trials 100

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
import yaml
from typing import Dict, Tuple, Optional
from tqdm import tqdm
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

from data.dataloader import get_dataloaders
from data.preprocess import DataPreprocessor, load_processed_data
from utils.metrics import calculate_metrics


def set_seed(seed: int) -> None:
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class DANNDeep(nn.Module):
    """
    深度可调的DANN模型
    
    支持动态调整：
    - CNN层数 (1-4层)
    - LSTM层数 (1-3层)
    - 分类器深度 (1-3层)
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        num_classes: int,
        # CNN配置
        cnn_layers: int = 2,
        cnn_channels: list = [64, 128],
        cnn_kernel_size: int = 5,
        # LSTM配置
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        lstm_bidirectional: bool = False,
        # 分类器配置
        classifier_layers: int = 2,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.5,
        # 域判别器配置
        discriminator_hidden: int = 64,
        discriminator_layers: int = 2,
        # 使用BatchNorm
        use_batchnorm: bool = True
    ):
        super(DANNDeep, self).__init__()
        
        self.lstm_bidirectional = lstm_bidirectional
        lstm_output_size = lstm_hidden * (2 if lstm_bidirectional else 1)
        
        # ========== 动态CNN层 ==========
        cnn_modules = []
        in_channels = n_features
        current_seq_len = seq_len
        
        for i, out_channels in enumerate(cnn_channels[:cnn_layers]):
            cnn_modules.append(nn.Conv1d(in_channels, out_channels, 
                                         kernel_size=cnn_kernel_size, 
                                         padding=cnn_kernel_size // 2))
            if use_batchnorm:
                cnn_modules.append(nn.BatchNorm1d(out_channels))
            cnn_modules.append(nn.ReLU(inplace=True))
            cnn_modules.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels
            current_seq_len = current_seq_len // 2
        
        self.cnn = nn.Sequential(*cnn_modules)
        self.cnn_output_channels = in_channels
        self.cnn_output_len = current_seq_len
        
        # ========== LSTM ==========
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=lstm_bidirectional
        )
        
        self.feature_dim = lstm_output_size
        
        # ========== 动态分类器 ==========
        classifier_modules = []
        in_dim = lstm_output_size
        for i in range(classifier_layers - 1):
            classifier_modules.append(nn.Linear(in_dim, classifier_hidden))
            if use_batchnorm:
                classifier_modules.append(nn.BatchNorm1d(classifier_hidden))
            classifier_modules.append(nn.ReLU(inplace=True))
            classifier_modules.append(nn.Dropout(p=classifier_dropout))
            in_dim = classifier_hidden
        classifier_modules.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_modules)
        
        # ========== 动态域判别器 ==========
        discriminator_modules = []
        in_dim = lstm_output_size
        for i in range(discriminator_layers - 1):
            discriminator_modules.append(nn.Linear(in_dim, discriminator_hidden))
            if use_batchnorm:
                discriminator_modules.append(nn.BatchNorm1d(discriminator_hidden))
            discriminator_modules.append(nn.ReLU(inplace=True))
            discriminator_modules.append(nn.Dropout(p=classifier_dropout))
            in_dim = discriminator_hidden
        discriminator_modules.append(nn.Linear(in_dim, 1))
        self.discriminator = nn.Sequential(*discriminator_modules)
        
        self.grl_lambda = 0.0
    
    def set_grl_lambda(self, lambda_val: float):
        self.grl_lambda = lambda_val
    
    def _extract_features(self, x):
        # (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # (B, C, T') -> (B, T', C)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        
        if self.lstm_bidirectional:
            # 合并双向隐藏状态
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]
        return h_n
    
    def forward(self, x_source, x_target=None):
        outputs = {}
        features_s = self._extract_features(x_source)
        outputs['features_source'] = features_s
        outputs['class_logits'] = self.classifier(features_s)
        
        reversed_s = GradientReversal.apply(features_s, self.grl_lambda)
        outputs['domain_logits_source'] = self.discriminator(reversed_s)
        
        if x_target is not None:
            features_t = self._extract_features(x_target)
            outputs['features_target'] = features_t
            reversed_t = GradientReversal.apply(features_t, self.grl_lambda)
            outputs['domain_logits_target'] = self.discriminator(reversed_t)
        
        return outputs


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


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


def objective_deep(trial: Trial, config: dict, data_dict: dict, device: torch.device) -> float:
    """
    深度优化目标函数
    """
    # ========== 网络深度 ==========
    cnn_layers = trial.suggest_int('cnn_layers', 1, 4)
    cnn_base_channels = trial.suggest_categorical('cnn_base_channels', [32, 48, 64, 96])
    cnn_channels = [cnn_base_channels * (2 ** i) for i in range(cnn_layers)]
    
    lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    lstm_hidden = trial.suggest_categorical('lstm_hidden', [64, 96, 128, 192, 256])
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5)
    lstm_bidirectional = trial.suggest_categorical('lstm_bidirectional', [True, False])
    
    classifier_layers = trial.suggest_int('classifier_layers', 1, 3)
    classifier_hidden = trial.suggest_categorical('classifier_hidden', [32, 64, 96, 128])
    classifier_dropout = trial.suggest_float('classifier_dropout', 0.2, 0.6)
    
    discriminator_layers = trial.suggest_int('discriminator_layers', 1, 3)
    discriminator_hidden = trial.suggest_categorical('discriminator_hidden', [32, 64, 96])
    
    # ========== 学习率策略 ==========
    lr_scheduler_type = trial.suggest_categorical('lr_scheduler', 
                                                   ['step', 'cosine', 'warmup_cosine', 'onecycle'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # 固定batch_size=512 以加速调优 (RTX 5070Ti 16GB显存, 模型较小)
    batch_size = 512
    
    # warmup设置
    warmup_epochs = trial.suggest_int('warmup_epochs', 0, 15)
    
    # ========== 域适应权重 ==========
    gamma_grl = trial.suggest_float('gamma_grl', 5.0, 20.0)
    domain_loss_weight = trial.suggest_float('domain_loss_weight', 0.1, 2.0)
    # 动态域权重策略
    domain_weight_schedule = trial.suggest_categorical('domain_weight_schedule', 
                                                        ['constant', 'linear', 'exponential'])
    
    # ========== 创建数据加载器 ==========
    config_copy = config.copy()
    config_copy['training'] = config['training'].copy()
    config_copy['training']['batch_size'] = batch_size
    
    loaders = get_dataloaders(config=config_copy, data_dict=data_dict)
    steps_per_epoch = len(loaders['dann_train'])
    
    # ========== 创建模型 ==========
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
        discriminator_layers=discriminator_layers
    ).to(device)
    
    # ========== 优化器和调度器 ==========
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, lr_scheduler_type, num_epochs=50, 
                              steps_per_epoch=steps_per_epoch,
                              warmup_epochs=warmup_epochs, max_lr=learning_rate)
    
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    
    # ========== 训练 ==========
    num_epochs = 50
    best_target_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        # GRL系数计算
        if epoch <= warmup_epochs:
            grl_lambda = 0.0
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            grl_lambda = 2.0 / (1.0 + np.exp(-gamma_grl * progress)) - 1.0
        
        model.set_grl_lambda(grl_lambda)
        
        # 动态域权重
        if domain_weight_schedule == 'constant':
            current_domain_weight = domain_loss_weight
        elif domain_weight_schedule == 'linear':
            current_domain_weight = domain_loss_weight * (epoch / num_epochs)
        else:  # exponential
            current_domain_weight = domain_loss_weight * (1 - np.exp(-3 * epoch / num_epochs))
        
        for source_batch, target_batch in loaders['dann_train']:
            x_s, y_s, _ = source_batch
            x_t, _, _ = target_batch
            
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t = x_t.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_s, x_t)
            
            cls_loss = cls_criterion(outputs['class_logits'], y_s)
            
            domain_s = torch.zeros(x_s.size(0), 1).to(device)
            domain_t = torch.ones(x_t.size(0), 1).to(device)
            domain_logits = torch.cat([outputs['domain_logits_source'], 
                                       outputs['domain_logits_target']], dim=0)
            domain_labels = torch.cat([domain_s, domain_t], dim=0)
            domain_loss = domain_criterion(domain_logits, domain_labels)
            
            total_loss = cls_loss + current_domain_weight * grl_lambda * domain_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler is not None and lr_scheduler_type != 'onecycle':
            scheduler.step()
        
        # 验证
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in loaders['target_test']:
                x, y, _ = batch
                x = x.to(device)
                features = model._extract_features(x)
                logits = model.classifier(features)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
        
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        target_acc = metrics['accuracy']
        
        if target_acc > best_target_acc:
            best_target_acc = target_acc
        
        trial.report(target_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_target_acc


def run_deep_optimization(config_path: str, n_trials: int = 100, study_name: str = 'dann_deep_hpo'):
    """运行深度优化"""
    if not OPTUNA_AVAILABLE:
        print("[错误] Optuna未安装")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("UAV-DANN 深度超参数优化 (扩展版)")
    print("=" * 70)
    print(f"搜索次数: {n_trials}")
    print("优化目标: 网络深度 + 学习率策略 + 域适应权重")
    
    if config['device']['use_gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        print(f"设备: GPU ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        print("设备: CPU")
    
    print("\n>>> 加载数据...")
    processed_path = os.path.join(config['data']['processed_dir'], 'processed_data.pkl')
    if os.path.exists(processed_path):
        data_dict = load_processed_data(config['data']['processed_dir'])
    else:
        preprocessor = DataPreprocessor(config_path=config_path)
        data_dict = preprocessor.process(save_processed=True)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15)
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    print("\n>>> 开始深度超参数搜索...")
    
    def objective_wrapper(trial):
        set_seed(42)
        return objective_deep(trial, config, data_dict, device)
    
    study.optimize(objective_wrapper, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))
    
    print("\n" + "=" * 70)
    print("深度优化完成!")
    print("=" * 70)
    print(f"\n最佳目标域准确率: {study.best_value:.4f}")
    print(f"\n最优超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    results_dir = './optuna_results'
    os.makedirs(results_dir, exist_ok=True)
    
    best_params_path = os.path.join(results_dir, f'{study_name}_best_params.yaml')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        yaml.dump(study.best_params, f, default_flow_style=False)
    print(f"\n最优参数已保存至: {best_params_path}")
    
    df = study.trials_dataframe()
    df_path = os.path.join(results_dir, f'{study_name}_trials.csv')
    df.to_csv(df_path, index=False)
    print(f"完整试验记录已保存至: {df_path}")
    
    return study.best_params


def main():
    parser = argparse.ArgumentParser(description='UAV-DANN 深度超参数优化')
    parser.add_argument('--config', type=str, default='./config/config.yaml')
    parser.add_argument('--n_trials', type=int, default=100, help='搜索次数')
    parser.add_argument('--study_name', type=str, default='dann_deep_hpo')
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    
    run_deep_optimization(args.config, args.n_trials, args.study_name)


if __name__ == "__main__":
    main()
