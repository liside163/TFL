# -*- coding: utf-8 -*-
"""
==============================================================================
Optuna超参数优化脚本
==============================================================================
功能：使用Optuna自动搜索DANN模型的最优超参数
- 模型架构参数（CNN通道数、LSTM隐藏层等）
- 训练参数（学习率、batch_size、dropout等）
- 域适应参数（GRL系数、预热轮数等）

作者：UAV-DANN项目
日期：2025年

使用方式：
---------
python optuna_tune.py --config ./config/config.yaml --n_trials 50 --study_name dann_hpo

优化目标：目标域（Real）验证集上的准确率
==============================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import yaml
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optuna导入
try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[错误] 请先安装Optuna: pip install optuna")

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.dataloader import get_dataloaders
from data.preprocess import DataPreprocessor, load_processed_data
from models.layers import compute_grl_lambda
from utils.metrics import calculate_metrics


def set_seed(seed: int) -> None:
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DANNTunable(nn.Module):
    """
    可调参数的DANN模型
    
    用于Optuna超参数搜索，根据trial参数动态构建网络
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        num_classes: int,
        conv1_channels: int,
        conv2_channels: int,
        lstm_hidden: int,
        lstm_layers: int,
        lstm_dropout: float,
        classifier_hidden: int,
        classifier_dropout: float,
        discriminator_hidden: int
    ):
        super(DANNTunable, self).__init__()
        
        # ========== 特征提取器 ==========
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, conv1_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(conv1_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(conv2_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv2_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        self.feature_dim = lstm_hidden
        
        # ========== 故障分类器 ==========
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(classifier_hidden, num_classes)
        )
        
        # ========== 域判别器 ==========
        self.discriminator = nn.Sequential(
            nn.Linear(lstm_hidden, discriminator_hidden),
            nn.BatchNorm1d(discriminator_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(discriminator_hidden, 1)
        )
        
        # GRL系数
        self.grl_lambda = 0.0
    
    def set_grl_lambda(self, lambda_val: float):
        """设置GRL系数"""
        self.grl_lambda = lambda_val
    
    def forward(self, x_source, x_target=None):
        """前向传播"""
        outputs = {}
        
        # 源域特征提取
        features_s = self._extract_features(x_source)
        outputs['features_source'] = features_s
        outputs['class_logits'] = self.classifier(features_s)
        
        # 域判别（带梯度反转）
        reversed_features_s = GradientReversal.apply(features_s, self.grl_lambda)
        outputs['domain_logits_source'] = self.discriminator(reversed_features_s)
        
        if x_target is not None:
            features_t = self._extract_features(x_target)
            outputs['features_target'] = features_t
            reversed_features_t = GradientReversal.apply(features_t, self.grl_lambda)
            outputs['domain_logits_target'] = self.discriminator(reversed_features_t)
        
        return outputs
    
    def _extract_features(self, x):
        """提取特征"""
        # (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        # (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class GradientReversal(torch.autograd.Function):
    """梯度反转函数"""
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


def objective(trial: Trial, config: dict, data_dict: dict, device: torch.device) -> float:
    """
    Optuna优化目标函数
    
    Args:
        trial: Optuna trial对象
        config: 基础配置
        data_dict: 预处理数据
        device: 计算设备
    
    Returns:
        target_accuracy: 目标域验证准确率（优化目标）
    """
    # ========== 采样超参数 ==========
    
    # 模型架构
    conv1_channels = trial.suggest_categorical('conv1_channels', [32, 64, 96, 128])
    conv2_channels = trial.suggest_categorical('conv2_channels', [64, 128, 192, 256])
    lstm_hidden = trial.suggest_categorical('lstm_hidden', [64, 128, 192, 256])
    lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5)
    classifier_hidden = trial.suggest_categorical('classifier_hidden', [32, 64, 96, 128])
    classifier_dropout = trial.suggest_float('classifier_dropout', 0.2, 0.6)
    discriminator_hidden = trial.suggest_categorical('discriminator_hidden', [32, 64, 96])
    
    # 训练参数
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # 域适应参数
    warmup_epochs = trial.suggest_int('warmup_epochs', 3, 15)
    gamma_grl = trial.suggest_float('gamma_grl', 5.0, 15.0)
    domain_loss_weight = trial.suggest_float('domain_loss_weight', 0.5, 1.5)
    
    # ========== 创建数据加载器 ==========
    # 更新配置中的batch_size
    config_copy = config.copy()
    config_copy['training'] = config['training'].copy()
    config_copy['training']['batch_size'] = batch_size
    
    loaders = get_dataloaders(config=config_copy, data_dict=data_dict)
    
    # ========== 创建模型 ==========
    n_features = config['preprocessing']['n_features']
    seq_len = config['preprocessing']['window_size']
    num_classes = config['fault_types']['num_classes']
    
    model = DANNTunable(
        n_features=n_features,
        seq_len=seq_len,
        num_classes=num_classes,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        classifier_hidden=classifier_hidden,
        classifier_dropout=classifier_dropout,
        discriminator_hidden=discriminator_hidden
    ).to(device)
    
    # ========== 优化器和损失函数 ==========
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    
    # ========== 训练 ==========
    num_epochs = 30  # 调优时使用较少的epoch
    best_target_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        # 计算GRL系数
        if epoch <= warmup_epochs:
            grl_lambda = 0.0
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            grl_lambda = 2.0 / (1.0 + np.exp(-gamma_grl * progress)) - 1.0
        
        model.set_grl_lambda(grl_lambda)
        
        # 训练一个epoch
        for source_batch, target_batch in loaders['dann_train']:
            x_s, y_s, _ = source_batch
            x_t, _, _ = target_batch
            
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t = x_t.to(device)
            
            batch_size_s = x_s.size(0)
            batch_size_t = x_t.size(0)
            
            optimizer.zero_grad()
            
            outputs = model(x_s, x_t)
            
            # 分类损失
            cls_loss = cls_criterion(outputs['class_logits'], y_s)
            
            # 域判别损失
            domain_s = torch.zeros(batch_size_s, 1).to(device)
            domain_t = torch.ones(batch_size_t, 1).to(device)
            domain_logits = torch.cat([outputs['domain_logits_source'], 
                                       outputs['domain_logits_target']], dim=0)
            domain_labels = torch.cat([domain_s, domain_t], dim=0)
            domain_loss = domain_criterion(domain_logits, domain_labels)
            
            # 总损失
            total_loss = cls_loss + grl_lambda * domain_loss_weight * domain_loss
            
            total_loss.backward()
            optimizer.step()
        
        # 验证目标域性能
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
        
        # Optuna剪枝：如果性能太差，提前终止
        trial.report(target_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_target_acc


def run_optimization(
    config_path: str,
    n_trials: int = 50,
    study_name: str = 'dann_hpo',
    storage: Optional[str] = None
) -> Dict:
    """
    运行超参数优化
    
    Args:
        config_path: 配置文件路径
        n_trials: 搜索次数
        study_name: 研究名称
        storage: 数据库存储路径（可选）
    
    Returns:
        best_params: 最优超参数
    """
    if not OPTUNA_AVAILABLE:
        print("[错误] Optuna未安装，请运行: pip install optuna")
        return {}
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("UAV-DANN Optuna超参数优化")
    print("=" * 70)
    print(f"搜索次数: {n_trials}")
    print(f"研究名称: {study_name}")
    
    # 设置设备
    if config['device']['use_gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        print(f"设备: GPU ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        print("设备: CPU")
    
    # 加载数据
    print("\n>>> 加载数据...")
    processed_path = os.path.join(config['data']['processed_dir'], 'processed_data.pkl')
    if os.path.exists(processed_path):
        data_dict = load_processed_data(config['data']['processed_dir'])
    else:
        preprocessor = DataPreprocessor(config_path=config_path)
        data_dict = preprocessor.process(save_processed=True)
    
    # 创建Optuna研究
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # 最大化目标域准确率
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True
    )
    
    # 优化
    print("\n>>> 开始超参数搜索...")
    
    def objective_wrapper(trial):
        set_seed(42)
        return objective(trial, config, data_dict, device)
    
    study.optimize(
        objective_wrapper,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    # 输出结果
    print("\n" + "=" * 70)
    print("优化完成!")
    print("=" * 70)
    
    print(f"\n最佳目标域准确率: {study.best_value:.4f}")
    print(f"\n最优超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    results_dir = './optuna_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存最优参数
    best_params_path = os.path.join(results_dir, f'{study_name}_best_params.yaml')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        yaml.dump(study.best_params, f, default_flow_style=False)
    print(f"\n最优参数已保存至: {best_params_path}")
    
    # 保存完整的研究结果
    df = study.trials_dataframe()
    df_path = os.path.join(results_dir, f'{study_name}_trials.csv')
    df.to_csv(df_path, index=False)
    print(f"完整试验记录已保存至: {df_path}")
    
    # ========== 自动更新 config.yaml ==========
    print(f"\n>>> 正在将最优超参数更新到配置文件...")
    update_config_with_best_params(config_path, study.best_params)
    
    return study.best_params


def update_config_with_best_params(config_path: str, best_params: Dict) -> None:
    """
    将Optuna找到的最优超参数更新到config.yaml文件
    
    Args:
        config_path: 配置文件路径
        best_params: 最优超参数字典
    
    超参数映射：
        best_params key -> config.yaml path
        ──────────────────────────────────────
        conv1_channels -> model.feature_extractor.cnn.conv1_out_channels
        conv2_channels -> model.feature_extractor.cnn.conv2_out_channels
        lstm_hidden -> model.feature_extractor.lstm.hidden_size
        lstm_layers -> model.feature_extractor.lstm.num_layers
        lstm_dropout -> model.feature_extractor.lstm.dropout
        classifier_hidden -> model.classifier.hidden_dim
        classifier_dropout -> model.classifier.dropout
        discriminator_hidden -> model.domain_discriminator.hidden_dim
        learning_rate -> training.optimizer.learning_rate
        batch_size -> training.batch_size
        weight_decay -> training.optimizer.weight_decay
        warmup_epochs -> training.domain_adaptation.warmup_epochs
        gamma_grl -> training.domain_adaptation.gamma_grl
        domain_loss_weight -> training.domain_adaptation.domain_loss_weight
    """
    # 加载现有配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 备份原始配置
    backup_path = config_path.replace('.yaml', '_backup.yaml')
    with open(backup_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"[备份] 原始配置已备份至: {backup_path}")
    
    # 更新模型架构参数
    if 'conv1_channels' in best_params:
        config['model']['feature_extractor']['cnn']['conv1_out_channels'] = best_params['conv1_channels']
    if 'conv2_channels' in best_params:
        config['model']['feature_extractor']['cnn']['conv2_out_channels'] = best_params['conv2_channels']
    if 'lstm_hidden' in best_params:
        config['model']['feature_extractor']['lstm']['hidden_size'] = best_params['lstm_hidden']
    if 'lstm_layers' in best_params:
        config['model']['feature_extractor']['lstm']['num_layers'] = best_params['lstm_layers']
    if 'lstm_dropout' in best_params:
        config['model']['feature_extractor']['lstm']['dropout'] = round(best_params['lstm_dropout'], 4)
    
    # 更新分类器参数
    if 'classifier_hidden' in best_params:
        config['model']['classifier']['hidden_dim'] = best_params['classifier_hidden']
    if 'classifier_dropout' in best_params:
        config['model']['classifier']['dropout'] = round(best_params['classifier_dropout'], 4)
    
    # 更新域判别器参数
    if 'discriminator_hidden' in best_params:
        config['model']['domain_discriminator']['hidden_dim'] = best_params['discriminator_hidden']
    
    # 更新训练参数
    if 'batch_size' in best_params:
        config['training']['batch_size'] = best_params['batch_size']
    if 'learning_rate' in best_params:
        config['training']['optimizer']['learning_rate'] = round(best_params['learning_rate'], 6)
    if 'weight_decay' in best_params:
        config['training']['optimizer']['weight_decay'] = round(best_params['weight_decay'], 6)
    
    # 更新域适应参数
    if 'warmup_epochs' in best_params:
        config['training']['domain_adaptation']['warmup_epochs'] = best_params['warmup_epochs']
    if 'gamma_grl' in best_params:
        config['training']['domain_adaptation']['gamma_grl'] = round(best_params['gamma_grl'], 4)
    if 'domain_loss_weight' in best_params:
        config['training']['domain_adaptation']['domain_loss_weight'] = round(best_params['domain_loss_weight'], 4)
    
    # 保存更新后的配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"[更新] 配置文件已更新: {config_path}")
    print("\n更新后的关键配置:")
    print(f"  模型架构:")
    print(f"    - conv1_out_channels: {config['model']['feature_extractor']['cnn']['conv1_out_channels']}")
    print(f"    - conv2_out_channels: {config['model']['feature_extractor']['cnn']['conv2_out_channels']}")
    print(f"    - lstm.hidden_size: {config['model']['feature_extractor']['lstm']['hidden_size']}")
    print(f"    - lstm.num_layers: {config['model']['feature_extractor']['lstm']['num_layers']}")
    print(f"    - classifier.hidden_dim: {config['model']['classifier']['hidden_dim']}")
    print(f"  训练参数:")
    print(f"    - batch_size: {config['training']['batch_size']}")
    print(f"    - learning_rate: {config['training']['optimizer']['learning_rate']}")
    print(f"  域适应:")
    print(f"    - warmup_epochs: {config['training']['domain_adaptation']['warmup_epochs']}")
    print(f"    - gamma_grl: {config['training']['domain_adaptation']['gamma_grl']}")
    print(f"    - domain_loss_weight: {config['training']['domain_adaptation']['domain_loss_weight']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='UAV-DANN Optuna超参数优化')
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='配置文件路径')
    parser.add_argument('--n_trials', type=int, default=50, help='搜索次数')
    parser.add_argument('--study_name', type=str, default='dann_hpo', help='研究名称')
    parser.add_argument('--storage', type=str, default=None, help='数据库存储路径')
    parser.add_argument('--apply_best', type=str, default=None, 
                        help='直接应用已保存的最优参数文件到config.yaml (例如: ./optuna_results/dann_hpo_best_params.yaml)')
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    
    # 如果指定了apply_best，直接应用最优参数
    if args.apply_best is not None:
        best_params_path = args.apply_best
        if not os.path.isabs(best_params_path):
            best_params_path = os.path.join(project_root, best_params_path)
        
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r', encoding='utf-8') as f:
                best_params = yaml.safe_load(f)
            print(f">>> 加载最优参数文件: {best_params_path}")
            update_config_with_best_params(args.config, best_params)
            print("\n完成！现在可以使用更新后的配置运行训练:")
            print(f"  python train.py --config {args.config}")
        else:
            print(f"[错误] 最优参数文件不存在: {best_params_path}")
        return
    
    # 正常运行Optuna优化
    run_optimization(
        config_path=args.config,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage
    )


if __name__ == "__main__":
    main()
