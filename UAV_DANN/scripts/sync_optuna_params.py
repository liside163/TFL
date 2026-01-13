# -*- coding: utf-8 -*-
"""
==============================================================================
Optuna 最优参数同步脚本
==============================================================================
功能：将 Optuna 超参数优化的最优结果同步到主配置文件 config.yaml

使用方式：
---------
python scripts/sync_optuna_params.py --optuna_result ./optuna_results/dann_deep_hpo_best_params.yaml --config ./config/config.yaml

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import sys
import argparse
import yaml
from typing import Dict, Any
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_yaml(path: str) -> Dict:
    """加载 YAML 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: str) -> None:
    """保存 YAML 文件，保留注释格式"""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def sync_optuna_params(optuna_result_path: str, config_path: str, backup: bool = True) -> None:
    """
    将 Optuna 最优参数同步到 config.yaml
    
    Args:
        optuna_result_path: Optuna 结果文件路径
        config_path: 主配置文件路径
        backup: 是否备份原配置文件
        
    参数映射：
    ---------
    Optuna 参数                    → config.yaml 路径
    cnn_layers                     → model.feature_extractor.cnn.num_layers
    cnn_base_channels              → model.feature_extractor.cnn.base_channels
    lstm_hidden                    → model.feature_extractor.lstm.hidden_size
    lstm_layers                    → model.feature_extractor.lstm.num_layers
    lstm_dropout                   → model.feature_extractor.lstm.dropout
    lstm_bidirectional             → model.feature_extractor.lstm.bidirectional
    classifier_hidden              → model.classifier.hidden_dim
    classifier_layers              → model.classifier.num_layers
    classifier_dropout             → model.classifier.dropout
    discriminator_hidden           → model.domain_discriminator.hidden_dim
    discriminator_layers           → model.domain_discriminator.num_layers
    learning_rate                  → training.optimizer.learning_rate
    weight_decay                   → training.optimizer.weight_decay
    lr_scheduler                   → training.scheduler.name
    warmup_epochs                  → training.scheduler.warmup_epochs
    gamma_grl                      → training.domain_adaptation.gamma_grl
    domain_loss_weight             → training.domain_adaptation.domain_loss_weight
    domain_weight_schedule         → training.domain_adaptation.domain_weight_schedule
    """
    print("=" * 60)
    print("Optuna 参数同步工具")
    print("=" * 60)
    
    # 加载文件
    print(f"\n[1/4] 加载 Optuna 结果: {optuna_result_path}")
    optuna_params = load_yaml(optuna_result_path)
    print(f"  找到 {len(optuna_params)} 个参数")
    
    print(f"\n[2/4] 加载配置文件: {config_path}")
    config = load_yaml(config_path)
    
    # 备份原配置
    if backup:
        backup_path = config_path.replace('.yaml', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
        save_yaml(config, backup_path)
        print(f"  已备份至: {backup_path}")
    
    # 同步参数
    print("\n[3/4] 同步参数...")
    sync_count = 0
    
    # CNN 参数
    if 'cnn_layers' in optuna_params:
        config['model']['feature_extractor']['cnn']['num_layers'] = optuna_params['cnn_layers']
        print(f"  ✓ cnn.num_layers = {optuna_params['cnn_layers']}")
        sync_count += 1
    
    if 'cnn_base_channels' in optuna_params:
        config['model']['feature_extractor']['cnn']['base_channels'] = optuna_params['cnn_base_channels']
        config['model']['feature_extractor']['cnn']['conv1_out_channels'] = optuna_params['cnn_base_channels']
        print(f"  ✓ cnn.base_channels = {optuna_params['cnn_base_channels']}")
        sync_count += 1
    
    # LSTM 参数
    if 'lstm_hidden' in optuna_params:
        config['model']['feature_extractor']['lstm']['hidden_size'] = optuna_params['lstm_hidden']
        print(f"  ✓ lstm.hidden_size = {optuna_params['lstm_hidden']}")
        sync_count += 1
    
    if 'lstm_layers' in optuna_params:
        config['model']['feature_extractor']['lstm']['num_layers'] = optuna_params['lstm_layers']
        print(f"  ✓ lstm.num_layers = {optuna_params['lstm_layers']}")
        sync_count += 1
    
    if 'lstm_dropout' in optuna_params:
        config['model']['feature_extractor']['lstm']['dropout'] = round(optuna_params['lstm_dropout'], 4)
        print(f"  ✓ lstm.dropout = {optuna_params['lstm_dropout']:.4f}")
        sync_count += 1
    
    if 'lstm_bidirectional' in optuna_params:
        config['model']['feature_extractor']['lstm']['bidirectional'] = optuna_params['lstm_bidirectional']
        print(f"  ✓ lstm.bidirectional = {optuna_params['lstm_bidirectional']}")
        sync_count += 1
    
    # 分类器参数
    if 'classifier_hidden' in optuna_params:
        config['model']['classifier']['hidden_dim'] = optuna_params['classifier_hidden']
        print(f"  ✓ classifier.hidden_dim = {optuna_params['classifier_hidden']}")
        sync_count += 1
    
    if 'classifier_layers' in optuna_params:
        config['model']['classifier']['num_layers'] = optuna_params['classifier_layers']
        print(f"  ✓ classifier.num_layers = {optuna_params['classifier_layers']}")
        sync_count += 1
    
    if 'classifier_dropout' in optuna_params:
        config['model']['classifier']['dropout'] = round(optuna_params['classifier_dropout'], 4)
        print(f"  ✓ classifier.dropout = {optuna_params['classifier_dropout']:.4f}")
        sync_count += 1
    
    # 域判别器参数
    if 'discriminator_hidden' in optuna_params:
        config['model']['domain_discriminator']['hidden_dim'] = optuna_params['discriminator_hidden']
        print(f"  ✓ domain_discriminator.hidden_dim = {optuna_params['discriminator_hidden']}")
        sync_count += 1
    
    if 'discriminator_layers' in optuna_params:
        config['model']['domain_discriminator']['num_layers'] = optuna_params['discriminator_layers']
        print(f"  ✓ domain_discriminator.num_layers = {optuna_params['discriminator_layers']}")
        sync_count += 1
    
    # 优化器参数
    if 'learning_rate' in optuna_params:
        config['training']['optimizer']['learning_rate'] = round(optuna_params['learning_rate'], 6)
        print(f"  ✓ optimizer.learning_rate = {optuna_params['learning_rate']:.6f}")
        sync_count += 1
    
    if 'weight_decay' in optuna_params:
        config['training']['optimizer']['weight_decay'] = round(optuna_params['weight_decay'], 6)
        print(f"  ✓ optimizer.weight_decay = {optuna_params['weight_decay']:.6f}")
        sync_count += 1
    
    # 学习率调度器参数
    if 'lr_scheduler' in optuna_params:
        config['training']['scheduler']['name'] = optuna_params['lr_scheduler']
        print(f"  ✓ scheduler.name = {optuna_params['lr_scheduler']}")
        sync_count += 1
    
    if 'warmup_epochs' in optuna_params:
        config['training']['scheduler']['warmup_epochs'] = optuna_params['warmup_epochs']
        # 同时更新域适应的 warmup
        config['training']['domain_adaptation']['warmup_epochs'] = optuna_params['warmup_epochs']
        print(f"  ✓ scheduler.warmup_epochs = {optuna_params['warmup_epochs']}")
        sync_count += 1
    
    # 域适应参数
    if 'gamma_grl' in optuna_params:
        config['training']['domain_adaptation']['gamma_grl'] = round(optuna_params['gamma_grl'], 4)
        print(f"  ✓ domain_adaptation.gamma_grl = {optuna_params['gamma_grl']:.4f}")
        sync_count += 1
    
    if 'domain_loss_weight' in optuna_params:
        config['training']['domain_adaptation']['domain_loss_weight'] = round(optuna_params['domain_loss_weight'], 4)
        print(f"  ✓ domain_adaptation.domain_loss_weight = {optuna_params['domain_loss_weight']:.4f}")
        sync_count += 1
    
    if 'domain_weight_schedule' in optuna_params:
        config['training']['domain_adaptation']['domain_weight_schedule'] = optuna_params['domain_weight_schedule']
        print(f"  ✓ domain_adaptation.domain_weight_schedule = {optuna_params['domain_weight_schedule']}")
        sync_count += 1
    
    # 保存更新后的配置
    print(f"\n[4/4] 保存更新后的配置...")
    save_yaml(config, config_path)
    
    print("\n" + "=" * 60)
    print(f"同步完成! 共更新 {sync_count} 个参数")
    print(f"配置文件: {config_path}")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Optuna 参数同步工具')
    
    parser.add_argument('--optuna_result', type=str, 
                        default='./optuna_results/dann_deep_hpo_best_params.yaml',
                        help='Optuna 最优参数文件路径')
    parser.add_argument('--config', type=str, 
                        default='./config/config.yaml',
                        help='主配置文件路径')
    parser.add_argument('--no_backup', action='store_true',
                        help='不备份原配置文件')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    if not os.path.isabs(args.optuna_result):
        args.optuna_result = os.path.join(project_root, args.optuna_result)
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    
    # 检查文件存在
    if not os.path.exists(args.optuna_result):
        print(f"[错误] Optuna 结果文件不存在: {args.optuna_result}")
        return
    
    if not os.path.exists(args.config):
        print(f"[错误] 配置文件不存在: {args.config}")
        return
    
    # 执行同步
    sync_optuna_params(
        optuna_result_path=args.optuna_result,
        config_path=args.config,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
