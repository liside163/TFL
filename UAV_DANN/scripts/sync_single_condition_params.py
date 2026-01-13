# -*- coding: utf-8 -*-
"""
==============================================================================
单工况 Optuna 参数同步工具
==============================================================================
功能：将 Optuna 调优结果应用到单工况训练

使用方式：
---------
# 同步特定工况的参数
python scripts/sync_single_condition_params.py --condition 0

# 同步所有工况的参数
python scripts/sync_single_condition_params.py --all

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import sys
import argparse
import yaml
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def sync_single_condition_params(condition: int, backup: bool = True):
    """
    同步单工况 Optuna 最优参数到对应工况的独立配置文件
    
    Args:
        condition: 工况代码 (0-5)
        backup: 是否备份配置文件
    """
    # 工况名称映射
    condition_names = {0: 'hover', 1: 'waypoint', 2: 'velocity', 3: 'circling', 4: 'acce', 5: 'dece'}
    cond_name = condition_names.get(condition, f'condition_{condition}')
    
    # Optuna 结果路径
    optuna_result_path = os.path.join(
        project_root, 
        'optuna_results', 
        f'condition_{condition}_best_params.yaml'
    )
    
    # 对应工况的独立配置文件路径
    sc_config_path = os.path.join(
        project_root, 
        'config', 
        f'condition_{condition}_{cond_name}.yaml'
    )
    
    # 检查文件存在
    if not os.path.exists(optuna_result_path):
        print(f"[错误] Optuna 结果文件不存在: {optuna_result_path}")
        return False
    
    if not os.path.exists(sc_config_path):
        print(f"[错误] 配置文件不存在: {sc_config_path}")
        return False
    
    print(f"[单工况参数同步] 工况 {condition}")
    print(f"Optuna 结果: {optuna_result_path}")
    print(f"目标配置: {sc_config_path}")
    
    # 加载文件
    with open(optuna_result_path, 'r', encoding='utf-8') as f:
        optuna_params = yaml.safe_load(f)
    
    with open(sc_config_path, 'r', encoding='utf-8') as f:
        sc_config = yaml.safe_load(f)
    
    # 备份
    if backup:
        backup_path = sc_config_path.replace(
            '.yaml', 
            f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
        )
        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(sc_config, f, default_flow_style=False, allow_unicode=True)
        print(f"[信息] 已备份至: {backup_path}")
    
    # 同步参数到 training 部分 (新配置结构)
    print("\n[同步参数]")
    sync_count = 0
    
    if 'training' not in sc_config:
        sc_config['training'] = {}
    
    # Batch size
    if 'batch_size' in optuna_params:
        sc_config['training']['batch_size'] = optuna_params['batch_size']
        print(f"  ✓ batch_size = {optuna_params['batch_size']}")
        sync_count += 1
    
    # Early stopping patience
    if 'early_stopping_patience' in optuna_params:
        sc_config['training']['early_stopping_patience'] = optuna_params['early_stopping_patience']
        print(f"  ✓ early_stopping_patience = {optuna_params['early_stopping_patience']}")
        sync_count += 1
    
    # Warmup epochs
    if 'warmup_epochs' in optuna_params:
        if 'scheduler' not in sc_config['training']:
            sc_config['training']['scheduler'] = {}
        sc_config['training']['scheduler']['warmup_epochs'] = optuna_params['warmup_epochs']
        print(f"  ✓ warmup_epochs = {optuna_params['warmup_epochs']}")
        sync_count += 1
    
    # Learning rate
    if 'learning_rate' in optuna_params:
        if 'optimizer' not in sc_config['training']:
            sc_config['training']['optimizer'] = {}
        sc_config['training']['optimizer']['learning_rate'] = round(optuna_params['learning_rate'], 6)
        print(f"  ✓ learning_rate = {optuna_params['learning_rate']:.6f}")
        sync_count += 1
    
    # Weight decay
    if 'weight_decay' in optuna_params:
        if 'optimizer' not in sc_config['training']:
            sc_config['training']['optimizer'] = {}
        sc_config['training']['optimizer']['weight_decay'] = round(optuna_params['weight_decay'], 6)
        print(f"  ✓ weight_decay = {optuna_params['weight_decay']:.6f}")
        sync_count += 1
    
    # Domain loss weight
    if 'domain_loss_weight' in optuna_params:
        if 'domain_adaptation' not in sc_config['training']:
            sc_config['training']['domain_adaptation'] = {}
        sc_config['training']['domain_adaptation']['domain_loss_weight'] = round(optuna_params['domain_loss_weight'], 4)
        print(f"  ✓ domain_loss_weight = {optuna_params['domain_loss_weight']:.4f}")
        sync_count += 1
    
    # Gamma GRL
    if 'gamma_grl' in optuna_params:
        if 'domain_adaptation' not in sc_config['training']:
            sc_config['training']['domain_adaptation'] = {}
        sc_config['training']['domain_adaptation']['gamma_grl'] = round(optuna_params['gamma_grl'], 4)
        print(f"  ✓ gamma_grl = {optuna_params['gamma_grl']:.4f}")
        sync_count += 1

    # Domain weight schedule
    if 'domain_weight_schedule' in optuna_params:
        if 'domain_adaptation' not in sc_config['training']:
            sc_config['training']['domain_adaptation'] = {}
        sc_config['training']['domain_adaptation']['domain_weight_schedule'] = optuna_params['domain_weight_schedule']
        print(f"  ✓ domain_weight_schedule = {optuna_params['domain_weight_schedule']}")
        sync_count += 1

    # ========== 深度结构参数同步 ==========
    if 'model_hyperparameters' not in sc_config['training']:
        sc_config['training']['model_hyperparameters'] = {}
    
    mh = sc_config['training']['model_hyperparameters']
    
    # 1. CNN
    if 'cnn_layers' in optuna_params:
        if 'cnn' not in mh: mh['cnn'] = {}
        mh['cnn']['num_layers'] = optuna_params['cnn_layers']
        print(f"  ✓ cnn_layers = {optuna_params['cnn_layers']}")
        sync_count += 1
        
        # 计算 cnn_channels 列表
        if 'cnn_base_channels' in optuna_params:
            base = optuna_params['cnn_base_channels']
            layers = optuna_params['cnn_layers']
            channels = [base * (2 ** i) for i in range(layers)]
            mh['cnn']['channels'] = channels
            print(f"  ✓ cnn_channels = {channels}")
            sync_count += 1

    # 2. LSTM
    if 'lstm_layers' in optuna_params:
        if 'lstm' not in mh: mh['lstm'] = {}
        mh['lstm']['num_layers'] = optuna_params['lstm_layers']
        mh['lstm']['hidden_size'] = optuna_params.get('lstm_hidden', 128)
        mh['lstm']['dropout'] = round(optuna_params.get('lstm_dropout', 0.5), 4)
        mh['lstm']['bidirectional'] = optuna_params.get('lstm_bidirectional', False)
        print(f"  ✓ lstm_config updated")
        sync_count += 1
        
    # 3. Classifier
    if 'classifier_layers' in optuna_params:
        if 'classifier' not in mh: mh['classifier'] = {}
        mh['classifier']['num_layers'] = optuna_params['classifier_layers']
        mh['classifier']['hidden_dim'] = optuna_params.get('classifier_hidden', 64)
        mh['classifier']['dropout'] = round(optuna_params.get('classifier_dropout', 0.5), 4)
        print(f"  ✓ classifier_config updated")
        sync_count += 1

    # 4. Discriminator
    if 'discriminator_layers' in optuna_params:
        if 'discriminator' not in mh: mh['discriminator'] = {}
        mh['discriminator']['num_layers'] = optuna_params['discriminator_layers']
        mh['discriminator']['hidden_dim'] = optuna_params.get('discriminator_hidden', 64)
        print(f"  ✓ discriminator_config updated")
        sync_count += 1
    
    # 保存更新后的配置
    with open(sc_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(sc_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\n[完成] 共同步 {sync_count} 个参数")
    print(f"配置文件已更新: {sc_config_path}")
    
    return True


def sync_all_conditions(backup: bool = True):
    """同步所有工况的参数"""
    print("=" * 70)
    print("同步所有工况的 Optuna 参数")
    print("=" * 70)
    
    success_count = 0
    failed_conditions = []
    
    for condition in range(6):
        print(f"\n>>> 同步工况 {condition}")
        if sync_single_condition_params(condition, backup=False):  # 只在第一次备份
            success_count += 1
        else:
            failed_conditions.append(condition)
    
    print("\n" + "=" * 70)
    print(f"同步完成: {success_count}/6 成功")
    if failed_conditions:
        print(f"失败的工况: {failed_conditions}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='单工况 Optuna 参数同步')
    
    parser.add_argument('--condition', type=int, default=None,
                        help='工况代码 (0-5)')
    parser.add_argument('--all', action='store_true',
                        help='同步所有工况')
    parser.add_argument('--no_backup', action='store_true',
                        help='不备份配置文件')
    
    args = parser.parse_args()
    
    if args.all:
        sync_all_conditions(backup=not args.no_backup)
    elif args.condition is not None:
        if args.condition not in range(6):
            print(f"[错误] 工况代码必须在0-5之间")
            return
        sync_single_condition_params(args.condition, backup=not args.no_backup)
    else:
        print("[错误] 请指定 --condition或 --all")
        parser.print_help()


if __name__ == "__main__":
    main()
