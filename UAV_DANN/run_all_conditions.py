# -*- coding: utf-8 -*-
"""
==============================================================================
批量运行单工况实验与结果对比
==============================================================================
功能：
- 遍历所有飞行状态（工况）进行单工况迁移训练
- 收集实验结果并生成对比报告

工况 (飞行状态) 共6种:
  0=hover, 1=waypoint, 2=velocity, 3=circling, 4=acce, 5=dece

使用方式：
---------
# 运行所有工况
python run_all_conditions.py

# 运行特定工况
python run_all_conditions.py --conditions 0 1 2

# 仅生成对比报告
python run_all_conditions.py --compare_only

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import sys
import argparse
import json
import yaml
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from train_single_condition import train_single_condition


def collect_results(sc_config_path: str) -> Dict[int, Dict]:
    """收集所有单工况实验结果"""
    with open(sc_config_path, 'r', encoding='utf-8') as f:
        sc_config = yaml.safe_load(f)
    
    results_dir = sc_config['output']['results_dir']
    results = {}
    
    for condition in range(6):
        result_path = os.path.join(results_dir, f'training_results_condition_{condition}.json')
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                results[condition] = json.load(f)
    
    return results


def load_mixed_results(config_path: str) -> Optional[Dict]:
    """加载混合迁移结果"""
    possible_paths = [
        os.path.join(project_root, 'logs', 'training_results.json'),
        os.path.join(project_root, 'results', 'mixed_training_results.json'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None


def generate_comparison_report(mixed_results: Optional[Dict], single_results: Dict[int, Dict], sc_config_path: str) -> str:
    """生成对比报告"""
    with open(sc_config_path, 'r', encoding='utf-8') as f:
        sc_config = yaml.safe_load(f)
    
    condition_names = sc_config['conditions']['names']
    
    lines = [
        "# UAV-DANN 迁移学习结果对比报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 实验概述",
        "",
        "对比混合迁移（所有工况）与单工况迁移（每种飞行状态单独训练）的效果。",
        "",
        "### 工况说明",
        "| 代码 | 名称 | 描述 |",
        "|------|------|------|",
        "| 0 | hover | 悬停 - 评估稳态容错能力 |",
        "| 1 | waypoint | 航点飞行 - 评估轨迹跟踪精度 |",
        "| 2 | velocity | 速度控制 - 评估控制效率 |",
        "| 3 | circling | 绕圈飞行 - 分析Yaw与Roll耦合 |",
        "| 4 | acce | 加速 - 验证动力系统极限 |",
        "| 5 | dece | 减速 - 验证减速控制 |",
        "",
        "---",
        "",
        "## 结果对比",
        "",
        "| 实验类型 | 工况 | 目标域准确率 | F1分数 | 精确率 | 召回率 | 训练时间(分钟) |",
        "|----------|------|--------------|--------|--------|--------|----------------|"
    ]
    
    # 混合迁移
    if mixed_results:
        acc = mixed_results.get('best_target_acc', 'N/A')
        f1 = mixed_results.get('f1_score', 'N/A')
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
        lines.append(f"| 混合迁移 | 全部工况 | {acc_str} | {f1_str} | - | - | - |")
    else:
        lines.append("| 混合迁移 | 全部工况 | N/A | N/A | - | - | - |")
    
    # 单工况
    for cond in range(6):
        cond_name = condition_names.get(cond, f"cond_{cond}")
        
        if cond in single_results:
            r = single_results[cond]
            m = r.get('final_metrics', {})
            acc = r.get('best_target_acc', m.get('accuracy', 'N/A'))
            f1 = m.get('f1_score', 'N/A')
            prec = m.get('precision', 'N/A')
            rec = m.get('recall', 'N/A')
            t = r.get('training_time', 0) / 60
            
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
            prec_str = f"{prec:.4f}" if isinstance(prec, float) else str(prec)
            rec_str = f"{rec:.4f}" if isinstance(rec, float) else str(rec)
            
            lines.append(f"| 单工况 | {cond_name} | {acc_str} | {f1_str} | {prec_str} | {rec_str} | {t:.2f} |")
        else:
            lines.append(f"| 单工况 | {cond_name} | 未运行 | - | - | - | - |")
    
    # 统计
    if single_results:
        accs = [r['best_target_acc'] for r in single_results.values() if 'best_target_acc' in r]
        if accs:
            lines.extend([
                "",
                "---",
                "",
                "## 统计摘要",
                "",
                f"- **单工况平均准确率**: {np.mean(accs):.4f}",
                f"- **单工况最高准确率**: {np.max(accs):.4f}",
                f"- **单工况最低准确率**: {np.min(accs):.4f}",
            ])
    
    return "\n".join(lines)


def run_all_conditions(config_path: str, sc_config_path: str, conditions: Optional[List[int]] = None, compare_only: bool = False):
    """批量运行单工况实验"""
    with open(sc_config_path, 'r', encoding='utf-8') as f:
        sc_config = yaml.safe_load(f)
    
    if conditions is None:
        conditions = sc_config['batch_experiment']['conditions_to_run']
    
    condition_names = sc_config['conditions']['names']
    interval_seconds = sc_config['batch_experiment'].get('interval_seconds', 10)
    
    print("=" * 70)
    print("UAV-DANN 批量单工况实验")
    print("=" * 70)
    print(f"工况: {[condition_names.get(c, c) for c in conditions]}")
    
    if not compare_only:
        for i, cond in enumerate(conditions):
            cond_name = condition_names.get(cond, f"cond_{cond}")
            
            print(f"\n[{i+1}/{len(conditions)}] 训练工况 {cond} ({cond_name})")
            
            # 构造特定工况的配置文件路径
            specific_sc_config_path = os.path.join(
                os.path.dirname(sc_config_path),
                f"condition_{cond}_{cond_name}.yaml"
            )
            
            if not os.path.exists(specific_sc_config_path):
                print(f"[警告] 工况专属配置 {specific_sc_config_path} 不存在，使用默认配置")
                specific_sc_config_path = sc_config_path

            try:
                train_single_condition(config_path, specific_sc_config_path, cond)
            except Exception as e:
                print(f"[错误] 工况 {cond} 训练失败: {e}")
            
            if i < len(conditions) - 1:
                print(f"等待 {interval_seconds} 秒...")
                time.sleep(interval_seconds)
    
    # 生成报告
    print("\n>>> 生成对比报告...")
    
    mixed_results = load_mixed_results(config_path)
    single_results = collect_results(sc_config_path)
    
    report = generate_comparison_report(mixed_results, single_results, sc_config_path)
    
    report_path = sc_config['output']['comparison_report']
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"[信息] 报告已保存: {report_path}")
    
    # CSV记录
    if single_results:
        rows = []
        for cond, result in single_results.items():
            metrics = result.get('final_metrics', {})
            rows.append({
                '实验时间': datetime.now().strftime('%Y-%m-%d'),
                '工况代码': cond,
                '工况名称': result.get('condition_name', ''),
                '最佳Epoch': result.get('best_epoch', ''),
                '目标域准确率': result.get('best_target_acc', ''),
                'F1分数': metrics.get('f1_score', ''),
                '训练时间(秒)': result.get('training_time', '')
            })
        
        df = pd.DataFrame(rows)
        csv_path = sc_config['output']['experiment_log']
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"[信息] 实验记录: {csv_path}")
    
    print("\n批量实验完成!")


def main():
    parser = argparse.ArgumentParser(description='UAV-DANN 批量单工况实验')
    parser.add_argument('--config', type=str, default='./config/config.yaml')
    parser.add_argument('--sc_config', type=str, default='./config/config_single_condition.yaml')
    parser.add_argument('--conditions', type=int, nargs='+', default=None, help='工况代码 (0-5)')
    parser.add_argument('--compare_only', action='store_true', help='仅生成对比报告')
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    if not os.path.isabs(args.sc_config):
        args.sc_config = os.path.join(project_root, args.sc_config)
    
    run_all_conditions(args.config, args.sc_config, args.conditions, args.compare_only)


if __name__ == "__main__":
    main()
