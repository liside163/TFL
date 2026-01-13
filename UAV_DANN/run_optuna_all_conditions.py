# -*- coding: utf-8 -*-
"""
==============================================================================
批量运行所有工况的 Optuna 调优
==============================================================================
功能：遍历所有6种飞行状态进行超参数优化

使用方式：
---------
# 运行所有工况调优
python run_optuna_all_conditions.py --n_trials 50

# 运行特定工况
python run_optuna_all_conditions.py --conditions 0 1 2 --n_trials 30

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import sys
import argparse
import time
import json
import yaml
from datetime import datetime
from typing import List, Optional, Dict

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from optuna_tune_single_condition import run_single_condition_optuna


def run_all_conditions_optuna(
    config_path: str,
    sc_config_path: str,
    conditions: Optional[List[int]] = None,
    n_trials: int = 50
):
    """
    批量运行所有工况的 Optuna 调优
    
    Args:
        config_path: 主配置路径
        sc_config_path: 单工况配置路径
        conditions: 要运行的工况列表，默认所有
        n_trials: 每个工况的试验次数
    """
    # 加载配置
    with open(sc_config_path, 'r', encoding='utf-8') as f:
        sc_config = yaml.safe_load(f)
    
    if conditions is None:
        conditions = list(range(6))  # 0-5
    
    condition_names = sc_config['conditions']['names']
    interval_seconds = sc_config['batch_experiment'].get('interval_seconds', 10)
    
    print("=" * 70)
    print("批量工况 Optuna 调优")
    print("=" * 70)
    print(f"工况: {[condition_names.get(c, c) for c in conditions]}")
    print(f"每个工况试验次数: {n_trials}")
    print(f"间隔时间: {interval_seconds} 秒")
    print()
    
    results = {}
    
    for i, condition in enumerate(conditions):
        condition_name = condition_names.get(condition, f"condition_{condition}")
        
        print(f"\n[{i+1}/{len(conditions)}] 开始调优工况 {condition} ({condition_name})")
        print("=" * 70)
        
        # 构造特定工况的配置文件路径
        specific_sc_config_path = os.path.join(
            os.path.dirname(sc_config_path),
            f"condition_{condition}_{condition_name}.yaml"
        )
        
        if not os.path.exists(specific_sc_config_path):
            print(f"[警告] 工况专属配置 {specific_sc_config_path} 不存在，使用默认配置")
            specific_sc_config_path = sc_config_path

        try:
            best_params = run_single_condition_optuna(
                condition=condition,
                config_path=config_path,
                sc_config_path=specific_sc_config_path,
                n_trials=n_trials
            )
            results[condition] = {
                'success': True,
                'best_params': best_params,
                'condition_name': condition_name
            }
        except Exception as e:
            print(f"\n[错误] 工况 {condition} ({condition_name}) 调优失败: {str(e)}")
            results[condition] = {
                'success': False,
                'error': str(e),
                'condition_name': condition_name
            }
        
        # 间隔等待
        if i < len(conditions) - 1:
            print(f"\n等待 {interval_seconds} 秒后继续...")
            time.sleep(interval_seconds)
    
    # 生成汇总报告
    print("\n" + "=" * 70)
    print("批量调优完成！")
    print("=" * 70)
    
    generate_summary_report(results, sc_config)
    
    print("\n所有工况调优完成!")


def generate_summary_report(results: Dict, sc_config: Dict):
    """生成汇总报告"""
    condition_names = sc_config['conditions']['names']
    
    # Markdown 报告
    report_lines = [
        "# 单工况 Optuna 调优汇总报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 调优结果",
        "",
        "| 工况代码 | 工况名称 | 状态 | 最优参数路径 |",
        "|----------|----------|------|--------------|"
    ]
    
    for condition, result in sorted(results.items()):
        condition_name = result.get('condition_name', f'condition_{condition}')
        
        if result.get('success'):
            status = "✅ 成功"
            params_path = f"`optuna_results/condition_{condition}_best_params.yaml`"
        else:
            status = f"❌ 失败: {result.get('error', 'Unknown')}"
            params_path = "-"
        
        report_lines.append(f"| {condition} | {condition_name} | {status} | {params_path} |")
    
    # 添加最优参数展示
    report_lines.extend([
        "",
        "---",
        "",
        "## 最优参数详情",
        ""
    ])
    
    for condition, result in sorted(results.items()):
        if result.get('success'):
            condition_name = result.get('condition_name', f'condition_{condition}')
            best_params = result.get('best_params', {})
            
            report_lines.extend([
                f"### {condition_name} (工况{condition})",
                "",
                "```yaml"
            ])
            
            for key, value in best_params.items():
                report_lines.append(f"{key}: {value}")
            
            report_lines.extend([
                "```",
                ""
            ])
    
    report = "\n".join(report_lines)
    
    # 保存报告
    optuna_results_dir = './optuna_results'
    os.makedirs(optuna_results_dir, exist_ok=True)
    
    report_path = os.path.join(optuna_results_dir, 'all_conditions_optuna_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[信息] 汇总报告已保存至: {report_path}")
    
    # 保存 JSON 格式
    json_path = os.path.join(optuna_results_dir, 'all_conditions_optuna_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"[信息] JSON 汇总已保存至: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='批量工况 Optuna 调优')
    
    parser.add_argument('--config', type=str, default='./config/config.yaml')
    parser.add_argument('--sc_config', type=str, default='./config/config_single_condition.yaml')
    parser.add_argument('--conditions', type=int, nargs='+', default=None,
                        help='要调优的工况代码 (0-5)，默认所有')
    parser.add_argument('--n_trials', type=int, default=150,
                        help='每个工况的试验次数')
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    if not os.path.isabs(args.sc_config):
        args.sc_config = os.path.join(project_root, args.sc_config)
    
    run_all_conditions_optuna(args.config, args.sc_config, args.conditions, args.n_trials)


if __name__ == "__main__":
    main()
