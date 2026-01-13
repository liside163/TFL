# -*- coding: utf-8 -*-
"""
==============================================================================
数据诊断脚本 - 检测类别不平衡和分布问题
==============================================================================
功能：诊断单工况数据集的潜在问题
- 类别分布统计
- 不平衡检测
- 零样本类别检测

使用方式：
---------
python diagnose_data.py --condition 0

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import Counter

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.preprocess_single_condition import load_single_condition_data


def diagnose_single_condition(condition: int, results_dir: str = None):
    """
    诊断单工况数据集

    Args:
        condition: 飞行状态代码 (0-5)
        results_dir: 结果目录
    """
    print("=" * 70)
    print(f"数据诊断报告 - 工况 {condition}")
    print("=" * 70)

    # 加载数据
    if results_dir is None:
        results_dir = './results/single_condition'

    try:
        data_dict = load_single_condition_data(condition, results_dir)
        condition_name = data_dict.get('condition_name', f'Condition_{condition}')
        print(f"\n工况名称: {condition_name}")
    except Exception as e:
        print(f"\n❌ 无法加载数据: {e}")
        print(f"   请确保已运行数据预处理: python data/preprocess_single_condition.py --condition {condition}")
        return

    # 1. 类别分布统计
    print("\n" + "=" * 70)
    print("1. 类别分布统计")
    print("=" * 70)

    source_train_dist = np.bincount(data_dict['y_source_train'], minlength=7)
    source_val_dist = np.bincount(data_dict['y_source_val'], minlength=7)
    target_train_dist = np.bincount(data_dict['y_target_train'], minlength=7)
    target_test_dist = np.bincount(data_dict['y_target_test'], minlength=7)

    fault_names = ['No_Fault', 'Motor', 'Accelerometer', 'Gyroscope',
                   'Magnetometer', 'Barometer', 'GPS']

    print("\n源域 (HIL) 训练集:")
    print(f"  总样本数: {len(data_dict['y_source_train']):,}")
    print(f"  类别分布:")
    for i, (count, name) in enumerate(zip(source_train_dist, fault_names)):
        pct = count / len(data_dict['y_source_train']) * 100 if len(data_dict['y_source_train']) > 0 else 0
        print(f"    类别{i} ({name:12s}): {count:6d} ({pct:5.2f}%)")

    print("\n源域 (HIL) 验证集:")
    print(f"  总样本数: {len(data_dict['y_source_val']):,}")
    print(f"  类别分布:")
    for i, (count, name) in enumerate(zip(source_val_dist, fault_names)):
        pct = count / len(data_dict['y_source_val']) * 100 if len(data_dict['y_source_val']) > 0 else 0
        print(f"    类别{i} ({name:12s}): {count:6d} ({pct:5.2f}%)")

    print("\n目标域 (Real) 测试集:")
    print(f"  总样本数: {len(data_dict['y_target_test']):,}")
    print(f"  类别分布:")
    for i, (count, name) in enumerate(zip(target_test_dist, fault_names)):
        pct = count / len(data_dict['y_target_test']) * 100 if len(data_dict['y_target_test']) > 0 else 0
        print(f"    类别{i} ({name:12s}): {count:6d} ({pct:5.2f}%)")

    # 2. 类别不平衡检测
    print("\n" + "=" * 70)
    print("2. 类别不平衡检测")
    print("=" * 70)

    # 源域训练集不平衡分析
    max_count = source_train_dist.max()
    min_count = source_train_dist[source_train_dist > 0].min() if source_train_dist[source_train_dist > 0].size > 0 else 0
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"\n源域训练集不平衡比: {imbalance_ratio:.1f}:1 (最多类/最少类)")
    print(f"  最多类样本数: {max_count}")
    print(f"  最少类样本数: {min_count}")

    if imbalance_ratio > 50:
        print("  ❌ 严重不平衡！建议:")
        print("     - 使用数据增强 (oversampling)")
        print("     - 调整类别权重")
        print("     - 考虑合并稀有类别")
    elif imbalance_ratio > 10:
        print("  ⚠️  中度不平衡，建议使用类别加权")
    else:
        print("  ✓ 相对平衡")

    # 3. 零样本类别检测
    print("\n" + "=" * 70)
    print("3. 零样本类别检测")
    print("=" * 70)

    zero_classes_train = np.where(source_train_dist == 0)[0]
    zero_classes_val = np.where(source_val_dist == 0)[0]
    zero_classes_target = np.where(target_test_dist == 0)[0]

    if len(zero_classes_train) > 0:
        print(f"\n❌ 源域训练集缺少类别: {zero_classes_train}")
        for cls in zero_classes_train:
            print(f"   类别{cls} ({fault_names[cls]}): 训练集中样本数为0，模型无法学习此故障！")
    else:
        print("\n✓ 源域训练集包含所有类别")

    if len(zero_classes_val) > 0:
        print(f"\n⚠️  源域验证集缺少类别: {zero_classes_val}")
    else:
        print("✓ 源域验证集包含所有类别")

    if len(zero_classes_target) > 0:
        print(f"\n⚠️  目标域测试集缺少类别: {zero_classes_target}")
    else:
        print("✓ 目标域测试集包含所有类别")

    # 4. 域间分布对比
    print("\n" + "=" * 70)
    print("4. 域间分布对比 (源域 vs 目标域)")
    print("=" * 70)

    source_total = source_train_dist.sum() + source_val_dist.sum()
    target_total = target_train_dist.sum() + target_test_dist.sum()

    print(f"\n总样本数对比:")
    print(f"  源域 (HIL):   {source_total:,}")
    print(f"  目标域 (Real): {target_total:,}")
    print(f"  源/目标比: {source_total/target_total:.2f}" if target_total > 0 else "  源/目标比: N/A")

    print(f"\n各类别源/目标域样本数对比:")
    print(f"{'类别':<8} {'故障类型':<15} {'源域':<10} {'目标域':<10} {'比例':<10}")
    print("-" * 60)
    for i in range(7):
        source_count = source_train_dist[i] + source_val_dist[i]
        target_count = target_train_dist[i] + target_test_dist[i]
        ratio = source_count / target_count if target_count > 0 else float('inf')
        print(f"{i:<8} {fault_names[i]:<15} {source_count:<10} {target_count:<10} {ratio:<10.2f}")

    # 5. 推荐的类别权重
    print("\n" + "=" * 70)
    print("5. 推荐的类别权重 (逆频率加权)")
    print("=" * 70)

    # 计算逆频率权重
    class_counts = source_train_dist.astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)  # 避免除零
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 7  # 归一化

    print(f"\n类别权重 (用于CrossEntropyLoss):")
    for i, (w, name) in enumerate(zip(weights, fault_names)):
        print(f"  类别{i} ({name:12s}): {w:.4f}")

    # 6. 总结和建议
    print("\n" + "=" * 70)
    print("6. 总结和建议")
    print("=" * 70)

    issues = []
    warnings = []

    # 检查问题
    if imbalance_ratio > 50:
        issues.append(f"严重类别不平衡 (比例{imbalance_ratio:.1f}:1)")

    if len(zero_classes_train) > 0:
        issues.append(f"训练集缺少{len(zero_classes_train)}个类别")

    if source_total < 5000:
        warnings.append(f"源域样本数较少 ({source_total})，可能欠拟合")

    if imbalance_ratio > 10:
        warnings.append(f"中度类别不平衡，建议使用类别权重")

    # 输出总结
    if len(issues) == 0 and len(warnings) == 0:
        print("\n✅ 数据质量良好，无明显问题")
    else:
        if len(issues) > 0:
            print("\n❌ 发现以下问题:")
            for issue in issues:
                print(f"   - {issue}")

        if len(warnings) > 0:
            print("\n⚠️  注意事项:")
            for warning in warnings:
                print(f"   - {warning}")

    # 训练建议
    print("\n📋 训练建议:")
    if imbalance_ratio > 10:
        print("   ✓ 使用类别加权损失函数 (已在配置中启用)")
        print("   ✓ 建议batch_size设为32-64以包含更多稀有类样本")

    if imbalance_ratio > 50:
        print("   ✓ 考虑使用数据增强 (SMOTE, ADASYN等)")
        print("   ✓ 考虑对稀有类进行oversampling")

    print("   ✓ 使用已修复的域适应参数 (gamma_grl=2.0, warmup=30)")
    print("   ✓ 监控训练过程中源域和目标域的准确率变化")

    print("\n" + "=" * 70)
    print("诊断完成！")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='UAV-DANN 数据诊断工具')
    parser.add_argument('--condition', type=int, default=0,
                        help='飞行状态代码 (0-5): 0=hover, 1=waypoint, 2=velocity, '
                             '3=circling, 4=acce, 5=dece')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='结果目录 (默认: ./results/single_condition)')

    args = parser.parse_args()

    if args.condition not in range(6):
        print(f"[错误] 飞行状态代码必须在0-5之间")
        return

    diagnose_single_condition(args.condition, args.results_dir)


if __name__ == "__main__":
    main()
