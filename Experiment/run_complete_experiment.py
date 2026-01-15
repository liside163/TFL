"""
完整训练流程脚本
在WSL torch128环境中运行
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# 导入我们的模块
from config import Config
from utils.seed import set_seed
from models.cnn_1d import CNN1D
from methods.baseline import BaselineMethod
from methods.pretrain_finetune import PretrainFinetuneMethod
from methods.mmd import MMDAdapter
from evaluators.metrics import MetricsCalculator
from data.dataset import create_data_loaders


def create_synthetic_data(n_samples=1000, window_size=100, n_features=27, n_classes=11):
    """创建合成数据用于测试"""
    print(f"创建合成数据: {n_samples} 样本")

    # 源域数据 (Hover状态)
    source_windows = np.random.randn(n_samples, window_size, n_features).astype(np.float32)
    source_labels = np.random.randint(0, n_classes, n_samples)

    # 目标域数据 (Waypoint状态 - 分布略有不同)
    target_windows = np.random.randn(n_samples, window_size, n_features).astype(np.float32) * 1.2 + 0.3
    target_labels = np.random.randint(0, n_classes, n_samples)

    # 划分目标域为训练集和测试集
    split_idx = int(n_samples * 0.2)
    target_train_windows = target_windows[:split_idx]
    target_train_labels = target_labels[:split_idx]
    target_test_windows = target_windows[split_idx:]
    target_test_labels = target_labels[split_idx:]

    return {
        'source': (source_windows, source_labels),
        'target_train': (target_train_windows, target_train_labels),
        'target_test': (target_test_windows, target_test_labels)
    }


def run_baseline_experiment():
    """运行Baseline实验"""
    print("\n" + "="*60)
    print("实验 1: Baseline (目标域直接训练)")
    print("="*60)

    config = Config()
    set_seed(config.RANDOM_SEED)

    # 创建数据
    data = create_synthetic_data(n_samples=500)

    # 创建数据加载器
    _, target_test_loader = create_data_loaders(
        *data['target_train'],
        *data['target_test'],
        batch_size=config.BATCH_SIZE
    )

    # 创建模型
    model = CNN1D(input_dim=27, num_classes=11)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\n设备: {device}")
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    method = BaselineMethod(model, config)
    train_loader, _ = create_data_loaders(
        *data['target_train'],
        *data['target_test'],
        batch_size=config.BATCH_SIZE
    )

    start_time = time.time()
    history, metrics = method.train(train_loader, None, num_epochs=5)  # 快速测试5轮
    elapsed = time.time() - start_time

    print(f"\n训练完成! 耗时: {elapsed:.2f}秒")
    print(f"最终F1: {metrics['f1_macro']:.4f}")

    return {
        'method': 'baseline',
        'f1_macro': metrics['f1_macro'],
        'accuracy': metrics['accuracy'],
        'elapsed_time': elapsed
    }


def run_pretrain_experiment():
    """运行Pretrain+Finetune实验"""
    print("\n" + "="*60)
    print("实验 2: Pretrain+Finetune")
    print("="*60)

    config = Config()
    set_seed(config.RANDOM_SEED)

    # 创建数据
    data = create_synthetic_data(n_samples=500)

    # 创建模型
    model = CNN1D(input_dim=27, num_classes=11)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 创建数据加载器
    source_loader, _ = create_data_loaders(
        *data['source'],
        *data['source'],
        batch_size=config.BATCH_SIZE
    )
    target_train_loader, target_test_loader = create_data_loaders(
        *data['target_train'],
        *data['target_test'],
        batch_size=config.BATCH_SIZE
    )

    # 训练
    method = PretrainFinetuneMethod(model, config)

    start_time = time.time()
    source_history, source_metrics = method.pretrain(source_loader, None)

    # 重新创建模型用于微调（简化）
    model2 = CNN1D(input_dim=27, num_classes=11)
    model2.load_state_dict(model.state_dict())
    method2 = PretrainFinetuneMethod(model2, config)
    finetune_history, target_metrics = method2.finetune(target_train_loader, None)

    elapsed = time.time() - start_time

    print(f"\n训练完成! 耗时: {elapsed:.2f}秒")
    print(f"源域F1: {source_metrics['f1_macro']:.4f}")
    print(f"目标域F1: {target_metrics['f1_macro']:.4f}")
    print(f"迁移率: {target_metrics['transfer_rate']:.2f}%")

    return {
        'method': 'pretrain',
        'source_f1': source_metrics['f1_macro'],
        'target_f1': target_metrics['f1_macro'],
        'transfer_rate': target_metrics['transfer_rate'],
        'elapsed_time': elapsed
    }


def run_mmd_experiment():
    """运行MMD实验"""
    print("\n" + "="*60)
    print("实验 3: MMD域适应")
    print("="*60)

    config = Config()
    set_seed(config.RANDOM_SEED)

    # 创建数据
    data = create_synthetic_data(n_samples=500)

    # 创建模型
    model = CNN1D(input_dim=27, num_classes=11)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 创建数据加载器
    source_loader, _ = create_data_loaders(
        *data['source'],
        *data['source'],
        batch_size=config.BATCH_SIZE
    )
    target_train_loader, target_test_loader = create_data_loaders(
        *data['target_train'],
        *data['target_test'],
        batch_size=config.BATCH_SIZE
    )

    # 训练
    method = MMDAdapter(model, config)

    start_time = time.time()
    history, metrics = method.train(
        source_loader,
        target_train_loader,
        None,
        lambda_mmd=config.MMD_LAMBDA,
        num_epochs=5
    )
    elapsed = time.time() - start_time

    print(f"\n训练完成! 耗时: {elapsed:.2f}秒")
    print(f"目标域F1: {metrics['f1_macro']:.4f}")

    return {
        'method': 'mmd',
        'f1_macro': metrics['f1_macro'],
        'elapsed_time': elapsed
    }


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" RflyMAD 迁移学习完整实验 - WSL torch128环境")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("使用CPU训练")

    print(f"PyTorch版本: {torch.__version__}")

    # 运行实验
    results = []

    try:
        # 实验1: Baseline
        result1 = run_baseline_experiment()
        results.append(result1)
    except Exception as e:
        print(f"Baseline实验失败: {e}")

    try:
        # 实验2: Pretrain+Finetune
        result2 = run_pretrain_experiment()
        results.append(result2)
    except Exception as e:
        print(f"Pretrain实验失败: {e}")

    try:
        # 实验3: MMD
        result3 = run_mmd_experiment()
        results.append(result3)
    except Exception as e:
        print(f"MMD实验失败: {e}")

    # 汇总结果
    print("\n" + "="*70)
    print(" 实验结果汇总")
    print("="*70)

    print(f"\n{'方法':<15} {'F1-Score':<12} {'准确率':<12} {'迁移率':<12} {'耗时(秒)':<10}")
    print("-" * 70)

    for r in results:
        f1 = r.get('f1_macro', r.get('target_f1', 0))
        acc = r.get('accuracy', 0)
        transfer = r.get('transfer_rate', 'N/A')
        elapsed = r.get('elapsed_time', 0)

        if transfer != 'N/A':
            transfer = f"{transfer:.2f}%"
        else:
            transfer = "N/A"

        print(f"{r['method']:<15} {f1:<12.4f} {acc:<12.4f} {transfer:<12} {elapsed:<10.2f}")

    # 找最佳方法
    f1_scores = [r.get('f1_macro', r.get('target_f1', 0)) for r in results]
    if f1_scores:
        best_idx = np.argmax(f1_scores)
        best_method = results[best_idx]['method']
        best_f1 = f1_scores[best_idx]

        print("\n" + "="*70)
        print(f" 最佳方法: {best_method} (F1={best_f1:.4f})")
        print("="*70)

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n实验完成!")


if __name__ == '__main__':
    main()
