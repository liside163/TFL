# -*- coding: utf-8 -*-
"""
==============================================================================
DANN评估脚本
==============================================================================
功能：加载训练好的模型进行评估
- 加载检查点
- 在目标域测试集上评估
- 生成详细的评估报告
- 可视化结果

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from typing import Dict, Optional, List
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.dataloader import get_dataloaders
from data.preprocess import load_processed_data
from models.dann import DANN
from utils.metrics import (
    calculate_metrics,
    get_confusion_matrix,
    plot_confusion_matrix,
    print_classification_report,
    compute_class_accuracy
)


@torch.no_grad()
def evaluate_model(
    model: DANN,
    dataloader,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    评估模型
    
    Args:
        model: DANN模型
        dataloader: 数据加载器
        device: 计算设备
    
    Returns:
        results: 包含预测和标签的字典
            - 'predictions': 预测标签
            - 'labels': 真实标签
            - 'features': 特征向量（用于可视化）
            - 'logits': 原始logits（用于置信度分析）
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_features = []
    all_logits = []
    
    print(">>> 正在评估模型...")
    for batch in tqdm(dataloader, desc='Evaluating'):
        x, y, d = batch
        x = x.to(device)
        
        # 提取特征
        features = model.feature_extractor(x)  # (B, 128)
        
        # 分类
        logits = model.classifier(features)  # (B, 11)
        predictions = torch.argmax(logits, dim=1)  # (B,)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(y.numpy())
        all_features.append(features.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    
    results = {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'features': np.concatenate(all_features, axis=0),
        'logits': np.concatenate(all_logits, axis=0)
    }
    
    return results


def analyze_results(
    results: Dict[str, np.ndarray],
    class_names: List[str],
    output_dir: str
) -> Dict[str, float]:
    """
    分析评估结果
    
    Args:
        results: 评估结果字典
        class_names: 类别名称列表
        output_dir: 输出目录
    
    Returns:
        metrics: 评估指标字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = results['predictions']
    labels = results['labels']
    
    print("\n" + "=" * 70)
    print("评估结果分析")
    print("=" * 70)
    
    # ==================== 1. 总体指标 ====================
    metrics = calculate_metrics(labels, predictions, average='weighted')
    
    print("\n>>> 总体指标:")
    print(f"  准确率 (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1分数 (F1-Score): {metrics['f1_score']:.4f}")
    print(f"  精确率 (Precision): {metrics['precision']:.4f}")
    print(f"  召回率 (Recall): {metrics['recall']:.4f}")
    
    # ==================== 2. 分类报告 ====================
    report = print_classification_report(labels, predictions, class_names)
    
    # 保存分类报告
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("UAV-DANN 目标域评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总体准确率: {metrics['accuracy']:.4f}\n")
        f.write(f"总体F1分数: {metrics['f1_score']:.4f}\n\n")
        f.write("详细分类报告:\n")
        f.write(report)
    print(f"\n[保存] 分类报告: {report_path}")
    
    # ==================== 3. 类别准确率 ====================
    class_acc = compute_class_accuracy(labels, predictions, num_classes=len(class_names))
    
    print("\n>>> 各类别准确率:")
    for c, acc in class_acc.items():
        count = np.sum(labels == c)
        print(f"  {class_names[c]:15s}: {acc:.4f} ({acc*100:.1f}%) - 样本数: {count}")
    
    # ==================== 4. 混淆矩阵 ====================
    cm = get_confusion_matrix(labels, predictions, num_classes=len(class_names))
    
    # 绘制混淆矩阵
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        cm, class_names,
        save_path=cm_path,
        title='Target Domain Confusion Matrix',
        normalize=True
    )
    
    # 非归一化版本
    cm_raw_path = os.path.join(output_dir, 'confusion_matrix_raw.png')
    plot_confusion_matrix(
        cm, class_names,
        save_path=cm_raw_path,
        title='Target Domain Confusion Matrix (Counts)',
        normalize=False
    )
    
    # ==================== 5. 错误分析 ====================
    print("\n>>> 错误分析:")
    
    # 找出最容易混淆的类别对
    np.fill_diagonal(cm, 0)  # 去掉对角线
    max_confusion_idx = np.unravel_index(np.argmax(cm), cm.shape)
    true_class = class_names[max_confusion_idx[0]]
    pred_class = class_names[max_confusion_idx[1]]
    confusion_count = cm[max_confusion_idx]
    
    print(f"  最容易混淆的类别对:")
    print(f"    真实类别: {true_class}")
    print(f"    误判为: {pred_class}")
    print(f"    误判次数: {int(confusion_count)}")
    
    # ==================== 6. 置信度分析 ====================
    logits = results['logits']
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    max_probs = np.max(probs, axis=1)
    
    correct_mask = (predictions == labels)
    
    print("\n>>> 置信度分析:")
    print(f"  平均置信度 (所有样本): {np.mean(max_probs):.4f}")
    print(f"  平均置信度 (正确预测): {np.mean(max_probs[correct_mask]):.4f}")
    print(f"  平均置信度 (错误预测): {np.mean(max_probs[~correct_mask]):.4f}")
    
    # 高置信度错误
    high_conf_wrong = (max_probs > 0.9) & (~correct_mask)
    print(f"  高置信度(>90%)错误数: {np.sum(high_conf_wrong)}")
    
    print("\n" + "=" * 70)
    
    return metrics


def visualize_features(
    results: Dict[str, np.ndarray],
    class_names: List[str],
    output_dir: str,
    method: str = 'tsne'
) -> None:
    """
    可视化特征分布
    
    Args:
        results: 评估结果
        class_names: 类别名称
        output_dir: 输出目录
        method: 降维方法 ('tsne' 或 'pca')
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError:
        print("[警告] 缺少可视化依赖，跳过特征可视化")
        return
    
    features = results['features']
    labels = results['labels']
    
    print(f"\n>>> 特征可视化 ({method.upper()})...")
    
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)
    
    features_2d = reducer.fit_transform(features)
    
    # 绘制
    plt.figure(figsize=(12, 10))
    
    # 使用不同颜色表示不同类别
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    
    # 添加图例
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [class_names[i] for i in range(len(class_names))],
               loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'Feature Visualization ({method.upper()}) - Target Domain')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'feature_visualization_{method}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[保存] 特征可视化: {save_path}")


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    完整评估流程
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录（可选）
    
    Returns:
        metrics: 评估指标
    """
    # ==================== 1. 加载配置 ====================
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("UAV-DANN 模型评估")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    print(f"模型检查点: {checkpoint_path}")
    
    # ==================== 2. 设置设备 ====================
    if config['device']['use_gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        print(f"设备: GPU ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        print("设备: CPU")
    
    # ==================== 3. 加载数据 ====================
    print("\n>>> 加载数据...")
    data_dict = load_processed_data(config['data']['processed_dir'])
    loaders = get_dataloaders(config=config, data_dict=data_dict)
    
    # ==================== 4. 加载模型 ====================
    print("\n>>> 加载模型...")
    model = DANN(config=config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"模型已加载，来自epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"检查点指标: {checkpoint['metrics']}")
    
    # ==================== 5. 评估 ====================
    results = evaluate_model(
        model=model,
        dataloader=loaders['target_test'],
        device=device
    )
    
    # ==================== 6. 分析结果 ====================
    class_names = list(config['fault_types']['labels'].values())
    
    if output_dir is None:
        output_dir = os.path.join(config['logging']['log_dir'], 'evaluation')
    
    metrics = analyze_results(results, class_names, output_dir)
    
    # ==================== 7. 特征可视化 ====================
    if config['evaluation']['visualization'].get('plot_tsne', True):
        visualize_features(results, class_names, output_dir, method='tsne')
    
    print(f"\n评估结果已保存至: {output_dir}")
    
    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='UAV-DANN 评估脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='./config/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型检查点路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 获取绝对路径
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    
    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
