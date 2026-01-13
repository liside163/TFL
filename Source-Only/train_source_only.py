# =====================================================================
# Source-Only 主训练脚本
# 仅使用源域(HIL)数据训练模型，在目标域(Real)上评估
# =====================================================================

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    DEVICE, 
    TRAIN_CONFIG, 
    FLIGHT_STATE_MAPPING,
    AVAILABLE_FLIGHT_STATES,
    FAULT_IDX_TO_LABEL,
    NUM_CLASSES,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    RESULT_DIR,
    print_config
)
from data.dataset import create_dataloaders, UAVDataset
from models.source_only_model import build_model
from utils.metrics import (
    compute_metrics, 
    compute_confusion_matrix, 
    print_classification_report,
    MetricTracker
)
from utils.visualization import (
    plot_training_curves, 
    plot_confusion_matrix,
    plot_tsne,
    plot_class_distribution
)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> tuple:
    """
    训练一个epoch
    
    维度说明:
    - 输入batch: [batch_size, seq_len, features] = [B, 1000, 30]
    - 模型输出: [batch_size, num_classes] = [B, 7]
    - 标签: [batch_size] = [B]
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
    
    Returns:
        (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        # 数据移至设备
        # data形状: [B, 1000, 30], labels形状: [B]
        data = data.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(data)  # [B, 7]
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)  # [B]
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    在验证/测试集上评估模型
    
    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        (average_loss, accuracy, all_preds, all_labels)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def extract_features(
    model: nn.Module,
    data_loader,
    device: torch.device
) -> tuple:
    """
    提取特征向量（用于t-SNE可视化）
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
    
    Returns:
        (features, labels)
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # 获取特征向量
            features = model.get_features(data)  # [B, 256]
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.concatenate(all_features, axis=0), np.array(all_labels)


def save_experiment_results(
    args,
    metrics: dict,
    history: dict,
    save_dir: Path
):
    """
    保存实验结果为JSON和表格格式
    
    Args:
        args: 命令行参数
        metrics: 评估指标
        history: 训练历史
        save_dir: 保存目录
    """
    # 创建结果记录
    result = {
        "实验名称": f"Source-Only_{args.flight_state}",
        "实验时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "飞行状态": args.flight_state,
        "方法": "Source-Only (无领域自适应)",
        "源域": "HIL",
        "目标域": "Real",
        "类别数": NUM_CLASSES,
        "配置": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
        },
        "源域结果": {
            "验证准确率": f"{metrics['val_acc']:.4f}",
            "验证精确率": f"{metrics['val_metrics']['precision']:.4f}",
            "验证召回率": f"{metrics['val_metrics']['recall']:.4f}",
            "验证F1": f"{metrics['val_metrics']['f1']:.4f}",
        },
        "目标域结果": {
            "准确率": f"{metrics['target_acc']:.4f}",
            "精确率": f"{metrics['target_metrics']['precision']:.4f}",
            "召回率": f"{metrics['target_metrics']['recall']:.4f}",
            "F1": f"{metrics['target_metrics']['f1']:.4f}",
        }
    }
    
    # 保存JSON
    json_path = save_dir / "experiment_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n实验结果已保存到: {json_path}")
    
    return result


def generate_experiment_table(results: list, save_path: Path):
    """
    生成实验记录表格
    
    Args:
        results: 实验结果列表
        save_path: 保存路径
    """
    import pandas as pd
    
    # 转换为DataFrame
    table_data = []
    for r in results:
        table_data.append({
            "实验时间": r["实验时间"],
            "飞行状态": r["飞行状态"],
            "方法": r["方法"],
            "源域验证ACC": r["源域结果"]["验证准确率"],
            "源域验证F1": r["源域结果"]["验证F1"],
            "目标域ACC": r["目标域结果"]["准确率"],
            "目标域F1": r["目标域结果"]["F1"],
            "备注": "基线实验"
        })
    
    df = pd.DataFrame(table_data)
    
    # 保存为CSV
    csv_path = save_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 保存为Markdown表格
    md_path = save_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 实验结果记录表\n\n")
        f.write(df.to_markdown(index=False))
    
    print(f"实验表格已保存到: {csv_path}")
    print(f"实验表格已保存到: {md_path}")


def main(args):
    """主训练函数"""
    
    # 打印配置
    print_config()
    print(f"\n飞行状态: {args.flight_state}")
    print(f"设备: {DEVICE}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    
    # 创建实验目录
    exp_name = f"source_only_{args.flight_state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = RESULT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1. 数据加载
    # ==========================================
    print("\n" + "=" * 60)
    print("1. 数据加载")
    print("=" * 60)
    
    train_loader, val_loader, target_loader, scaler = create_dataloaders(
        flight_state=args.flight_state,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # ==========================================
    # 2. 模型构建
    # ==========================================
    print("\n" + "=" * 60)
    print("2. 模型构建")
    print("=" * 60)
    
    model = build_model(DEVICE)
    
    # ==========================================
    # 3. 训练设置
    # ==========================================
    print("\n" + "=" * 60)
    print("3. 训练设置")
    print("=" * 60)
    
    # 获取类别权重（处理类别不平衡）
    train_dataset = train_loader.dataset.dataset  # 获取原始数据集
    class_weights = train_dataset.get_class_weights().to(DEVICE)
    print(f"类别权重: {class_weights}")
    
    # 损失函数（使用类别权重）
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=TRAIN_CONFIG["scheduler_factor"],
        patience=TRAIN_CONFIG["scheduler_patience"],
        min_lr=TRAIN_CONFIG["min_lr"]
    )
    print(f"学习率调度器已配置: ReduceLROnPlateau (mode=max, factor={TRAIN_CONFIG['scheduler_factor']}, patience={TRAIN_CONFIG['scheduler_patience']})")
    
    # 指标跟踪器
    tracker = MetricTracker()
    
    # 早停设置
    best_val_acc = 0.0
    patience_counter = 0
    
    # ==========================================
    # 4. 训练循环
    # ==========================================
    print("\n" + "=" * 60)
    print("4. 开始训练")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # 验证
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, DEVICE
        )
        
        # 目标域评估（使用伪标签）
        target_loss, target_acc, target_preds, target_labels = evaluate(
            model, target_loader, criterion, DEVICE
        )
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新指标
        tracker.update(
            epoch, train_loss, train_acc, 
            val_loss, val_acc, target_acc, current_lr
        )
        
        # 更新学习率
        scheduler.step(val_acc)
        
        # 打印进度
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch:3d}/{args.epochs}] | "
              f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
              f"Target Acc={target_acc:.4f} | "
              f"LR={current_lr:.2e} | "
              f"Time={epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # 保存模型
            checkpoint_path = exp_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'target_acc': target_acc,
            }, checkpoint_path)
            print(f"  ✓ 保存最佳模型 (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
            print(f"\n早停触发: {patience_counter} epochs没有改善")
            break
        
        # 测试模式：只训练1个epoch
        if args.test_run:
            print("\n测试模式：只训练1个epoch")
            break
    
    total_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {total_time/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {tracker.best_epoch})")
    
    # ==========================================
    # 5. 最终评估
    # ==========================================
    print("\n" + "=" * 60)
    print("5. 最终评估")
    print("=" * 60)
    
    # 加载最佳模型
    checkpoint = torch.load(exp_dir / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 源域验证集评估
    _, val_acc, val_preds, val_labels = evaluate(
        model, val_loader, criterion, DEVICE
    )
    val_metrics = compute_metrics(val_labels, val_preds)
    
    print("\n--- 源域 (HIL) 验证集结果 ---")
    print_classification_report(val_labels, val_preds)
    
    # 目标域评估
    _, target_acc, target_preds, target_labels = evaluate(
        model, target_loader, criterion, DEVICE
    )
    target_metrics = compute_metrics(target_labels, target_preds)
    
    print("\n--- 目标域 (Real) 结果 ---")
    print_classification_report(target_labels, target_preds)
    
    # ==========================================
    # 6. 可视化
    # ==========================================
    print("\n" + "=" * 60)
    print("6. 生成可视化")
    print("=" * 60)
    
    # 训练曲线
    plot_training_curves(
        tracker.get_history(),
        save_path=exp_dir / "training_curves.png",
        title=f"Source-Only ({args.flight_state}) 训练曲线"
    )
    
    # 源域混淆矩阵
    val_cm = compute_confusion_matrix(val_labels, val_preds)
    plot_confusion_matrix(
        val_cm,
        save_path=exp_dir / "confusion_matrix_source.png",
        title=f"源域 (HIL) 混淆矩阵 - {args.flight_state}"
    )
    
    # 目标域混淆矩阵
    target_cm = compute_confusion_matrix(target_labels, target_preds)
    plot_confusion_matrix(
        target_cm,
        save_path=exp_dir / "confusion_matrix_target.png",
        title=f"目标域 (Real) 混淆矩阵 - {args.flight_state}"
    )
    
    # t-SNE可视化（可选）
    if args.tsne:
        print("正在生成t-SNE可视化...")
        val_features, val_feat_labels = extract_features(model, val_loader, DEVICE)
        target_features, target_feat_labels = extract_features(model, target_loader, DEVICE)
        
        # 合并特征
        all_features = np.concatenate([val_features, target_features], axis=0)
        all_labels = np.concatenate([val_feat_labels, target_feat_labels], axis=0)
        domain_labels = np.concatenate([
            np.zeros(len(val_features)),  # 源域=0
            np.ones(len(target_features))  # 目标域=1
        ], axis=0)
        
        plot_tsne(
            all_features, all_labels, domain_labels,
            save_path=exp_dir / "tsne_visualization.png",
            title=f"t-SNE 特征可视化 - {args.flight_state}"
        )
    
    # ==========================================
    # 7. 保存结果
    # ==========================================
    print("\n" + "=" * 60)
    print("7. 保存实验结果")
    print("=" * 60)
    
    # 汇总指标
    all_metrics = {
        'val_acc': val_acc,
        'val_metrics': val_metrics,
        'target_acc': target_acc,
        'target_metrics': target_metrics
    }
    
    # 保存实验结果
    result = save_experiment_results(args, all_metrics, tracker.get_history(), exp_dir)
    
    # 保存标准化器
    scaler.save(exp_dir / "scaler.npz")
    
    # 生成实验表格
    generate_experiment_table([result], RESULT_DIR / "experiment_records")
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)
    print(f"结果保存目录: {exp_dir}")
    print(f"\n--- 关键结果 ---")
    print(f"源域验证准确率: {val_acc:.4f}")
    print(f"目标域准确率: {target_acc:.4f}")
    print(f"域偏移导致的性能下降: {val_acc - target_acc:.4f} ({(val_acc-target_acc)/val_acc*100:.1f}%)")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source-Only 迁移学习实验")
    
    # 必需参数
    parser.add_argument(
        "--flight_state", 
        type=str, 
        default="hover",
        choices=AVAILABLE_FLIGHT_STATES,  # 使用可用的飞行状态（排除dece）
        help="飞行状态 (hover, waypoint, velocity, circling, acce，注意:dece在REAL中无数据)"
    )
    
    # 可选参数
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=TRAIN_CONFIG["batch_size"],
        help="批次大小"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=TRAIN_CONFIG["epochs"],
        help="训练轮数"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=TRAIN_CONFIG["learning_rate"],
        help="学习率"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=TRAIN_CONFIG["num_workers"],
        help="DataLoader工作进程数"
    )
    parser.add_argument(
        "--tsne", 
        action="store_true",
        help="是否生成t-SNE可视化"
    )
    parser.add_argument(
        "--test_run", 
        action="store_true",
        help="测试模式（只训练1个epoch）"
    )
    
    args = parser.parse_args()
    
    main(args)
