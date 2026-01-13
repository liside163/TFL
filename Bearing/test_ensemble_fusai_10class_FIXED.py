#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复赛10分类集成预测测试脚本（TTA增强版）- 修复归一化问题

🔧 关键修复:
1. 测试集使用训练集的mean/std进行归一化
2. 保存和加载归一化参数
3. 确保训练和测试的一致性
"""

import os
import sys
import argparse
import pickle
import json
import re
from collections import Counter
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 配置参数 ====================
DATASET_PATH = "/home/zxy_2024/FUSAI_Bearing_Fault_Diagnosis/results/optimization_cv_20251105_105552/trial_54/dataset_trial_54.pkl"
MODEL_DIR = "/home/zxy_2024/FUSAI_Bearing_Fault_Diagnosis/results/optimization_cv_20251105_105552/trial_54"
TTA_PARAMS_PATH = "/home/zxy_2024/FUSAI_Bearing_Fault_Diagnosis/tta_best_params.json"
OUTPUT_DIR = f"{MODEL_DIR}/test_predictions_FIXED"

# ==================== SimplePModel 模型定义 ====================
class SimplePModel(nn.Module):
    """简化的P模型（复赛10分类版本）"""
    def __init__(self, n_spec_bands, n_env_centers, out_len, num_classes=10, dropout=0.3):
        super().__init__()
        
        self.spec_branch = nn.Sequential(
            nn.Conv1d(n_spec_bands, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout(dropout)
        )
        
        self.env_branch = nn.Sequential(
            nn.Conv1d(n_env_centers, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, spec, env):
        spec_out = self.spec_branch(spec).view(spec.size(0), -1)
        env_out = self.env_branch(env).view(env.size(0), -1)
        combined = torch.cat([spec_out, env_out], dim=1)
        return self.classifier(combined)

# ==================== 归一化函数 ====================
def compute_normalization_stats(spec_train, env_train):
    """
    计算训练集的归一化统计量
    返回: (spec_mean, spec_std, env_mean, env_std)
    """
    print("  计算训练集归一化统计量...")
    
    # 按通道计算均值和标准差
    spec_mean = spec_train.mean(axis=(0, 2), keepdims=True)  # (1, n_bands, 1)
    spec_std = spec_train.std(axis=(0, 2), keepdims=True) + 1e-8
    
    env_mean = env_train.mean(axis=(0, 2), keepdims=True)  # (1, n_centers, 1)
    env_std = env_train.std(axis=(0, 2), keepdims=True) + 1e-8
    
    print(f"    频谱: mean={spec_mean.mean():.4f}, std={spec_std.mean():.4f}")
    print(f"    包络: mean={env_mean.mean():.4f}, std={env_std.mean():.4f}")
    
    return spec_mean, spec_std, env_mean, env_std

def normalize_data(spec, env, spec_mean, spec_std, env_mean, env_std):
    """
    使用给定的统计量归一化数据
    """
    spec_norm = (spec - spec_mean) / spec_std
    env_norm = (env - env_mean) / env_std
    return spec_norm, env_norm

# ==================== TTA增强函数 ====================
def _add_noise(arr, snr_db):
    """添加高斯噪声"""
    sig_pow = np.mean(arr ** 2)
    if sig_pow < 1e-10:
        return arr
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_pow), size=arr.shape)
    return arr + noise

def _time_shift(arr, shift_ratio):
    """时间平移"""
    L = arr.shape[-1]
    shift = int(np.round(L * shift_ratio))
    return np.roll(arr, shift, axis=-1)

def _scale(arr, factor):
    """幅值缩放"""
    return arr * factor

def _mirror(arr):
    """镜像翻转"""
    return np.flip(arr, axis=-1).copy()

def augment_sample_tta(spec_sample, env_sample, tta_params):
    """为单个样本生成8个TTA变体"""
    shift_ratio = tta_params['shift_ratio']
    snr_db = tta_params['snr_db']
    scale_low = tta_params['scale_low']
    scale_high = tta_params['scale_high']
    
    spec_vars = []
    env_vars = []

    # 1-8: 各种增强
    spec_vars.append(spec_sample.copy())
    env_vars.append(env_sample.copy())
    
    spec_vars.append(_time_shift(spec_sample, +shift_ratio))
    env_vars.append(_time_shift(env_sample, +shift_ratio))
    
    spec_vars.append(_time_shift(spec_sample, -shift_ratio))
    env_vars.append(_time_shift(env_sample, -shift_ratio))
    
    spec_vars.append(_add_noise(spec_sample, snr_db))
    env_vars.append(_add_noise(env_sample, snr_db))
    
    spec_vars.append(_add_noise(spec_sample, snr_db))
    env_vars.append(_add_noise(env_sample, snr_db))
    
    spec_vars.append(_scale(spec_sample, scale_low))
    env_vars.append(_scale(env_sample, scale_low))
    
    spec_vars.append(_scale(spec_sample, scale_high))
    env_vars.append(_scale(env_sample, scale_high))
    
    spec_vars.append(_mirror(spec_sample))
    env_vars.append(_mirror(env_sample))

    return spec_vars, env_vars

# ==================== TTA集成预测 ====================
def tta_ensemble_predict(models, spec_test_norm, env_test_norm, device, tta_params, T=1.0, fusion='softmax'):
    """
    使用TTA + ensemble在测试集上预测
    注意: spec_test_norm 和 env_test_norm 必须已经归一化
    """
    n_samples = spec_test_norm.shape[0]
    final_preds = []
    decision_details = []

    for idx in tqdm(range(n_samples), desc="TTA集成预测"):
        spec_s = spec_test_norm[idx]
        env_s = env_test_norm[idx]

        # 生成8个TTA变体
        spec_vars, env_vars = augment_sample_tta(spec_s, env_s, tta_params)

        all_logits = []
        all_preds = []

        with torch.no_grad():
            for model in models:
                for spec_v, env_v in zip(spec_vars, env_vars):
                    spec_t = torch.tensor(spec_v, dtype=torch.float32).unsqueeze(0).to(device)
                    env_t = torch.tensor(env_v, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    logits = model(spec_t, env_t)
                    all_logits.append(logits.cpu().numpy()[0])
                    pred = torch.argmax(logits, dim=1).item()
                    all_preds.append(pred)

        all_logits = np.array(all_logits)
        
        if fusion == 'softmax':
            scaled_logits = all_logits / T
            probs = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            avg_probs = np.mean(probs, axis=0)
            final_pred = np.argmax(avg_probs)
            confidence = avg_probs[final_pred]
        else:  # vote
            vote_counts = Counter(all_preds)
            final_pred = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[final_pred] / len(all_preds)
            avg_probs = np.zeros(10)
            for pred_id, count in vote_counts.items():
                avg_probs[pred_id] = count / len(all_preds)

        final_preds.append(final_pred)
        
        detail = {
            'individual_predictions': all_preds,
            'vote_counts': dict(Counter(all_preds)),
            'avg_probabilities': avg_probs.tolist(),
            'confidence': float(confidence)
        }
        decision_details.append(detail)

    return np.array(final_preds), decision_details

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='复赛10分类集成预测（修复归一化）')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH)
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR)
    parser.add_argument('--tta_params', type=str, default=TTA_PARAMS_PATH)
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    parser.add_argument('--vote', action='store_true')
    parser.add_argument('--no_tta', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载数据集
    print(f"\n加载数据集: {args.dataset}")
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)
    
    spec_train = dataset['x_train']['spec']
    env_train = dataset['x_train']['env']
    spec_test = dataset['x_test']['spec']
    env_test = dataset['x_test']['env']
    test_files = dataset['test_files']
    label_map = dataset['label_map']
    id_to_label = dataset['id_to_label']
    
    test_sample_names = sorted(list(set(test_files)))
    
    print(f"  训练集: spec {spec_train.shape}, env {env_train.shape}")
    print(f"  测试集: spec {spec_test.shape}, env {env_test.shape}")
    print(f"  测试文件数: {len(test_sample_names)}")
    
    # 🔧 检查数据是否已归一化
    is_normalized = dataset.get('metadata', {}).get('normalized', False)
    
    if is_normalized:
        print(f"\n✓ 数据已归一化（pkl文件中）- 直接使用")
        print(f"  训练集统计: spec mean={spec_train.mean():.4f}, std={spec_train.std():.4f}")
        print(f"  测试集统计: spec mean={spec_test.mean():.4f}, std={spec_test.std():.4f}")
        
        # 直接使用已归一化的数据
        spec_train_norm = spec_train
        env_train_norm = env_train
        spec_test_norm = spec_test
        env_test_norm = env_test
        
        # 从dataset中获取归一化参数
        if 'normalization' in dataset:
            norm_params = {
                'spec_mean': dataset['normalization']['spec_mean'].tolist() if hasattr(dataset['normalization']['spec_mean'], 'tolist') else dataset['normalization']['spec_mean'],
                'spec_std': dataset['normalization']['spec_std'].tolist() if hasattr(dataset['normalization']['spec_std'], 'tolist') else dataset['normalization']['spec_std'],
                'env_mean': dataset['normalization']['env_mean'].tolist() if hasattr(dataset['normalization']['env_mean'], 'tolist') else dataset['normalization']['env_mean'],
                'env_std': dataset['normalization']['env_std'].tolist() if hasattr(dataset['normalization']['env_std'], 'tolist') else dataset['normalization']['env_std']
            }
        else:
            norm_params = None
    else:
        print(f"\n⚠️  数据未归一化 - 需要手动归一化")
        print(f"  原始训练集统计: spec mean={spec_train.mean():.4f}, std={spec_train.std():.4f}")
        
        # 计算训练集的归一化统计量
        spec_mean, spec_std, env_mean, env_std = compute_normalization_stats(spec_train, env_train)
        
        # 归一化训练集和测试集
        spec_train_norm, env_train_norm = normalize_data(spec_train, env_train, spec_mean, spec_std, env_mean, env_std)
        spec_test_norm, env_test_norm = normalize_data(spec_test, env_test, spec_mean, spec_std, env_mean, env_std)
        
        print(f"  归一化后:")
        print(f"    训练集: spec mean={spec_train_norm.mean():.4f}, std={spec_train_norm.std():.4f}")
        print(f"    测试集: spec mean={spec_test_norm.mean():.4f}, std={spec_test_norm.std():.4f}")
        
        # 保存归一化参数
        norm_params = {
            'spec_mean': spec_mean.tolist(),
            'spec_std': spec_std.tolist(),
            'env_mean': env_mean.tolist(),
            'env_std': env_std.tolist()
        }
    
    # 保存归一化参数（如果有）
    if norm_params:
        os.makedirs(args.output, exist_ok=True)
        with open(f"{args.output}/normalization_params.json", 'w') as f:
            json.dump(norm_params, f, indent=2)
        print(f"  ✓ 归一化参数已保存: {args.output}/normalization_params.json")
    
    # 2. 加载TTA参数
    print(f"\n加载TTA参数: {args.tta_params}")
    with open(args.tta_params, 'r') as f:
        tta_params = json.load(f)
    
    if args.no_tta:
        tta_params = {'shift_ratio': 0.0, 'snr_db': 1000.0, 'scale_low': 1.0, 'scale_high': 1.0}
    
    # 3. 加载模型
    print(f"\n加载模型: {args.model_dir}")
    model_files = sorted([f for f in os.listdir(args.model_dir) if f.startswith('fold_') and f.endswith('.pth')])
    
    feature_params = dataset.get('feature_params', dataset.get('config', {}).get('feature_params', {}))
    n_spec_bands = feature_params['spectrum']['n_bands']
    n_env_centers = feature_params['envelope']['n_centers']
    out_len = max(feature_params['spectrum']['out_len'], feature_params['envelope']['out_len'])
    
    models = []
    for model_file in model_files:
        model_path = os.path.join(args.model_dir, model_file)
        model = SimplePModel(n_spec_bands, n_env_centers, out_len, num_classes=10, dropout=0.3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    
    print(f"  成功加载 {len(models)} 个模型")
    
    # 4. TTA集成预测（使用归一化后的数据）
    print(f"\n开始预测...")
    
    file_to_indices = {}
    for idx, filename in enumerate(test_files):
        if filename not in file_to_indices:
            file_to_indices[filename] = []
        file_to_indices[filename].append(idx)
    
    # 🔧 使用归一化后的测试数据
    window_predictions, decision_details = tta_ensemble_predict(
        models, spec_test_norm, env_test_norm, device, tta_params, T=1.0, fusion='softmax'
    )
    
    # 按文件聚合
    file_predictions = {}
    file_confidences = {}  # 存储每个文件的置信度
    
    for filename in test_sample_names:
        indices = file_to_indices[filename]
        window_preds = window_predictions[indices]
        
        # 对该文件的所有窗口进行投票
        vote_counts = Counter(window_preds)
        final_pred = vote_counts.most_common(1)[0][0]
        file_predictions[filename] = final_pred
        
        # 计算该文件的平均置信度（所有窗口的平均）
        window_confidences = [decision_details[idx]['confidence'] for idx in indices]
        avg_confidence = np.mean(window_confidences)
        file_confidences[filename] = avg_confidence
    
    # 5. 生成结果
    print("\n生成预测结果...")
    
    simple_results = ["测试集名称\t故障类型"]
    detailed_results = ["测试集名称\t故障类型\t平均置信度\t窗口数"]
    
    for filename in test_sample_names:
        pred_id = file_predictions[filename]
        pred_label = id_to_label[int(pred_id)]
        clean_name = filename.replace('.xlsx', '').replace('.XLSX', '')
        confidence = file_confidences[filename]
        n_windows = len(file_to_indices[filename])
        
        simple_results.append(f"{clean_name}\t{pred_label}")
        detailed_results.append(f"{clean_name}\t{pred_label}\t{confidence:.4f}\t{n_windows}")
    
    # 6. 保存结果
    output_txt = os.path.join(args.output, "test_predictions.txt")
    output_detailed = os.path.join(args.output, "test_predictions_detailed.txt")
    
    with open(output_txt, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(simple_results))
    
    with open(output_detailed, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(detailed_results))
    
    # 7. 计算置信度统计
    all_confidences = list(file_confidences.values())
    avg_confidence = np.mean(all_confidences)
    min_confidence = np.min(all_confidences)
    max_confidence = np.max(all_confidences)
    std_confidence = np.std(all_confidences)
    median_confidence = np.median(all_confidences)
    
    print(f"\n✅ 预测完成!")
    print(f"  测试文件数: {len(test_sample_names)}")
    print(f"  简单结果: {output_txt}")
    print(f"  详细结果: {output_detailed}")
    
    # 8. 置信度统计
    print(f"\n📊 置信度统计:")
    print(f"  平均置信度: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    print(f"  中位数置信度: {median_confidence:.4f} ({median_confidence*100:.2f}%)")
    print(f"  最小置信度: {min_confidence:.4f} ({min_confidence*100:.2f}%)")
    print(f"  最大置信度: {max_confidence:.4f} ({max_confidence*100:.2f}%)")
    print(f"  置信度标准差: {std_confidence:.4f}")
    
    # 置信度分段统计
    high_conf = sum(1 for c in all_confidences if c >= 0.8)
    med_conf = sum(1 for c in all_confidences if 0.5 <= c < 0.8)
    low_conf = sum(1 for c in all_confidences if c < 0.5)
    
    print(f"\n  置信度分段:")
    print(f"    高置信度 (>=80%): {high_conf} ({100.0*high_conf/len(all_confidences):.1f}%)")
    print(f"    中置信度 (50-80%): {med_conf} ({100.0*med_conf/len(all_confidences):.1f}%)")
    print(f"    低置信度 (<50%): {low_conf} ({100.0*low_conf/len(all_confidences):.1f}%)")
    
    # 9. 预测分布统计
    pred_counts = Counter(file_predictions.values())
    print("\n📊 预测类别分布:")
    for label_id in sorted(pred_counts.keys()):
        label_name = id_to_label[label_id]
        count = pred_counts[label_id]
        percentage = 100.0 * count / len(test_sample_names)
        
        # 计算该类别的平均置信度
        class_confidences = [file_confidences[f] for f in test_sample_names if file_predictions[f] == label_id]
        class_avg_conf = np.mean(class_confidences) if class_confidences else 0
        
        print(f"  {label_name}: {count} ({percentage:.1f}%) - 平均置信度: {class_avg_conf:.3f}")
    
    print("\n🔧 修复说明:")
    print("  ✓ 测试集已使用训练集的mean/std进行归一化")
    print("  ✓ 确保训练和测试的一致性")
    print("  ✓ 归一化参数已保存供后续使用")
    
    print("\n" + "="*60)
    print("✅ 测试完成（已修复归一化问题）!")
    print("="*60)

if __name__ == "__main__":
    main()

