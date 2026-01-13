#!/usr/bin/env python3
"""
数据集制作脚本 - 从feature_config.json加载参数并制作数据集
使用指定的特征参数提取训练集和测试集特征，保存到pkl文件
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
import argparse
from pathlib import Path
from datetime import datetime
from scipy.signal import butter, filtfilt, hilbert, detrend
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 数据路径（复赛数据集）
TRAIN_ROOT = "/home/zxy_2024/FUSAI_Bearing_Fault_Diagnosis/复赛数据集/复赛训练集"
TEST_ROOT = "/home/zxy_2024/FUSAI_Bearing_Fault_Diagnosis/复赛数据集/复赛测试集"

# 采样频率和窗口参数
FS = 20480
WINDOW_SIZE = 1024
STEP_SIZE = 1024

# 标签映射（10分类 - 复赛数据集）
LABEL_MAP = {
    "inner_broken_train100": 0,
    "inner_missing_train125": 1,
    "inner_wear_train110": 2,
    "normal_train70": 3,
    "outer_broken_train120": 4,
    "outer_missing_train80": 5,
    "outer_wear_train140": 6,
    "roller_broken_train150": 7,
    "roller_missing_train90": 8,
    "roller_wear_train130": 9,
}
ID_TO_LABEL = {v: k.split("_train")[0] for k, v in LABEL_MAP.items()}

# ==================== 特征提取函数 ====================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """带通滤波"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = max(low, 1e-6)
    high = min(high, 0.999999)
    if low >= high:
        return data * 0
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_fft_mag(vib, fs=FS):
    """计算FFT幅度谱"""
    N = len(vib)
    fft_vals = np.abs(fft(vib))[:N // 2]
    freqs = np.linspace(0, fs / 2, len(fft_vals))
    return freqs, fft_vals

def extract_band_psd(vib, spec_bands, out_len_spec, use_log=False):
    """提取多频段频谱特征"""
    freqs, fft_vals = compute_fft_mag(vib)
    feats = []
    for (f1, f2) in spec_bands:
        mask = (freqs >= f1) & (freqs < f2)
        if not np.any(mask):
            feats.append(np.zeros(out_len_spec, dtype=np.float32))
            continue
        subf, subv = freqs[mask], (fft_vals[mask] ** 2)
        tgt = np.linspace(f1 if len(subf) > 0 else 0, f2, out_len_spec)
        if use_log:
            feats.append(np.log1p(np.interp(tgt, subf, subv)).astype(np.float32))
        else:
            feats.append(np.interp(tgt, subf, subv).astype(np.float32))
    return np.stack(feats, axis=0)

def extract_multi_env(vib, env_centers, env_bw, out_len_env, use_log=False):
    """提取多中心包络谱特征"""
    vib_clean = detrend(vib)
    vib_clean = (vib_clean - vib_clean.mean()) / (vib_clean.std() + 1e-8)
    feats = []
    for center in env_centers:
        low = max(1, center - env_bw // 2)
        high = min(int(FS/2) - 1, center + env_bw // 2)
        try:
            filtered = bandpass_filter(vib_clean, low, high, FS)
            env = np.abs(hilbert(filtered))
            N = len(env)
            env_fft = np.abs(fft(env))[:N // 2]
            env_freqs = np.linspace(0, FS / 2, len(env_fft))
            mask = env_freqs <= 200
            if not np.any(mask):
                feats.append(np.zeros(out_len_env, dtype=np.float32))
                continue
            ef, ev = env_freqs[mask], (env_fft[mask] ** 2)
            tgt = np.linspace(0, 200, out_len_env)
            if use_log:
                feats.append(np.log1p(np.interp(tgt, ef, ev)).astype(np.float32))
            else:
                feats.append(np.interp(tgt, ef, ev).astype(np.float32))
        except Exception:
            feats.append(np.zeros(out_len_env, dtype=np.float32))
    return np.stack(feats, axis=0)

# ==================== 数据加载函数 ====================
def load_train_data_with_params(spec_bands, env_centers, env_bw, out_len_spec, out_len_env, use_log=False):
    """
    使用指定参数加载并提取训练集特征
    返回: (spec_features, env_features, labels)
    """
    print("\n开始加载训练集...")
    spec_features = []
    env_features = []
    labels = []
    
    for folder_name, label in LABEL_MAP.items():
        folder_path = os.path.join(TRAIN_ROOT, folder_name)
        if not os.path.exists(folder_path):
            print(f"  警告: 文件夹不存在 {folder_name}")
            continue
        
        file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".xlsx")])
        print(f"  处理类别 {label} ({folder_name}): {len(file_list)} 个文件")
        
        for fname in file_list:
            file_path = os.path.join(folder_path, fname)
            
            try:
                df = pd.read_excel(file_path, header=None, engine="openpyxl")
                vib_data = df.values[:, 1].astype(np.float32)  # 只取第二列（振动信号）
            except Exception as e:
                print(f"    警告: 读取失败 {fname}: {e}")
                continue
            
            # 切割窗口
            n_windows = (len(vib_data) - WINDOW_SIZE) // STEP_SIZE + 1
            if n_windows <= 0:
                continue
            
            for i in range(n_windows):
                start = i * STEP_SIZE
                end = start + WINDOW_SIZE
                window_vib = vib_data[start:end]
                
                # 提取特征
                spec_feat = extract_band_psd(window_vib, spec_bands, out_len_spec, use_log)
                env_feat = extract_multi_env(window_vib, env_centers, env_bw, out_len_env, use_log)
                
                spec_features.append(spec_feat)
                env_features.append(env_feat)
                labels.append(label)
    
    return np.array(spec_features), np.array(env_features), np.array(labels, dtype=np.int64)

def load_test_data_with_params(spec_bands, env_centers, env_bw, out_len_spec, out_len_env, use_log=False):
    """
    使用指定参数加载并提取测试集特征
    返回: (spec_features, env_features, file_names)
    """
    print("\n开始加载测试集...")
    spec_features = []
    env_features = []
    file_names = []
    
    if not os.path.exists(TEST_ROOT):
        print(f"  警告: 测试集路径不存在 {TEST_ROOT}")
        return np.array([]), np.array([]), []
    
    file_list = sorted([f for f in os.listdir(TEST_ROOT) if f.endswith(".xlsx")])
    print(f"  测试集文件数: {len(file_list)}")
    
    for fname in file_list:
        file_path = os.path.join(TEST_ROOT, fname)
        
        try:
            df = pd.read_excel(file_path, header=None, engine="openpyxl")
            vib_data = df.values[:, 1].astype(np.float32)
        except Exception as e:
            print(f"    警告: 读取失败 {fname}: {e}")
            continue
        
        # 切割窗口
        n_windows = (len(vib_data) - WINDOW_SIZE) // STEP_SIZE + 1
        if n_windows <= 0:
            continue
        
        for i in range(n_windows):
            start = i * STEP_SIZE
            end = start + WINDOW_SIZE
            window_vib = vib_data[start:end]
            
            spec_feat = extract_band_psd(window_vib, spec_bands, out_len_spec, use_log)
            env_feat = extract_multi_env(window_vib, env_centers, env_bw, out_len_env, use_log)
            
            spec_features.append(spec_feat)
            env_features.append(env_feat)
            file_names.append(fname)
    
    return np.array(spec_features), np.array(env_features), file_names

# ==================== 主函数 ====================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从feature_config.json制作数据集')
    parser.add_argument('--config', type=str, required=True, 
                       help='feature_config.json文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出pkl文件路径（可选，默认保存到config同目录）')
    
    args = parser.parse_args()
    
    # 读取配置文件
    print("="*60)
    print("数据集制作脚本 - 从feature_config.json加载参数")
    print("="*60)
    print(f"\n配置文件: {args.config}")
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在 {args.config}")
        return
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 解析特征参数
    feature_params = config.get('feature_params', {})
    
    # 频谱参数
    spectrum_params = feature_params.get('spectrum', {})
    spec_bands = [tuple(band) for band in spectrum_params.get('bands', [])]
    out_len_spec = spectrum_params.get('out_len', 128)
    use_log = spectrum_params.get('use_log', False)
    
    # 包络参数
    envelope_params = feature_params.get('envelope', {})
    env_centers = envelope_params.get('centers', [])
    env_bw = envelope_params.get('bandwidth', 456.0)
    out_len_env = envelope_params.get('out_len', 32)
    
    # 打印参数信息
    print(f"\n特征参数:")
    print(f"  频谱参数:")
    print(f"    频段数: {len(spec_bands)}")
    print(f"    频率范围: {spec_bands[0][0]:.1f} - {spec_bands[-1][1]:.1f} Hz")
    print(f"    输出维度: {out_len_spec}")
    print(f"    使用对数: {use_log}")
    print(f"  包络参数:")
    print(f"    中心数: {len(env_centers)}")
    print(f"    中心频率范围: {env_centers[0]:.1f} - {env_centers[-1]:.1f} Hz")
    print(f"    带宽: {env_bw:.1f} Hz")
    print(f"    输出维度: {out_len_env}")
    print(f"  其他信息:")
    print(f"    试验编号: {config.get('trial_number', 'N/A')}")
    print(f"    平均准确率: {config.get('mean_acc', 'N/A'):.2f}%")
    print(f"    最佳准确率: {config.get('best_acc', 'N/A'):.2f}%")
    
    # 提取训练集特征
    print(f"\n{'='*60}")
    print("开始提取训练集特征...")
    print(f"{'='*60}")
    
    spec_train, env_train, labels_train = load_train_data_with_params(
        spec_bands, env_centers, env_bw, out_len_spec, out_len_env, use_log
    )
    
    print(f"\n训练集特征提取完成:")
    print(f"  频谱特征: {spec_train.shape}")
    print(f"  包络特征: {env_train.shape}")
    print(f"  标签: {labels_train.shape}")
    print(f"  各类别样本数:")
    for label_id in range(10):
        count = np.sum(labels_train == label_id)
        print(f"    类别 {label_id} ({ID_TO_LABEL[label_id]}): {count} 个样本")
    
    # 提取测试集特征
    print(f"\n{'='*60}")
    print("开始提取测试集特征...")
    print(f"{'='*60}")
    
    spec_test, env_test, test_files = load_test_data_with_params(
        spec_bands, env_centers, env_bw, out_len_spec, out_len_env, use_log
    )
    
    if len(spec_test) > 0:
        print(f"\n测试集特征提取完成:")
        print(f"  频谱特征: {spec_test.shape}")
        print(f"  包络特征: {env_test.shape}")
        print(f"  样本数: {len(spec_test)}")
        print(f"  文件数: {len(set(test_files))}")
    else:
        print("\n测试集为空或路径不存在")
    
    # ==================== 关键修复：归一化数据 ====================
    print(f"\n{'='*60}")
    print("归一化数据（使用训练集统计量）...")
    print(f"{'='*60}")
    
    # 计算训练集的归一化统计量（按通道计算）
    spec_mean = spec_train.mean(axis=(0, 2), keepdims=True)  # (1, n_bands, 1)
    spec_std = spec_train.std(axis=(0, 2), keepdims=True) + 1e-8
    
    env_mean = env_train.mean(axis=(0, 2), keepdims=True)  # (1, n_centers, 1)
    env_std = env_train.std(axis=(0, 2), keepdims=True) + 1e-8
    
    print(f"  训练集统计量:")
    print(f"    频谱: mean={spec_mean.mean():.4f}, std={spec_std.mean():.4f}")
    print(f"    包络: mean={env_mean.mean():.4f}, std={env_std.mean():.4f}")
    
    # 归一化训练集
    spec_train_norm = (spec_train - spec_mean) / spec_std
    env_train_norm = (env_train - env_mean) / env_std
    
    print(f"  归一化后训练集:")
    print(f"    频谱: mean={spec_train_norm.mean():.4f}, std={spec_train_norm.std():.4f}")
    print(f"    包络: mean={env_train_norm.mean():.4f}, std={env_train_norm.std():.4f}")
    
    # 使用训练集的统计量归一化测试集
    if len(spec_test) > 0:
        spec_test_norm = (spec_test - spec_mean) / spec_std
        env_test_norm = (env_test - env_mean) / env_std
        
        print(f"  归一化后测试集:")
        print(f"    频谱: mean={spec_test_norm.mean():.4f}, std={spec_test_norm.std():.4f}")
        print(f"    包络: mean={env_test_norm.mean():.4f}, std={env_test_norm.std():.4f}")
    else:
        spec_test_norm = spec_test
        env_test_norm = env_test
    
    # 构建数据集字典（保存归一化后的数据）
    dataset = {
        'x_train': {
            'spec': spec_train_norm,
            'env': env_train_norm
        },
        'y_train': labels_train,
        'x_test': {
            'spec': spec_test_norm,
            'env': env_test_norm
        },
        'test_files': test_files,
        'label_map': LABEL_MAP,
        'id_to_label': ID_TO_LABEL,
        'feature_params': feature_params,
        'config': config,
        # 归一化参数（重要！）
        'normalization': {
            'spec_mean': spec_mean,
            'spec_std': spec_std,
            'env_mean': env_mean,
            'env_std': env_std,
            'method': 'per_channel',
            'description': '每个通道独立计算均值和标准差，使用训练集统计量归一化测试集'
        },
        'metadata': {
            'fs': FS,
            'window_size': WINDOW_SIZE,
            'step_size': STEP_SIZE,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'n_train_samples': len(labels_train),
            'n_test_samples': len(spec_test) if len(spec_test) > 0 else 0,
            'num_classes': 10,
            'trial_number': config.get('trial_number', None),
            'mean_acc': config.get('mean_acc', None),
            'std_acc': config.get('std_acc', None),
            'best_acc': config.get('best_acc', None),
            'fold_accuracies': config.get('fold_accuracies', None),
            'normalized': True,
            'description': f'使用feature_config.json (trial {config.get("trial_number", "N/A")}) 制作的数据集（已归一化）'
        }
    }
    
    # 确定输出路径
    if args.output is None:
        config_dir = os.path.dirname(os.path.abspath(args.config))
        trial_number = config.get('trial_number', 'unknown')
        output_path = os.path.join(config_dir, f'dataset_trial_{trial_number}.pkl')
    else:
        output_path = args.output
    
    # 保存数据集
    print(f"\n{'='*60}")
    print("保存数据集...")
    print(f"{'='*60}")
    print(f"输出路径: {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    # 打印数据集摘要
    print(f"\n{'='*60}")
    print("数据集制作完成！")
    print(f"{'='*60}")
    print(f"\n数据集信息:")
    print(f"  文件路径: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"\n数据集结构:")
    print(f"  x_train:")
    print(f"    spec: {spec_train_norm.shape} - 频谱特征（已归一化）")
    print(f"    env: {env_train_norm.shape} - 包络特征（已归一化）")
    print(f"  y_train: {labels_train.shape} - 训练标签")
    print(f"  x_test:")
    print(f"    spec: {spec_test_norm.shape if len(spec_test)>0 else spec_test.shape} - 测试频谱特征（已归一化）")
    print(f"    env: {env_test_norm.shape if len(spec_test)>0 else env_test.shape} - 测试包络特征（已归一化）")
    print(f"  test_files: {len(test_files)} - 测试文件名")
    print(f"  label_map: 类别映射字典")
    print(f"  id_to_label: ID到类别名映射")
    print(f"  feature_params: 特征参数配置")
    print(f"  normalization: 归一化参数（spec_mean, spec_std, env_mean, env_std）")
    print(f"  config: 完整配置信息")
    print(f"  metadata: 元数据")
    
    print(f"\n⚠️  重要提示:")
    print(f"  ✓ 数据已按通道归一化（训练集统计量）")
    print(f"  ✓ 测试时直接使用，无需再次归一化")
    print(f"  ✓ 归一化参数已保存在 dataset['normalization']")
    
    print(f"\n使用示例:")
    print(f"  import pickle")
    print(f"  with open('{output_path}', 'rb') as f:")
    print(f"      dataset = pickle.load(f)")
    print(f"  # 直接使用归一化后的数据")
    print(f"  spec_train = dataset['x_train']['spec']  # 已归一化")
    print(f"  env_train = dataset['x_train']['env']    # 已归一化")
    print(f"  y_train = dataset['y_train']")
    print(f"  spec_test = dataset['x_test']['spec']    # 已归一化")
    print(f"  env_test = dataset['x_test']['env']      # 已归一化")
    
    print(f"\n{'='*60}")
    print("✅ 完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

