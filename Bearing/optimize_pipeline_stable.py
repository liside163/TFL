#!/usr/bin/env python3
"""
轴承故障诊断参数优化流程 - 稳定版（复赛数据集 - 10分类）
整合数据处理、特征提取和模型训练，使用Optuna寻找最佳参数组合
每次试验进行5折交叉验证，取平均准确率，提高评估稳定性
"""

import os
import shutil
import numpy as np
import pandas as pd
import pickle
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
from datetime import datetime
import json
from scipy.signal import butter, filtfilt, hilbert, detrend
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 数据路径（复赛数据集）
TRAIN_ROOT = "/home/zxy_2024/FUSAI_Bearing_Fault_Diagnosis/复赛数据集/复赛训练集"  # 训练集文件夹（10个子文件夹）
TEST_ROOT = "/home/zxy_2024/FUSAI_Bearing_Fault_Diagnosis/复赛数据集/复赛测试集"    # 测试集文件夹（所有测试文件）

# 优化参数
N_TRIALS = 500  # Optuna 试验次数
OPTIMIZATION_TIMEOUT = 2400000  # 优化超时时间（秒）
N_FOLDS = 5  # 5折交叉验证

# 模型训练参数
TRAIN_EPOCHS = 100  # 每次trial的训练轮数（较少以加快优化）
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 60

# 采样频率
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
        return data * 0  # 返回零信号
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_fft_mag(vib, fs=FS):
    """计算FFT幅度谱"""
    N = len(vib)
    fft_vals = np.abs(fft(vib))[:N // 2]
    freqs = np.linspace(0, fs / 2, len(fft_vals))
    return freqs, fft_vals

def extract_band_psd(vib, spec_bands, out_len_spec, use_log=True):
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

def extract_multi_env(vib, env_centers, env_bw, out_len_env, use_log=True):
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
def load_train_data_with_params(spec_bands, env_centers, env_bw, out_len_spec, out_len_env, use_log=True):
    """
    使用指定参数加载并提取训练集特征
    返回: (spec_features, env_features, labels)
    """
    spec_features = []
    env_features = []
    labels = []
    
    for folder_name, label in LABEL_MAP.items():
        folder_path = os.path.join(TRAIN_ROOT, folder_name)
        if not os.path.exists(folder_path):
            continue
        
        # 对文件列表排序，确保顺序一致
        file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".xlsx")])
        
        for fname in file_list:
            file_path = os.path.join(folder_path, fname)
            
            try:
                df = pd.read_excel(file_path, header=None, engine="openpyxl")
                vib_data = df.values[:, 1].astype(np.float32)  # 只取第二列（振动信号）
            except Exception:
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

def load_test_data_with_params(spec_bands, env_centers, env_bw, out_len_spec, out_len_env, use_log=True):
    """
    使用指定参数加载并提取测试集特征
    返回: (spec_features, env_features, file_names)
    """
    spec_features = []
    env_features = []
    file_names = []
    
    if not os.path.exists(TEST_ROOT):
        return np.array([]), np.array([]), []
    
    # 对文件列表排序，确保顺序一致
    file_list = sorted([f for f in os.listdir(TEST_ROOT) if f.endswith(".xlsx")])
    
    for fname in file_list:
        file_path = os.path.join(TEST_ROOT, fname)
        
        try:
            df = pd.read_excel(file_path, header=None, engine="openpyxl")
            vib_data = df.values[:, 1].astype(np.float32)
        except Exception:
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

# ==================== 简化的P模型 ====================
class SimplePModel(nn.Module):
    """简化的P模型用于快速优化"""
    def __init__(self, n_spec_bands, n_env_centers, out_len, num_classes=10, dropout=0.3):
        super().__init__()
        
        # 频谱分支
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
        
        # 包络分支
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
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, spec, env):
        # spec: (batch, n_spec_bands, out_len)
        # env: (batch, n_env_centers, out_len)
        spec_out = self.spec_branch(spec).view(spec.size(0), -1)
        env_out = self.env_branch(env).view(env.size(0), -1)
        combined = torch.cat([spec_out, env_out], dim=1)
        return self.classifier(combined)

# ==================== 训练和评估函数（5折交叉验证）====================
def train_and_evaluate_single_fold(spec_train, env_train, y_train, spec_val, env_val, y_val, device, n_spec_bands, n_env_centers, out_len, random_seed):
    """
    单折训练并返回验证集准确率
    """
    # 设置随机种子以保证可重复性
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # 归一化
    spec_mean = spec_train.mean(axis=(0, 2), keepdims=True)
    spec_std = spec_train.std(axis=(0, 2), keepdims=True) + 1e-8
    spec_train_norm = (spec_train - spec_mean) / spec_std
    spec_val_norm = (spec_val - spec_mean) / spec_std
    
    env_mean = env_train.mean(axis=(0, 2), keepdims=True)
    env_std = env_train.std(axis=(0, 2), keepdims=True) + 1e-8
    env_train_norm = (env_train - env_mean) / env_std
    env_val_norm = (env_val - env_mean) / env_std
    
    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(spec_train_norm),
        torch.FloatTensor(env_train_norm),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(spec_val_norm),
        torch.FloatTensor(env_val_norm),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型（10分类）
    model = SimplePModel(n_spec_bands, n_env_centers, out_len, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(TRAIN_EPOCHS):
        # 训练阶段
        model.train()
        for spec_batch, env_batch, y_batch in train_loader:
            spec_batch = spec_batch.to(device)
            env_batch = env_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(spec_batch, env_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for spec_batch, env_batch, y_batch in val_loader:
                spec_batch = spec_batch.to(device)
                env_batch = env_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(spec_batch, env_batch)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break
    
    return best_val_acc

def train_and_evaluate_cross_validation(spec_features, env_features, labels, device, n_spec_bands, n_env_centers, out_len, n_folds=N_FOLDS, return_best_model=False, return_all_models=False):
    """
    5折交叉验证训练并返回平均准确率
    如果return_best_model=True，返回(mean_acc, std_acc, best_fold_acc, best_model_state)
    如果return_all_models=True，返回(mean_acc, std_acc, best_fold_acc, all_fold_models, fold_accuracies)
    """
    # 设置随机种子
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # 创建分层K折交叉验证
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    fold_results = []
    fold_accuracies = []
    best_fold_acc = 0
    best_model_state = None
    all_fold_models = []  # 存储所有折的模型
    
    print(f"  开始{n_folds}折交叉验证...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(spec_features, labels)):
        print(f"    折 {fold_idx + 1}/{n_folds}: 训练样本 {len(train_idx)}, 验证样本 {len(val_idx)}")
        
        # 划分数据
        spec_train_fold = spec_features[train_idx]
        env_train_fold = env_features[train_idx]
        y_train_fold = labels[train_idx]
        
        spec_val_fold = spec_features[val_idx]
        env_val_fold = env_features[val_idx]
        y_val_fold = labels[val_idx]
        
        # 如果只需要准确率，使用快速评估
        if not return_best_model and not return_all_models:
            fold_acc = train_and_evaluate_single_fold(
                spec_train_fold, env_train_fold, y_train_fold,
                spec_val_fold, env_val_fold, y_val_fold,
                device, n_spec_bands, n_env_centers, out_len, 
                random_seed + fold_idx
            )
            fold_results.append(fold_acc)
            fold_accuracies.append(fold_acc)
        else:
            # 需要模型权重，获取完整模型
            fold_model_state, fold_acc = train_and_get_best_model_with_acc(
                spec_train_fold, env_train_fold, y_train_fold,
                spec_val_fold, env_val_fold, y_val_fold,
                device, n_spec_bands, n_env_centers, out_len, 
                random_seed + fold_idx
            )
            fold_results.append(fold_acc)
            fold_accuracies.append(fold_acc)
            
            # 保存这一折的模型
            if return_all_models:
                all_fold_models.append(fold_model_state)
            
            # 记录最佳模型（如果需要）
            if return_best_model and fold_acc > best_fold_acc:
                best_fold_acc = fold_acc
                best_model_state = fold_model_state
        
        print(f"      折 {fold_idx + 1} 准确率: {fold_acc:.2f}%")
    
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    best_acc = np.max(fold_results)
    
    print(f"  {n_folds}折交叉验证结果:")
    print(f"    各折准确率: {[f'{acc:.2f}%' for acc in fold_results]}")
    print(f"    平均准确率: {mean_acc:.2f}% (±{std_acc:.2f}%)")
    print(f"    最佳准确率: {best_acc:.2f}%")
    
    if return_all_models:
        return mean_acc, std_acc, best_acc, all_fold_models, fold_accuracies
    elif return_best_model:
        return mean_acc, std_acc, best_acc, best_model_state
    else:
        return mean_acc, std_acc, best_acc

def train_and_get_best_model_with_acc(spec_train, env_train, y_train, spec_val, env_val, y_val, device, n_spec_bands, n_env_centers, out_len, random_seed):
    """
    训练并返回最佳模型状态和准确率
    返回: (best_model_state, best_val_acc)
    """
    # 设置随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # 归一化
    spec_mean = spec_train.mean(axis=(0, 2), keepdims=True)
    spec_std = spec_train.std(axis=(0, 2), keepdims=True) + 1e-8
    spec_train_norm = (spec_train - spec_mean) / spec_std
    spec_val_norm = (spec_val - spec_mean) / spec_std
    
    env_mean = env_train.mean(axis=(0, 2), keepdims=True)
    env_std = env_train.std(axis=(0, 2), keepdims=True) + 1e-8
    env_train_norm = (env_train - env_mean) / env_std
    env_val_norm = (env_val - env_mean) / env_std
    
    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(spec_train_norm),
        torch.FloatTensor(env_train_norm),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(spec_val_norm),
        torch.FloatTensor(env_val_norm),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型（10分类）
    model = SimplePModel(n_spec_bands, n_env_centers, out_len, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(TRAIN_EPOCHS):
        # 训练阶段
        model.train()
        for spec_batch, env_batch, y_batch in train_loader:
            spec_batch = spec_batch.to(device)
            env_batch = env_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(spec_batch, env_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for spec_batch, env_batch, y_batch in val_loader:
                spec_batch = spec_batch.to(device)
                env_batch = env_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(spec_batch, env_batch)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break
    
    return best_model_state, best_val_acc

def train_and_get_best_model(spec_train, env_train, y_train, spec_val, env_val, y_val, device, n_spec_bands, n_env_centers, out_len, random_seed):
    """
    训练并返回最佳模型状态（兼容旧接口）
    """
    model_state, _ = train_and_get_best_model_with_acc(
        spec_train, env_train, y_train, spec_val, env_val, y_val,
        device, n_spec_bands, n_env_centers, out_len, random_seed
    )
    return model_state


# ==================== Optuna优化目标函数 ====================
class BearingOptimizer:
    def __init__(self, device, save_dir):
        self.device = device
        self.save_dir = save_dir
        self.best_params = None
        self.best_score = 0
        self.all_trial_results = []  # 保存所有试验的详细结果
        
        # 保存最佳模型：最高平均准确率（5折交叉验证）
        self.best_mean_acc = 0  # 全局最高平均准确率
        self.best_mean_model = None  # 对应模型
        self.best_mean_params = None  # 对应参数
        self.best_fold_results = []  # 最佳试验的各折结果
        self.best_all_fold_models = []  # 最佳试验的所有5折模型
        self.best_fold_accuracies = []  # 最佳试验的各折准确率
    
    def objective(self, trial):
        """Optuna目标函数（5折交叉验证）"""
        try:
            # 1. 定义搜索空间
            n_spec_bands = trial.suggest_int('n_spec_bands', 3, 8)
            n_env_centers = trial.suggest_int('n_env_centers', 3, 8)
            
            # 频谱频段参数
            spec_low_freq = trial.suggest_int('spec_low_freq', 0, 2000)
            spec_high_freq = trial.suggest_int('spec_high_freq', 3000, 8000)
            
            # 包络参数
            env_low_center = trial.suggest_int('env_low_center', 500, 2000)
            env_high_center = trial.suggest_int('env_high_center', 3000, 8000)
            env_bw = trial.suggest_int('env_bw', 200, 800)
            
            # 特征维度
            out_len_spec = trial.suggest_categorical('out_len_spec', [32, 48, 64, 96, 128, 256])
            out_len_env = trial.suggest_categorical('out_len_env', [32, 48, 64, 96, 128, 256])
            
            # 其他参数
            use_log = trial.suggest_categorical('use_log', [True, False])
            
            # 确保参数有效
            if spec_low_freq >= spec_high_freq:
                return 0.0
            if env_low_center >= env_high_center:
                return 0.0
            
            # 2. 生成频段和中心频率
            spec_bands = list(zip(
                np.linspace(spec_low_freq, spec_high_freq, n_spec_bands + 1)[:-1],
                np.linspace(spec_low_freq, spec_high_freq, n_spec_bands + 1)[1:]
            ))
            
            env_centers = np.linspace(env_low_center, env_high_center, n_env_centers).tolist()
            
            print(f"\n{'='*60}")
            print(f"Trial {trial.number}: 提取特征...")
            print(f"  频谱频段: {n_spec_bands}段, {spec_low_freq}-{spec_high_freq}Hz")
            print(f"  包络中心: {n_env_centers}个, {env_low_center}-{env_high_center}Hz, 带宽={env_bw}")
            print(f"  输出维度: spec={out_len_spec}, env={out_len_env}")
            
            # 3. 提取特征
            spec_features, env_features, labels = load_train_data_with_params(
                spec_bands, env_centers, env_bw, out_len_spec, out_len_env, use_log
            )
            
            if len(labels) < 100:  # 样本数太少
                print(f"  样本数不足: {len(labels)}")
                return 0.0
            
            print(f"  特征提取完成: {spec_features.shape}, {env_features.shape}")
            
            # 4. 5折交叉验证训练并评估（获取所有5折的模型）
            print(f"  开始5折交叉验证训练...")
            mean_acc, std_acc, best_acc, all_fold_models, fold_accuracies = train_and_evaluate_cross_validation(
                spec_features, env_features, labels,
                self.device, n_spec_bands, n_env_centers,
                max(out_len_spec, out_len_env), n_folds=N_FOLDS, return_all_models=True
            )
            
            print(f"  5折交叉验证结果:")
            print(f"    平均准确率: {mean_acc:.2f}% (±{std_acc:.2f}%)")
            print(f"    最佳准确率: {best_acc:.2f}%")
            print(f"    各折准确率: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
            print(f"{'='*60}")
            
            # 5. 记录最佳参数（包含特征参数配置）
            trial_params = {
                'n_spec_bands': n_spec_bands,
                'n_env_centers': n_env_centers,
                'spec_bands': spec_bands,  # 频谱频段配置
                'env_centers': env_centers,  # 包络中心频率配置
                'env_bw': env_bw,  # 包络带宽
                'out_len_spec': out_len_spec,
                'out_len_env': out_len_env,
                'use_log': use_log,
                'spec_low_freq': spec_low_freq,
                'spec_high_freq': spec_high_freq,
                'env_low_center': env_low_center,
                'env_high_center': env_high_center
            }
            
            trial_result = {
                'trial_number': trial.number,
                'params': trial_params.copy(),
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'best_acc': best_acc,
                'fold_accuracies': fold_accuracies,
                'n_folds': N_FOLDS
            }
            
            self.all_trial_results.append(trial_result)
            
            # 更新全局最佳平均准确率模型
            if mean_acc > self.best_mean_acc:
                # 删除旧的最佳模型文件（如果存在）
                old_model_path = f"{self.save_dir}/best_cv_model.pth"
                old_cv_models_dir = f"{self.save_dir}/best_cv_5fold_models"
                
                if os.path.exists(old_model_path):
                    try:
                        os.remove(old_model_path)
                        print(f"   已删除旧的最佳折模型: {old_model_path}")
                    except Exception as e:
                        print(f"   警告: 删除旧模型失败: {e}")
                
                if os.path.exists(old_cv_models_dir):
                    try:
                        shutil.rmtree(old_cv_models_dir)
                        print(f"   已删除旧的5折模型目录: {old_cv_models_dir}")
                    except Exception as e:
                        print(f"   警告: 删除旧模型目录失败: {e}")
                
                self.best_mean_acc = mean_acc
                # 保存最佳折的模型（向后兼容）
                best_fold_idx = np.argmax(fold_accuracies)
                self.best_mean_model = all_fold_models[best_fold_idx]
                self.best_mean_params = trial_params.copy()
                self.best_mean_params['best_acc'] = best_acc
                self.best_mean_params['mean_acc'] = mean_acc
                self.best_mean_params['std_acc'] = std_acc
                self.best_fold_results = fold_accuracies.copy()
                self.best_all_fold_models = all_fold_models.copy()  # 保存所有5折模型
                self.best_fold_accuracies = fold_accuracies.copy()
                
                # 立即保存最佳平均准确率模型（最佳折）
                model_path = f"{self.save_dir}/best_cv_model.pth"
                torch.save(self.best_mean_model, model_path)
                
                # 保存所有5折的模型权重
                cv_models_dir = f"{self.save_dir}/best_cv_5fold_models"
                Path(cv_models_dir).mkdir(parents=True, exist_ok=True)
                
                for fold_idx, fold_model in enumerate(all_fold_models):
                    fold_model_path = f"{cv_models_dir}/fold_{fold_idx+1}_acc{fold_accuracies[fold_idx]:.2f}.pth"
                    torch.save(fold_model, fold_model_path)
                
                # 保存特征参数配置（JSON格式，便于读取）
                feature_config = {
                    'trial_number': trial.number,
                    'mean_acc': mean_acc,
                    'std_acc': std_acc,
                    'best_acc': best_acc,
                    'fold_accuracies': fold_accuracies,
                    'feature_params': {
                        'spectrum': {
                            'n_bands': n_spec_bands,
                            'bands': [[float(f1), float(f2)] for f1, f2 in spec_bands],
                            'low_freq': float(spec_low_freq),
                            'high_freq': float(spec_high_freq),
                            'out_len': out_len_spec,
                            'use_log': use_log
                        },
                        'envelope': {
                            'n_centers': n_env_centers,
                            'centers': [float(c) for c in env_centers],
                            'bandwidth': float(env_bw),
                            'low_center': float(env_low_center),
                            'high_center': float(env_high_center),
                            'out_len': out_len_env,
                            'use_log': use_log
                        }
                    }
                }
                
                config_path = f"{cv_models_dir}/feature_config.json"
                with open(config_path, 'w') as f:
                    json.dump(feature_config, f, indent=2)
                
                print(f"\n🎯 新的最佳平均准确率 (试验 #{trial.number}): {mean_acc:.2f}% (±{std_acc:.2f}%)")
                print(f"   最佳折模型已保存: {model_path}")
                print(f"   所有5折模型已保存到: {cv_models_dir}/")
                print(f"   特征参数配置已保存: {config_path}")
                print(f"   各折准确率: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
            
            # 使用平均准确率作为优化目标
            if mean_acc > self.best_score:
                self.best_score = mean_acc
                self.best_params = trial_params.copy()
                self.best_params['best_acc'] = best_acc
                self.best_params['mean_acc'] = mean_acc
                self.best_params['std_acc'] = std_acc
            
            # 返回平均准确率作为优化目标
            return mean_acc
            
        except Exception as e:
            print(f"Trial {trial.number} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def optimize(self, n_trials=N_TRIALS):
        """执行优化"""
        print("🚀 开始参数优化（5折交叉验证版 - 10分类）...")
        print(f"总试验次数: {n_trials}")
        print(f"每次试验交叉验证折数: {N_FOLDS}")
        print(f"每次训练轮数: {TRAIN_EPOCHS}")
        print(f"结果保存目录: {self.save_dir}")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(self.objective, n_trials=n_trials, timeout=OPTIMIZATION_TIMEOUT)
        
        print(f"\n🎯 优化完成!")
        print(f"\n全局最佳结果:")
        print(f"  最佳平均准确率: {self.best_mean_acc:.2f}%")
        print(f"  优化目标准确率: {self.best_score:.2f}%")
        
        print(f"\n📊 最佳平均准确率对应参数:")
        for key, value in self.best_mean_params.items():
            if key not in ['best_acc', 'mean_acc', 'std_acc', 'spec_bands', 'env_centers']:
                print(f"  {key}: {value}")
        
        # 保存最佳模型的配置信息（包含所有5折模型和特征参数）
        cv_acc_info = {
            'model_type': 'best_cv_accuracy',
            'num_classes': 10,
            'mean_accuracy': self.best_mean_acc,
            'std_accuracy': self.best_mean_params.get('std_acc', 0),
            'best_fold_accuracy': self.best_mean_params.get('best_acc', 0),
            'fold_accuracies': self.best_fold_accuracies,
            'params': self.best_mean_params,
            'feature_params': {
                'spectrum': {
                    'n_bands': self.best_mean_params.get('n_spec_bands'),
                    'bands': [[float(f1), float(f2)] for f1, f2 in self.best_mean_params.get('spec_bands', [])],
                    'low_freq': float(self.best_mean_params.get('spec_low_freq', 0)),
                    'high_freq': float(self.best_mean_params.get('spec_high_freq', 0)),
                    'out_len': self.best_mean_params.get('out_len_spec'),
                    'use_log': self.best_mean_params.get('use_log', False)
                },
                'envelope': {
                    'n_centers': self.best_mean_params.get('n_env_centers'),
                    'centers': [float(c) for c in self.best_mean_params.get('env_centers', [])],
                    'bandwidth': float(self.best_mean_params.get('env_bw', 0)),
                    'low_center': float(self.best_mean_params.get('env_low_center', 0)),
                    'high_center': float(self.best_mean_params.get('env_high_center', 0)),
                    'out_len': self.best_mean_params.get('out_len_env'),
                    'use_log': self.best_mean_params.get('use_log', False)
                }
            },
            'description': '全局最高5折交叉验证平均准确率模型（10分类）',
            'n_folds': N_FOLDS,
            'models_info': {
                'best_fold_model': 'best_cv_model.pth',
                'all_5fold_models_dir': 'best_cv_5fold_models/',
                'feature_config': 'best_cv_5fold_models/feature_config.json'
            }
        }
        
        with open(f"{self.save_dir}/best_cv_config.json", 'w') as f:
            json.dump(cv_acc_info, f, indent=2)
        
        print(f"\n💾 模型配置已保存:")
        print(f"  - best_cv_model.pth: 最佳折模型")
        print(f"  - best_cv_5fold_models/: 所有5折模型目录")
        print(f"    - fold_1_acc*.pth ~ fold_5_acc*.pth: 5个折的模型权重")
        print(f"    - feature_config.json: 特征参数配置")
        print(f"  - best_cv_config.json: 完整配置信息")
        
        return self.best_params, self.best_score, study

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("="*60)
    print("轴承故障诊断参数优化流程（稳定版 - 复赛数据集10分类）")
    print("="*60)
    
    # 检查数据路径
    if not os.path.exists(TRAIN_ROOT):
        print(f"错误: 训练集路径不存在 {TRAIN_ROOT}")
        return None
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/home/zxy_2024/FUSAI_Bearing_Fault_Diagnosis/results/optimization_cv_{timestamp}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"结果保存目录: {save_dir}")
    
    # 创建优化器
    optimizer = BearingOptimizer(device, save_dir)
    
    # 执行优化
    best_params, best_score, study = optimizer.optimize(n_trials=N_TRIALS)
    
    # 保存最佳参数
    with open(f"{save_dir}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # 保存所有试验的详细结果
    with open(f"{save_dir}/all_trial_results.json", 'w') as f:
        json.dump(optimizer.all_trial_results, f, indent=2)
    
    # 保存Optuna study
    with open(f"{save_dir}/study.pkl", 'wb') as f:
        pickle.dump(study, f)
    
    # 生成试验结果汇总
    summary = {
        'best_trial': {
            'mean_acc': best_params.get('mean_acc'),
            'std_acc': best_params.get('std_acc'),
            'best_acc': best_params.get('best_acc'),
            'params': {k: v for k, v in best_params.items() 
                      if k not in ['best_acc', 'mean_acc', 'std_acc']}
        },
        'optimization_config': {
            'n_trials': N_TRIALS,
            'n_folds': N_FOLDS,
            'train_epochs': TRAIN_EPOCHS,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'num_classes': 10
        },
        'all_trials_summary': []
    }
    
    for result in optimizer.all_trial_results:
        summary['all_trials_summary'].append({
            'trial_number': result['trial_number'],
            'mean_acc': result['mean_acc'],
            'std_acc': result['std_acc'],
            'best_acc': result['best_acc'],
            'n_folds': result['n_folds']
        })
    
    with open(f"{save_dir}/optimization_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 优化结果已保存到: {save_dir}")
    print(f"  模型权重:")
    print(f"    - best_cv_model.pth: 最佳折模型")
    print(f"    - best_cv_5fold_models/: 所有5折模型目录")
    print(f"      * fold_1_acc*.pth ~ fold_5_acc*.pth: 5个折的模型权重")
    print(f"      * feature_config.json: 特征参数配置（频谱和包络谱）")
    print(f"  配置文件:")
    print(f"    - best_cv_config.json: 最佳5折交叉验证完整配置")
    print(f"    - best_params.json: 最佳参数")
    print(f"  结果文件:")
    print(f"    - all_trial_results.json: 所有试验详细结果")
    print(f"    - optimization_summary.json: 优化汇总")
    print(f"    - study.pkl: Optuna优化历史")
    
    # ========== 使用最佳5折交叉验证参数生成数据集 ==========
    print(f"\n{'='*60}")
    print("使用最佳5折交叉验证参数生成数据集...")
    print(f"{'='*60}")
    
    cv_params = optimizer.best_mean_params
    spec_bands_cv = [(f1, f2) for f1, f2 in cv_params['spec_bands']]
    env_centers_cv = cv_params['env_centers']
    env_bw_cv = cv_params['env_bw']
    out_len_spec_cv = cv_params['out_len_spec']
    out_len_env_cv = cv_params['out_len_env']
    use_log_cv = cv_params['use_log']
    
    # 提取训练集特征
    print("\n提取训练集特征（5折交叉验证最佳参数）...")
    spec_train_cv, env_train_cv, labels_train_cv = load_train_data_with_params(
        spec_bands_cv, env_centers_cv, env_bw_cv, out_len_spec_cv, out_len_env_cv, use_log_cv
    )
    
    print(f"训练集特征提取完成:")
    print(f"  频谱特征: {spec_train_cv.shape}")
    print(f"  包络特征: {env_train_cv.shape}")
    print(f"  标签: {labels_train_cv.shape}")
    print(f"  各类别样本数:")
    for label_id in range(10):  # 10分类
        count = np.sum(labels_train_cv == label_id)
        print(f"    类别 {label_id} ({ID_TO_LABEL[label_id]}): {count} 个样本")
    
    # 提取测试集特征
    print("\n提取测试集特征（5折交叉验证最佳参数）...")
    spec_test_cv, env_test_cv, test_files_cv = load_test_data_with_params(
        spec_bands_cv, env_centers_cv, env_bw_cv, out_len_spec_cv, out_len_env_cv, use_log_cv
    )
    
    if len(spec_test_cv) > 0:
        print(f"测试集特征提取完成:")
        print(f"  频谱特征: {spec_test_cv.shape}")
        print(f"  包络特征: {env_test_cv.shape}")
        print(f"  文件数: {len(set(test_files_cv))}")
    else:
        print("测试集路径不存在或无数据")
    
    # 保存5折交叉验证数据集
    dataset_cv = {
        'x_train': {
            'spec': spec_train_cv,
            'env': env_train_cv
        },
        'y_train': labels_train_cv,
        'x_test': {
            'spec': spec_test_cv,
            'env': env_test_cv
        },
        'test_files': test_files_cv,
        'label_map': LABEL_MAP,
        'id_to_label': ID_TO_LABEL,
        'best_params': cv_params,
        'best_score': optimizer.best_mean_acc,
        'model_type': 'best_cv_accuracy',
        'metadata': {
            'fs': FS,
            'window_size': WINDOW_SIZE,
            'step_size': STEP_SIZE,
            'timestamp': timestamp,
            'n_train_samples': len(labels_train_cv),
            'n_test_samples': len(spec_test_cv) if len(spec_test_cv) > 0 else 0,
            'optimization_mode': 'cv',
            'n_folds': N_FOLDS,
            'num_classes': 10,
            'mean_acc': cv_params.get('mean_acc'),
            'std_acc': cv_params.get('std_acc'),
            'best_fold_acc': cv_params.get('best_acc'),
            'description': '使用全局最佳5折交叉验证平均准确率参数生成的数据集（10分类）'
        }
    }
    
    dataset_cv_path = f"{save_dir}/dataset_best_cv.pkl"
    with open(dataset_cv_path, 'wb') as f:
        pickle.dump(dataset_cv, f)
    
    print(f"\n💾 5折交叉验证数据集已保存: {dataset_cv_path}")
    
    # 打印数据集结构总结
    print(f"\n{'='*60}")
    print("数据集生成完成总结")
    print(f"{'='*60}")
    print(f"\n5折交叉验证数据集: {dataset_cv_path}")
    print(f"   平均准确率: {optimizer.best_mean_acc:.2f}% (±{cv_params.get('std_acc', 0):.2f}%)")
    print(f"   最佳折准确率: {cv_params.get('best_acc', 0):.2f}%")
    print(f"   训练集: {spec_train_cv.shape[0]} 样本")
    if len(spec_test_cv) > 0:
        print(f"   测试集: {spec_test_cv.shape[0]} 样本")
    
    print(f"\n{'='*60}")
    print("✅ 优化流程完成！")
    print(f"{'='*60}")
    print(f"\n输出文件:")
    print(f"  模型权重:")
    print(f"    1. best_cv_model.pth - 最佳折模型 ({optimizer.best_mean_acc:.2f}%)")
    print(f"    2. best_cv_5fold_models/ - 所有5折模型目录")
    print(f"       - fold_1_acc*.pth ~ fold_5_acc*.pth: 5个折的模型权重")
    print(f"       - feature_config.json: 特征参数配置（频谱和包络谱参数）")
    print(f"  模型配置:")
    print(f"    3. best_cv_config.json - 最佳5折交叉验证完整配置（包含所有特征参数）")
    print(f"  数据集:")
    print(f"    4. dataset_best_cv.pkl - 5折交叉验证最佳参数数据集")
    print(f"  优化结果:")
    print(f"    5. best_params.json - 最佳参数")
    print(f"    6. all_trial_results.json - 所有试验详细结果")
    print(f"    7. optimization_summary.json - 优化汇总")
    print(f"    8. study.pkl - Optuna优化历史")
    print(f"\n最佳结果:")
    print(f"  全局最佳平均准确率: {optimizer.best_mean_acc:.2f}%")
    print(f"  优化目标准确率: {best_score:.2f}%")
    print(f"  准确率标准差: {best_params.get('std_acc', 0):.2f}%")
    print(f"  交叉验证折数: {N_FOLDS}")
    print(f"  分类类别数: 10")
    
    return best_params, best_score, save_dir, optimizer

if __name__ == "__main__":
    result = main()
    if result is not None:
        best_params, best_score, save_dir, optimizer = result
        print(f"\n{'='*60}")
        print("后续步骤:")
        print(f"{'='*60}")
        print(f"1. 使用最佳5折交叉验证模型进行预测:")
        print(f"   最佳折模型: {save_dir}/best_cv_model.pth")
        print(f"   所有5折模型: {save_dir}/best_cv_5fold_models/")
        print(f"      - fold_1_acc*.pth ~ fold_5_acc*.pth: 5个折的模型权重")
        print(f"   特征参数配置: {save_dir}/best_cv_5fold_models/feature_config.json")
        print(f"   数据集: {save_dir}/dataset_best_cv.pkl")
        print(f"   平均准确率: {optimizer.best_mean_acc:.2f}% (±{optimizer.best_mean_params.get('std_acc', 0):.2f}%)")
        print(f"   最佳折准确率: {optimizer.best_mean_params.get('best_acc', 0):.2f}%")
        print(f"   各折准确率: {[f'{acc:.2f}%' for acc in optimizer.best_fold_accuracies]}")
        print(f"")
        print(f"2. 查看特征参数配置:")
        print(f"   频谱参数: {save_dir}/best_cv_5fold_models/feature_config.json")
        print(f"   包含: 频段数量、频段范围、输出维度、是否使用对数等")
        print(f"   包络参数: 中心频率、带宽、输出维度等")
        print(f"")
        print(f"3. 查看详细的试验结果:")
        print(f"   所有试验结果: {save_dir}/all_trial_results.json")
        print(f"   优化汇总: {save_dir}/optimization_summary.json")
        print(f"{'='*60}")

