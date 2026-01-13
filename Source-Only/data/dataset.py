# =====================================================================
# PyTorch Dataset 实现
# 功能: 加载UAV故障诊断数据，支持按域和飞行状态筛选
# =====================================================================

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple, Union
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    DATA_ROOT, 
    SEQUENCE_LENGTH, 
    PAD_VALUE,
    FAULT_LABEL_TO_IDX,
    NUM_FEATURES,
    TRAIN_CONFIG
)
from data.preprocess import (
    parse_filename, 
    filter_files_by_condition, 
    get_feature_columns,
    FeatureScaler,
    print_dataset_statistics
)


class UAVDataset(Dataset):
    """
    无人机故障诊断数据集
    
    维度变换说明:
    - 原始CSV: [seq_len_i, all_features] 每个样本长度不同
    - 特征选择后: [seq_len_i, 30] 选择30个共有特征
    - 序列处理后: [SEQUENCE_LENGTH, 30] 统一长度为1000
    - 输出: data=[1000, 30], label=int
    """
    
    def __init__(
        self,
        flight_state: str,
        domain: str,
        data_dir: Optional[Path] = None,
        scaler: Optional[FeatureScaler] = None,
        fit_scaler: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        初始化数据集
        
        Args:
            flight_state: 飞行状态 ("hover", "waypoint", "velocity", etc.)
            domain: 数据域 ("HIL" 或 "Real")
            data_dir: 数据目录，默认使用配置中的DATA_ROOT
            scaler: 特征标准化器，如果为None则不进行标准化
            fit_scaler: 是否使用当前数据拟合标准化器
            max_samples: 最大样本数，用于调试
        """
        self.flight_state = flight_state
        self.domain = domain
        self.data_dir = data_dir if data_dir else DATA_ROOT
        self.scaler = scaler
        self.seq_length = SEQUENCE_LENGTH
        
        # 筛选符合条件的文件
        self.files = filter_files_by_condition(
            self.data_dir, 
            domain=domain, 
            flight_state=flight_state
        )
        
        if max_samples is not None:
            self.files = self.files[:max_samples]
        
        if len(self.files) == 0:
            raise ValueError(
                f"未找到符合条件的数据文件: domain={domain}, flight_state={flight_state}, "
                f"data_dir={self.data_dir}"
            )
        
        print(f"加载数据: {domain} - {flight_state}, 共 {len(self.files)} 个样本")
        
        # 加载所有数据到内存 (根据数据集大小可选择延迟加载)
        self.data, self.labels, self.file_infos = self._load_all_data()
        
        # 标准化处理
        if fit_scaler and self.scaler is not None:
            print("使用当前数据拟合标准化器...")
            self.scaler.fit(self.data)
        
        if self.scaler is not None and self.scaler.is_fitted:
            print("应用特征标准化...")
            self.data = self.scaler.transform(self.data)
    
    def _load_all_data(self) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        加载所有数据文件
        
        Returns:
            data: [N, seq_length, features] 形状的数据数组
            labels: [N] 形状的标签数组
            file_infos: 每个样本的文件信息列表
        """
        all_data = []
        all_labels = []
        all_infos = []
        
        for file_path in self.files:
            # 解析文件名获取元信息
            info = parse_filename(file_path.name)
            
            # 读取CSV
            df = pd.read_csv(file_path)
            
            # 获取特征列
            feature_cols = get_feature_columns(df, self.domain)
            
            if len(feature_cols) < NUM_FEATURES:
                print(f"警告: 文件 {file_path.name} 只有 {len(feature_cols)} 个特征，跳过")
                continue
            
            # 提取特征数据
            # 维度: [原始seq_len, 30]
            features = df[feature_cols].values.astype(np.float32)
            
            # 处理NaN值
            features = np.nan_to_num(features, nan=0.0)
            
            # 序列长度处理: 截断或填充到固定长度
            # 维度变换: [原始seq_len, 30] -> [SEQUENCE_LENGTH, 30]
            processed = self._process_sequence(features)
            
            # 获取标签
            label = FAULT_LABEL_TO_IDX[info["fault_type"]]
            
            all_data.append(processed)
            all_labels.append(label)
            all_infos.append(info)
        
        # 转换为numpy数组
        # 最终维度: data=[N, 1000, 30], labels=[N]
        data = np.stack(all_data, axis=0)
        labels = np.array(all_labels, dtype=np.int64)
        
        print(f"数据加载完成: 形状 {data.shape}, 标签形状 {labels.shape}")
        return data, labels, all_infos
    
    def _process_sequence(self, seq: np.ndarray) -> np.ndarray:
        """
        处理序列长度，截断或填充到固定长度
        
        维度变换:
        输入: [原始seq_len, 30]
        输出: [SEQUENCE_LENGTH, 30] = [1000, 30]
        
        Args:
            seq: 原始序列数据
        
        Returns:
            处理后的固定长度序列
        """
        current_len = seq.shape[0]
        target_len = self.seq_length
        
        if current_len >= target_len:
            # 截断: 取前target_len个时间步
            return seq[:target_len]
        else:
            # 填充: 在末尾填充PAD_VALUE
            pad_len = target_len - current_len
            padding = np.full((pad_len, seq.shape[1]), PAD_VALUE, dtype=np.float32)
            return np.concatenate([seq, padding], axis=0)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            data: [seq_length, features] = [1000, 30] 的数据张量
            label: 标量标签张量
        """
        data = torch.from_numpy(self.data[idx])  # [1000, 30]
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # 标量
        return data, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重，用于处理类别不平衡
        
        Returns:
            各类别的权重张量
        """
        from collections import Counter
        label_counts = Counter(self.labels)
        
        total = len(self.labels)
        num_classes = len(FAULT_LABEL_TO_IDX)
        
        weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 1)  # 避免除零
            # 使用逆频率作为权重
            weight = total / (num_classes * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    flight_state: str,
    batch_size: int = None,
    val_split: float = None,
    source_domain: str = "HIL",
    target_domain: str = "Real",
    data_dir: Optional[Path] = None,
    num_workers: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureScaler]:
    """
    创建训练、验证和测试数据加载器
    
    维度说明:
    - DataLoader输出: [batch_size, seq_length, features] = [B, 1000, 30]
    
    Args:
        flight_state: 飞行状态
        batch_size: 批次大小，默认使用配置
        val_split: 验证集比例，默认使用配置
        source_domain: 源域名称
        target_domain: 目标域名称
        data_dir: 数据目录
        num_workers: DataLoader工作进程数
    
    Returns:
        train_loader: 源域训练数据加载器
        val_loader: 源域验证数据加载器
        target_loader: 目标域数据加载器
        scaler: 使用源域数据拟合的标准化器
    """
    if batch_size is None:
        batch_size = TRAIN_CONFIG["batch_size"]
    if val_split is None:
        val_split = TRAIN_CONFIG["val_split"]
    if num_workers is None:
        num_workers = TRAIN_CONFIG["num_workers"]
    
    # 创建标准化器
    scaler = FeatureScaler()
    
    # 加载源域数据 (HIL)
    print(f"\n{'='*50}")
    print(f"加载源域数据: {source_domain} - {flight_state}")
    print(f"{'='*50}")
    
    source_dataset = UAVDataset(
        flight_state=flight_state,
        domain=source_domain,
        data_dir=data_dir,
        scaler=scaler,
        fit_scaler=True  # 使用源域数据拟合标准化器
    )
    
    # 划分训练/验证集
    total_size = len(source_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # 使用随机划分
    torch.manual_seed(TRAIN_CONFIG["random_seed"])
    train_dataset, val_dataset = torch.utils.data.random_split(
        source_dataset, [train_size, val_size]
    )
    
    print(f"源域划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    
    # 加载目标域数据 (Real)
    print(f"\n{'='*50}")
    print(f"加载目标域数据: {target_domain} - {flight_state}")
    print(f"{'='*50}")
    
    target_dataset = UAVDataset(
        flight_state=flight_state,
        domain=target_domain,
        data_dir=data_dir,
        scaler=scaler,  # 使用源域拟合的标准化器
        fit_scaler=False
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=TRAIN_CONFIG["pin_memory"],
        drop_last=True  # 丢弃不完整的最后一个batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TRAIN_CONFIG["pin_memory"]
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TRAIN_CONFIG["pin_memory"]
    )
    
    print(f"\nDataLoader创建完成:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  目标域批次数: {len(target_loader)}")
    print(f"  批次形状: [batch_size={batch_size}, seq_len={SEQUENCE_LENGTH}, features={NUM_FEATURES}]")
    
    return train_loader, val_loader, target_loader, scaler


if __name__ == "__main__":
    # 测试代码
    print("测试UAVDataset...")
    
    try:
        # 测试数据集创建
        dataset = UAVDataset(
            flight_state="hover",
            domain="HIL",
            max_samples=10  # 只加载10个样本用于测试
        )
        
        print(f"\n数据集大小: {len(dataset)}")
        
        # 测试获取样本
        data, label = dataset[0]
        print(f"样本形状: data={data.shape}, label={label}")
        print(f"数据类型: data={data.dtype}, label={label.dtype}")
        
        # 测试类别权重
        weights = dataset.get_class_weights()
        print(f"类别权重: {weights}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
