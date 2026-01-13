# -*- coding: utf-8 -*-
"""
==============================================================================
数据加载器模块
==============================================================================
功能：创建PyTorch数据加载器，支持DANN域适应训练
- 源域数据加载器（带标签）
- 目标域数据加载器（用于域适应）
- 联合加载器（同时采样源域和目标域数据）

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, Tuple, Optional, Iterator
import pickle
import yaml


class UAVDataset(Dataset):
    """
    无人机故障数据集类
    
    实现PyTorch Dataset接口，支持源域和目标域数据
    
    Attributes:
        X (np.ndarray): 特征数据，shape = (N, T, F)
        y (np.ndarray): 标签数据，shape = (N,)
        domain_label (int): 域标签（0=源域/HIL, 1=目标域/Real）
    
    维度说明：
    ----------
    输入维度: (N_samples, seq_len, n_features) = (N, 100, 21)
        - N_samples: 样本数量
        - seq_len: 时间序列长度（滑动窗口大小）
        - n_features: 特征维度
    
    输出维度（__getitem__返回）:
        - x: (seq_len, n_features) = (100, 21) → 单个样本
        - y: scalar → 故障类别标签
        - domain: scalar → 域标签 (0 or 1)
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        domain_label: int = 0,
        transform: Optional[callable] = None
    ):
        """
        初始化数据集
        
        Args:
            X: 特征数据，shape = (N, T, F)
            y: 标签数据，shape = (N,)
            domain_label: 域标签，0表示源域(HIL)，1表示目标域(Real)
            transform: 可选的数据增强/变换函数
        """
        # 确保数据类型正确
        self.X = torch.FloatTensor(X)  # (N, T, F)
        self.y = torch.LongTensor(y)   # (N,)
        self.domain_label = domain_label
        self.transform = transform
        
        # 记录数据集信息
        self.n_samples = len(self.X)
        self.seq_len = self.X.shape[1]
        self.n_features = self.X.shape[2]
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            x: 特征张量，shape = (seq_len, n_features) = (100, 21)
            y: 标签张量，shape = () 标量
            domain: 域标签张量，shape = () 标量
        
        维度变化：
            数据集存储: (N, T, F) 
            取出单个: (T, F) = (100, 21)
        """
        x = self.X[idx]  # (T, F) = (100, 21)
        y = self.y[idx]  # scalar
        domain = torch.tensor(self.domain_label, dtype=torch.long)  # scalar
        
        # 应用变换（如数据增强）
        if self.transform is not None:
            x = self.transform(x)
        
        return x, y, domain


class InfiniteDataLoader:
    """
    无限数据加载器
    
    用于目标域数据加载，当迭代完所有数据后自动重新开始
    确保源域和目标域可以同步迭代
    """
    
    def __init__(self, dataloader: DataLoader):
        """
        初始化无限加载器
        
        Args:
            dataloader: PyTorch DataLoader实例
        """
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def __iter__(self) -> Iterator:
        return self
    
    def __next__(self) -> Tuple:
        """获取下一批数据，迭代完成后自动重置"""
        try:
            batch = next(self.iterator)
        except StopIteration:
            # 重置迭代器
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch


class DANNDataLoader:
    """
    DANN联合数据加载器
    
    同时从源域和目标域采样数据，用于域对抗训练
    
    使用方式：
    ---------
    for (x_s, y_s, d_s), (x_t, _, d_t) in dann_loader:
        # x_s: 源域数据, y_s: 源域标签, d_s: 源域域标签(0)
        # x_t: 目标域数据, d_t: 目标域域标签(1)
        # 注意：目标域标签 _ 仅用于评估，训练时不使用
    
    维度说明：
    ---------
    每次迭代返回:
        源域批次: (x_s, y_s, d_s)
            - x_s: (batch_size, seq_len, n_features) = (32, 100, 21)
            - y_s: (batch_size,) = (32,)
            - d_s: (batch_size,) = (32,) 全为0
        
        目标域批次: (x_t, y_t, d_t)
            - x_t: (batch_size, seq_len, n_features) = (32, 100, 21)
            - y_t: (batch_size,) = (32,) [仅评估用]
            - d_t: (batch_size,) = (32,) 全为1
    """
    
    def __init__(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader
    ):
        """
        初始化DANN数据加载器
        
        Args:
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器
        """
        self.source_loader = source_loader
        self.target_loader = InfiniteDataLoader(target_loader)
        
        # 迭代次数以源域为准
        self.n_batches = len(source_loader)
    
    def __len__(self) -> int:
        """返回每个epoch的批次数"""
        return self.n_batches
    
    def __iter__(self) -> Iterator:
        """迭代器"""
        target_iter = iter(self.target_loader)
        
        for source_batch in self.source_loader:
            target_batch = next(target_iter)
            yield source_batch, target_batch


def get_dataloaders(
    config_path: str = None,
    config: dict = None,
    data_dict: Dict[str, np.ndarray] = None
) -> Dict[str, DataLoader]:
    """
    创建所有需要的数据加载器
    
    Args:
        config_path: 配置文件路径
        config: 配置字典
        data_dict: 预处理后的数据字典（可选，否则从文件加载）
        
    Returns:
        loaders: 包含以下键的字典
            - 'source_train': 源域训练加载器
            - 'source_val': 源域验证加载器
            - 'target_train': 目标域训练加载器
            - 'target_test': 目标域测试加载器
            - 'dann_train': DANN联合训练加载器
    
    维度变化说明：
    -------------
    DataLoader返回的批次维度:
        输入: 单个样本 (T, F) = (100, 21)
        批量堆叠后: (B, T, F) = (batch_size, 100, 21) = (32, 100, 21)
    """
    # 加载配置 - 使用 config_loader 自动处理环境变量占位符
    if config is None:
        if config_path is None:
            raise ValueError("必须提供config_path或config参数")
        # 导入并使用 config_loader 加载配置（自动处理 ${DATA_ROOT} 等环境变量）
        import sys
        import os as _os
        # 确保可以导入 config_loader
        _script_dir = _os.path.dirname(_os.path.abspath(__file__))
        _parent_dir = _os.path.dirname(_script_dir)
        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
        from config_loader import load_config
        config = load_config(config_path)
    
    # 获取训练参数
    batch_size = config['training']['batch_size']
    
    # 多进程数据加载配置
    # Windows下多进程容易出现内存问题，建议使用0
    # 如果GPU利用率低，瓶颈在于模型太小，不在数据加载
    num_workers = 8  # Windows下设为0避免内存问题
    
    # prefetch_factor和persistent_workers在num_workers=0时不使用
    prefetch_factor = 4
    persistent_workers = True
    
    # 加载预处理数据
    if data_dict is None:
        processed_dir = config['data']['processed_dir']
        data_path = os.path.join(processed_dir, 'processed_data.pkl')
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"[信息] 从 {data_path} 加载预处理数据")
    
    # -------------------- 创建数据集 --------------------
    
    # 源域数据集（域标签=0）
    source_train_dataset = UAVDataset(
        X=data_dict['X_source_train'],
        y=data_dict['y_source_train'],
        domain_label=0  # 源域
    )
    
    source_val_dataset = UAVDataset(
        X=data_dict['X_source_val'],
        y=data_dict['y_source_val'],
        domain_label=0
    )
    
    # 目标域数据集（域标签=1）
    target_train_dataset = UAVDataset(
        X=data_dict['X_target_train'],
        y=data_dict['y_target_train'],
        domain_label=1  # 目标域
    )

    # ==================== 改进：划分目标域验证集 ====================
    # 从目标域测试集中划分一部分作为验证集（用于早停）
    target_val_ratio = config['data'].get('target_val_ratio', 0.2)

    # 合并目标域训练和测试数据用于重新划分
    X_target_all = np.concatenate([data_dict['X_target_train'], data_dict['X_target_test']], axis=0)
    y_target_all = np.concatenate([data_dict['y_target_train'], data_dict['y_target_test']], axis=0)

    # 计算验证集大小
    n_target_total = len(X_target_all)
    n_target_val = int(n_target_total * target_val_ratio)

    # 随机打乱
    indices = np.random.permutation(n_target_total)
    val_indices = indices[:n_target_val]
    test_indices = indices[n_target_val:]

    # 划分目标域验证集和测试集
    target_val_dataset = UAVDataset(
        X=X_target_all[val_indices],
        y=y_target_all[val_indices],
        domain_label=1
    )

    target_test_dataset = UAVDataset(
        X=X_target_all[test_indices],
        y=y_target_all[test_indices],
        domain_label=1
    )
    
    # -------------------- 创建数据加载器 --------------------
    
    # 通用DataLoader参数
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers
    } if num_workers > 0 else {
        'num_workers': 0,
        'pin_memory': True
    }
    
    # 源域训练加载器（打乱顺序）
    source_train_loader = DataLoader(
        source_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # 丢弃不完整的最后一批，保证批大小一致
        **loader_kwargs
    )
    
    # 源域验证加载器（不打乱）
    source_val_loader = DataLoader(
        source_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )

    # ==================== 改进：目标域验证加载器 ====================
    target_val_loader = DataLoader(
        target_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )

    # 目标域训练加载器
    target_train_loader = DataLoader(
        target_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )
    
    # 目标域测试加载器
    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )
    
    # DANN联合训练加载器
    dann_train_loader = DANNDataLoader(
        source_loader=source_train_loader,
        target_loader=target_train_loader
    )
    
    # -------------------- 输出数据加载器信息 --------------------
    print("\n" + "=" * 60)
    print("数据加载器创建完成")
    print("=" * 60)
    print(f"批次大小 (batch_size): {batch_size}")
    print(f"\n源域训练集: {len(source_train_dataset)} 样本, {len(source_train_loader)} 批次")
    print(f"源域验证集: {len(source_val_dataset)} 样本, {len(source_val_loader)} 批次")
    print(f"目标域训练集: {len(target_train_dataset)} 样本, {len(target_train_loader)} 批次")
    print(f"目标域验证集: {len(target_val_dataset)} 样本, {len(target_val_loader)} 批次 [改进]")
    print(f"目标域测试集: {len(target_test_dataset)} 样本, {len(target_test_loader)} 批次")
    print(f"\nDANN联合加载器: 每epoch {len(dann_train_loader)} 批次")
    
    # 显示单个批次的维度
    sample_batch = next(iter(source_train_loader))
    x, y, d = sample_batch
    print(f"\n单批次维度验证:")
    print(f"  x (特征): {x.shape} = (batch_size, seq_len, n_features)")
    print(f"  y (标签): {y.shape} = (batch_size,)")
    print(f"  d (域标签): {d.shape} = (batch_size,)")
    
    loaders = {
        'source_train': source_train_loader,
        'source_val': source_val_loader,
        'target_train': target_train_loader,
        'target_val': target_val_loader,      # 改进：添加目标域验证加载器
        'target_test': target_test_loader,
        'dann_train': dann_train_loader,
        'train_source_dataset': source_train_dataset,  # 改进：用于计算类别权重
    }
    
    return loaders


if __name__ == "__main__":
    """
    独立运行此脚本测试数据加载器
    """
    import sys
    
    # 获取配置文件路径
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    
    print("=" * 60)
    print("UAV-DANN 数据加载器模块测试")
    print("=" * 60)
    
    # 创建数据加载器
    loaders = get_dataloaders(config_path=config_path)
    
    # 测试DANN加载器
    print("\n>>> 测试DANN联合加载器...")
    dann_loader = loaders['dann_train']
    
    for i, (source_batch, target_batch) in enumerate(dann_loader):
        if i >= 2:  # 只测试2个批次
            break
        
        x_s, y_s, d_s = source_batch
        x_t, y_t, d_t = target_batch
        
        print(f"\n批次 {i+1}:")
        print(f"  源域 - x: {x_s.shape}, y: {y_s.shape}, d: {d_s.unique()}")
        print(f"  目标域 - x: {x_t.shape}, y: {y_t.shape}, d: {d_t.unique()}")
    
    print("\n测试完成！")
