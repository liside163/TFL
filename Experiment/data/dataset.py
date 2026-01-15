"""
PyTorch数据集包装器
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class HILDataset(Dataset):
    """HIL数据集PyTorch包装器"""

    def __init__(self, windows, labels):
        """
        参数:
            windows: np.ndarray, [n_samples, window_size, features]
            labels: np.ndarray, [n_samples]
        """
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32, num_workers=2):
    """
    创建训练和测试数据加载器

    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        batch_size: 批次大小
        num_workers: 数据加载线程数

    返回:
        train_loader, test_loader
    """
    train_dataset = HILDataset(X_train, y_train)
    test_dataset = HILDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
