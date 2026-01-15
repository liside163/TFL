"""
随机种子设置 - 确保实验可复现
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    设置所有随机种子以确保可复现性

    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置: {seed}")
