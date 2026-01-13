"""
信号数据增强模块。

包含针对 1D 信号 (numpy array 或 Tensor) 的增强操作。
"""

from typing import Tuple, Union

import numpy as np
import torch

from src.data.transforms import register_augment

__all__ = ["SignalRandomCrop", "SignalGaussianNoise", "SignalAmplitudeScale"]


@register_augment("signal_random_crop")
class SignalRandomCrop:
    """
    随机裁剪固定长度窗口。
    如果信号长度小于窗口大小，则进行填充（零填充）。
    """

    def __init__(self, window_size: int):
        """
        Args:
            window_size (int): 裁剪窗口长度。
        """
        self.window_size = window_size

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: 输入信号，形状为 (L,) 或 (C, L)。

        Returns:
            裁剪后的信号，形状为 (window_size,) 或 (C, window_size)。
        """
        # 获取信号长度 (假设最后一个维度是时间/长度维度)
        length = x.shape[-1]
        
        if length < self.window_size:
            # 需要填充
            pad_len = self.window_size - length
            if isinstance(x, np.ndarray):
                # numpy padding
                pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad_len)]
                return np.pad(x, pad_width, mode='constant', constant_values=0)
            elif isinstance(x, torch.Tensor):
                # torch padding (pad start, pad end) -> (0, pad_len)
                # F.pad 对最后一个维度操作
                import torch.nn.functional as F
                # F.pad 需要根据维度调整，这里简化为对最后维度 pad
                # 对于 1D (L,) -> pad=(0, pad_len)
                # 对于 2D (C, L) -> pad=(0, pad_len)
                return F.pad(x, (0, pad_len), "constant", 0)
        
        elif length > self.window_size:
            # 需要裁剪
            start = np.random.randint(0, length - self.window_size + 1)
            return x[..., start : start + self.window_size]
        
        return x


@register_augment("signal_gaussian_noise")
class SignalGaussianNoise:
    """
    在信号上叠加零均值高斯噪声。
    """

    def __init__(self, std: float = 0.01):
        """
        Args:
            std (float): 噪声的标准差。
        """
        self.std = std

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: 输入信号。

        Returns:
            叠加噪声后的信号。
        """
        if self.std <= 0:
            return x

        if isinstance(x, np.ndarray):
            noise = np.random.normal(loc=0.0, scale=self.std, size=x.shape).astype(x.dtype)
            return x + noise
        elif isinstance(x, torch.Tensor):
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


@register_augment("signal_amplitude_scale")
class SignalAmplitudeScale:
    """
    随机缩放信号幅值。
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Args:
            scale_range (tuple): 缩放因子的范围 (min, max)。
        """
        self.scale_range = scale_range

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: 输入信号。

        Returns:
            缩放后的信号。
        """
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale
