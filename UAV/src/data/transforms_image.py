"""
图像数据增强模块。

封装 torchvision.transforms 或自定义图像增强操作。
支持 PIL Image, Tensor 或 ndarray (需注意格式转换)。
"""

from typing import Tuple, Union

import torch
import torchvision.transforms as T
from PIL import Image

from src.data.transforms import register_augment

__all__ = ["ImageRandomCrop", "ImageRandomHorizontalFlip", "ImageColorJitter"]


@register_augment("image_random_crop")
class ImageRandomCrop(T.RandomCrop):
    """
    随机裁剪图像。
    继承自 torchvision.transforms.RandomCrop。
    """
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        """
        Args:
            size (sequence or int): 裁剪输出尺寸 (h, w)。
            padding (int or sequence, optional): 图像各边的填充。
            pad_if_needed (boolean): 如果图像小于裁剪尺寸，是否进行填充。
            fill: 像素填充值。
            padding_mode: 填充模式。
        """
        super().__init__(size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode)


@register_augment("image_random_horizontal_flip")
class ImageRandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    随机水平翻转图像。
    继承自 torchvision.transforms.RandomHorizontalFlip。
    """
    def __init__(self, p=0.5):
        """
        Args:
            p (float): 翻转概率。
        """
        super().__init__(p=p)


@register_augment("image_color_jitter")
class ImageColorJitter(T.ColorJitter):
    """
    随机调整亮度、对比度、饱和度和色相。
    继承自 torchvision.transforms.ColorJitter。
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Args:
            brightness (float or tuple): 亮度抖动范围。
            contrast (float or tuple): 对比度抖动范围。
            saturation (float or tuple): 饱和度抖动范围。
            hue (float or tuple): 色相抖动范围。
        """
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

# 如果需要支持 numpy array 输入，可以在此添加 Wrapper 类或在 dataset 中统一转为 Tensor/PIL
