"""
深度特征提取器模块。

封装预训练的神经网络模型（Backbone）作为特征提取器。
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from src.features.registry import register_feature_extractor

__all__ = ["DeepFeatureExtractor", "build_deep_feature_extractor"]


class DeepFeatureExtractor:
    """
    使用预训练 backbone 提取特征的包装器。

    该类接收一个 PyTorch Module，将其设为 eval 模式，
    并在调用时禁用梯度计算，输出 Backbone 的特征。
    """

    def __init__(self, backbone: nn.Module, device: Optional[str] = None):
        """
        Args:
            backbone (nn.Module): 预训练的骨干网络。
            device (Optional[str]): 运行设备 ('cpu', 'cuda')。如果不指定，保持原样。
        """
        self.backbone = backbone
        if device:
            self.backbone = self.backbone.to(device)
        self.backbone.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征。

        Args:
            x (torch.Tensor): 输入数据。
                - 如果是图像，通常为 (C, H, W) 或 (B, C, H, W)。
                - 如果是信号，通常为 (C, L) 或 (B, C, L)。
                注意：如果输入没有 Batch 维度，函数内部会自动添加并去除。

        Returns:
            torch.Tensor: 特征向量。
        """
        # 检查是否需要添加 batch 维度
        is_batched = x.ndim > 3 if x.ndim >= 3 else x.ndim > 2 # 简单启发式，需根据具体模型调整
        # 更稳健的方式：假设 dataset 输出单个样本 (C, ...)，需要 unsqueeze(0)
        # 如果是在 DataLoader 之后调用，通常已经是 Batched。
        # 但作为 FeatureExtractor，通常在 Dataset.__getitem__ 中调用，即单样本。
        
        input_tensor = x
        if self.device:
            input_tensor = input_tensor.to(self.device)

        if input_tensor.ndim == 3: # 图像 (C, H, W) -> (1, C, H, W)
             input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.ndim == 2: # 信号 (C, L) -> (1, C, L)
             input_tensor = input_tensor.unsqueeze(0)
        # 如果是 1D (L,) -> (1, 1, L) ? 视模型而定
        
        features = self.backbone(input_tensor)
        
        # 如果输入是单样本，去除 batch 维度
        if features.shape[0] == 1:
            features = features.squeeze(0)
            
        return features.cpu() # 返回 CPU Tensor 以便后续处理 (如 numpy 转换)


@register_feature_extractor("deep_backbone")
def build_deep_feature_extractor(backbone: nn.Module, device: str = "cpu", **kwargs) -> DeepFeatureExtractor:
    """
    构建深度特征提取器的工厂函数。

    Args:
        backbone (nn.Module): 预训练模型实例。
        device (str): 运行设备。
        **kwargs: 其他参数。

    Returns:
        DeepFeatureExtractor: 包装后的特征提取器。
    
    注意:
        由于 Dataset 初始化时无法直接传递 nn.Module 对象（通常从配置读取），
        这个工厂函数更多用于代码层面的构建，或者配合 Model Registry 使用。
        如果在配置文件中使用，可能需要传递 backbone 的名称而不是实例，
        然后在此函数内部实例化 backbone。
    """
    return DeepFeatureExtractor(backbone, device=device)
