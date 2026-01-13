"""
手工特征提取器模块。

包含基于统计学和信号处理（如 STFT）的特征提取方法。
"""

from typing import Optional, Union

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F

from src.features.registry import register_feature_extractor

__all__ = ["StatisticalFeatureExtractor", "STFTFeatureExtractor"]


@register_feature_extractor("statistical")
class StatisticalFeatureExtractor:
    """
    统计特征提取器。
    
    计算时域信号的统计指标：
    - 均值 (Mean)
    - 方差 (Variance)
    - 均方根 (RMS)
    - 峭度 (Kurtosis)
    - 偏度 (Skewness)
    - 峰峰值 (Peak-to-Peak)
    - 峰值因子 (Crest Factor)
    - 脉冲因子 (Impulse Factor)
    - 波形因子 (Shape Factor)
    - 裕度因子 (Clearance Factor)
    
    输入: 1D numpy array 或 Tensor (L,) 或 (C, L)
    输出: 1D Tensor (num_features,) 或 (C * num_features,)
    """

    def __init__(self, axis: int = -1):
        """
        Args:
            axis (int): 计算统计量的维度，默认为最后一个维度。
        """
        self.axis = axis

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: 输入信号。

        Returns:
            特征向量 Tensor。
        """
        # 统一转为 numpy 处理 (scipy.stats 方便)
        if isinstance(x, torch.Tensor):
            data = x.detach().cpu().numpy()
        else:
            data = x

        # 确保数据是浮点型
        data = data.astype(np.float32)

        # 基础统计量
        mean = np.mean(data, axis=self.axis)
        std = np.std(data, axis=self.axis)
        var = np.var(data, axis=self.axis)
        rms = np.sqrt(np.mean(data**2, axis=self.axis))
        peak = np.max(np.abs(data), axis=self.axis)
        p2p = np.ptp(data, axis=self.axis)  # max - min

        # 高阶统计量
        skew = scipy.stats.skew(data, axis=self.axis)
        kurt = scipy.stats.kurtosis(data, axis=self.axis)

        # 因子指标 (加 eps 防止除零)
        eps = 1e-8
        crest_factor = peak / (rms + eps)
        impulse_factor = peak / (np.abs(mean) + eps) # 通常分母是绝对平均值 mean(abs(x))
        # 修正: Impulse Factor 分母通常是 mean(abs(x))
        abs_mean = np.mean(np.abs(data), axis=self.axis)
        impulse_factor = peak / (abs_mean + eps)
        shape_factor = rms / (abs_mean + eps)
        
        # Clearance Factor: peak / (mean(sqrt(abs(x)))^2)
        sqrt_abs_mean = np.mean(np.sqrt(np.abs(data)), axis=self.axis)**2
        clearance_factor = peak / (sqrt_abs_mean + eps)

        # 拼接特征
        features = [
            mean, var, rms, peak, p2p, skew, kurt,
            crest_factor, impulse_factor, shape_factor, clearance_factor
        ]
        
        # 堆叠 -> (num_features, ...)
        features_arr = np.stack(features, axis=-1)
        
        # 展平为 1D 向量
        return torch.from_numpy(features_arr.flatten()).float()


@register_feature_extractor("stft")
class STFTFeatureExtractor:
    """
    STFT (短时傅里叶变换) 特征提取器。

    计算信号的 STFT，并对频率和时间维度进行池化，输出固定长度向量。
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        output_size: int = 64,
        use_magnitude: bool = True
    ):
        """
        Args:
            n_fft (int): FFT 窗口大小。
            hop_length (int): 帧移。
            win_length (int): 窗口长度。
            output_size (int): 最终输出特征向量的长度（通过自适应池化实现）。
            use_magnitude (bool): 是否使用幅值谱（模长）。如果不使用，则保留复数实部虚部（需自行处理）。
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.output_size = output_size
        self.use_magnitude = use_magnitude

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: 输入信号 (L,) 或 (C, L)。

        Returns:
            特征向量 Tensor (output_size,)。
        """
        # 统一转为 Torch Tensor
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float()

        # 处理维度，确保是 (..., L)
        # torch.stft 输入需要是 1D (L) 或 2D (B, L)。这里假设单样本输入。
        # 如果输入是 (C, L)，可以视为 batch 处理，也可以展平。
        # 这里简化处理：如果是 (C, L)，先在该维度平均或者分别提取。
        # 假设输入是单通道 (L,)，如果是 (C, L) 则对每个通道提取后拼接或平均。
        
        if x_tensor.ndim == 1:
            # (L,) -> (1, L) 用于 stft
            x_tensor = x_tensor.unsqueeze(0)
        
        # STFT return: (B, n_fft/2 + 1, T, 2) if return_complex=False (old) 
        # or complex tensor (new). 
        # 为了兼容性，建议使用 return_complex=True 并取 abs
        
        stft_res = torch.stft(
            x_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            window=torch.hann_window(self.n_fft).to(x_tensor.device)
        )
        # stft_res shape: (B, F, T) complex
        
        if self.use_magnitude:
            features = torch.abs(stft_res) # (B, F, T)
        else:
            # 仅取实部作为示例
            features = stft_res.real

        # 池化到固定大小
        # 将 (B, F, T) 视为图像特征图，展平或进行 Global Pooling
        # 这里使用 AdaptiveAvgPool1d 将 (B, F*T) 降维到 (B, output_size)
        # 或者先对 F, T 维度 pool
        
        batch_size = features.shape[0]
        flattened = features.view(batch_size, -1) # (B, F*T)
        
        # 使用线性层或池化层降维。这里用 AdaptiveAvgPool1d 简单缩放
        # input: (B, C_in, L_in) -> AdaptiveAvgPool1d -> (B, C_in, L_out)
        # 这里我们将 flattened 视为 (B, 1, F*T)
        
        pooled = F.adaptive_avg_pool1d(flattened.unsqueeze(1), self.output_size)
        # shape: (B, 1, output_size)
        
        output = pooled.squeeze(1) # (B, output_size)
        
        # 如果原始输入是 (L,) -> B=1 -> (output_size,)
        # 如果原始输入是 (C, L) -> B=C -> (C, output_size) -> flatten -> (C*output_size,)
        
        return output.flatten()
