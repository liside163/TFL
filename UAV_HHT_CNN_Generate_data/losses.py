from typing import Optional

import torch
import torch.nn as nn


def _gaussian_kernel_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_mul: float,
    kernel_num: int,
    fix_sigma: Optional[float],
) -> torch.Tensor:
    # 为什么：多核高斯核可以覆盖不同尺度的域偏移，比单核更鲁棒
    z = torch.cat([x, y], dim=0)
    z0 = z.unsqueeze(0)
    z1 = z.unsqueeze(1)
    l2 = ((z0 - z1) ** 2).sum(2)

    if fix_sigma is not None:
        base_sigma = float(fix_sigma)
    else:
        base_sigma = torch.detach(l2).mean().item()
        base_sigma = max(base_sigma, 1e-6)

    bandwidths = [base_sigma * (kernel_mul ** (i - kernel_num // 2)) for i in range(kernel_num)]
    kernels = [torch.exp(-l2 / bw) for bw in bandwidths]
    return sum(kernels)


def mk_mmd(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    # 为什么：MK-MMD 直接最小化源域与目标域特征分布差异，实现无监督域对齐
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError("MK-MMD 输入必须是二维张量 (B, D)")
    if source.size(0) < 2 or target.size(0) < 2:
        return source.new_tensor(0.0)

    kernels = _gaussian_kernel_matrix(source, target, kernel_mul, kernel_num, fix_sigma)
    bs = source.size(0)
    bt = target.size(0)

    k_xx = kernels[:bs, :bs]
    k_yy = kernels[bs : bs + bt, bs : bs + bt]
    k_xy = kernels[:bs, bs : bs + bt]

    # 为什么：使用有偏估计更稳定，训练时不会因为小 batch 波动太大
    mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return torch.clamp(mmd2, min=0.0)


class MKMMDLoss(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma: Optional[float] = None):
        super().__init__()
        self.kernel_mul = float(kernel_mul)
        self.kernel_num = int(kernel_num)
        self.fix_sigma = fix_sigma

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mk_mmd(source, target, self.kernel_mul, self.kernel_num, self.fix_sigma)
