from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .model import BiLSTMRegressor, freeze_first_n_bilstm_layers
from .train import save_checkpoint, select_device


class _SubsetDataset(Dataset):
    def __init__(self, base: Dataset, indices: List[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


def _save_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.no_grad()
def _window_mse_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    在给定 loader 上计算“窗口级别”的 MSE（每个样本一个误差）。
    返回 shape=[N] 的 numpy 数组。
    """
    model.eval()
    errs: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        p = model(x)
        e = torch.mean((p - y) ** 2, dim=1)  # [B]
        errs.append(e.detach().cpu().numpy().astype(np.float32))
    if not errs:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(errs, axis=0)


@torch.no_grad()
def _abs_error_stats_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    """
    统计目标域上的 |e| 分布（逐通道）。

    说明（中文注释）：
    - 这里的监督信号来自回归目标 y（例如执行器真实 PWM），并不是故障类别标签
    - 因此即使目标域“无故障标签”，也能计算误差分布用于阈值选择/稳定微调
    """
    model.eval()
    abs_errs: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        p = model(x)
        e = torch.abs(y - p)  # |e|，更贴合阈值检测 (8)
        abs_errs.append(e.detach().cpu().numpy().astype(np.float32))
    if not abs_errs:
        return {"n": 0, "mu": [], "sigma": []}
    ae = np.concatenate(abs_errs, axis=0)  # [N,C]
    mu = np.mean(ae, axis=0)
    sigma = np.std(ae, axis=0)
    return {
        "n": int(ae.shape[0]),
        "mu": mu.astype(np.float32).tolist(),
        "sigma": sigma.astype(np.float32).tolist(),
        "p50": np.quantile(ae, 0.50, axis=0).astype(np.float32).tolist(),
        "p90": np.quantile(ae, 0.90, axis=0).astype(np.float32).tolist(),
        "p99": np.quantile(ae, 0.99, axis=0).astype(np.float32).tolist(),
    }


def _pseudo_normal_keep_indices(
    window_mse: np.ndarray,
    keep_low_error_quantile: float,
) -> Tuple[List[int], Dict[str, object]]:
    """
    pseudo-normal：
    - 按窗口 MSE 从小到大排序
    - 取最小的 keep_low_error_quantile 比例窗口作为“更接近正常”的训练子集
    """
    e = np.asarray(window_mse, dtype=float).reshape(-1)
    n = int(e.shape[0])
    if n == 0:
        return [], {"n": 0}

    keep_ratio = float(keep_low_error_quantile)
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_low_error_quantile 必须在 (0,1] 内")

    order = np.argsort(e)  # 小误差在前
    keep_n = max(1, int(np.ceil(n * keep_ratio)))
    keep = order[:keep_n].astype(int).tolist()

    dist = {
        "n": n,
        "keep_ratio": keep_ratio,
        "keep_n": keep_n,
        "mse_mean": float(np.mean(e)),
        "mse_std": float(np.std(e)),
        "mse_p50": float(np.quantile(e, 0.50)),
        "mse_p90": float(np.quantile(e, 0.90)),
        "mse_p95": float(np.quantile(e, 0.95)),
        "mse_p99": float(np.quantile(e, 0.99)),
        "keep_threshold": float(e[order[keep_n - 1]]),
    }
    return keep, dist


def adapt_source_to_target(
    model: BiLSTMRegressor,
    source_model_ckpt: str,
    target_real_dataloader: DataLoader,
    cfg: ExperimentConfig,
    out_dir: str,
    *,
    target_tag: str = "target_real",
    source_tag: Optional[str] = None,
) -> Dict[str, object]:
    """
    无标签“故障类别”下的迁移微调（source -> target）。

    为什么 target 可以在“无故障标签”下也能微调？
    - 这里训练目标不是“故障分类标签”，而是连续回归输出 Y（例如执行器真实 PWM/输出量）。
    - 即使没有 fault label，我们仍然有 (X -> Y) 的监督信号，因此可以用 MSE 做微调。

    输入：
    - source_model_ckpt：源域训练好的 checkpoint（包含 model_state）
    - target_real_dataloader：目标域 DataLoader（每个 batch 仍是 (x, y) 回归标签）
    - cfg.transfer：控制冻结层数、学习率、pseudo-normal 等
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    metric_dir = out / "metrics"

    device = select_device(cfg.train.device)
    model.to(device)

    payload = torch.load(source_model_ckpt, map_location="cpu")
    if "model_state" not in payload:
        raise KeyError(f"checkpoint 缺少 key=model_state：{source_model_ckpt}")
    model.load_state_dict(payload["model_state"], strict=True)

    # 1) 冻结前 N 层 BiLSTM：只训练最后一层 BiLSTM + MLP head（符合你的描述）
    freeze_n = cfg.transfer.freeze_bilstm_layers
    if freeze_n is None:
        freeze_n = int(cfg.transfer.freeze_first_n_bilstm)
    freeze_first_n_bilstm_layers(model, int(freeze_n))

    # 2) pseudo-normal：先用当前模型在 target 上推理并计算窗口 MSE
    window_mse_before = _window_mse_on_loader(model, target_real_dataloader, device)
    pseudo_info: Dict[str, object] = {"enabled": bool(cfg.transfer.pseudo_normal_enable)}
    abs_error_before_stats = _abs_error_stats_on_loader(model, target_real_dataloader, device)

    finetune_loader = target_real_dataloader
    if bool(cfg.transfer.pseudo_normal_enable):
        keep_ratio = cfg.transfer.keep_low_error_quantile
        if keep_ratio is None:
            # 兼容旧字段名（原来是 quantile 的阈值概念，这里按“保留比例”解释）
            keep_ratio = float(cfg.transfer.pseudo_keep_quantile)
        keep_idx, dist = _pseudo_normal_keep_indices(window_mse_before, float(keep_ratio))
        pseudo_info["before_dist"] = dist
        pseudo_info["kept_indices"] = {"n": len(keep_idx)}

        if len(keep_idx) == 0:
            raise ValueError("pseudo-normal 筛选后无可用窗口样本；请调大 keep_low_error_quantile")

        finetune_ds = _SubsetDataset(target_real_dataloader.dataset, keep_idx)
        finetune_loader = DataLoader(
            finetune_ds,
            batch_size=target_real_dataloader.batch_size,
            shuffle=True,
            num_workers=target_real_dataloader.num_workers,
            pin_memory=target_real_dataloader.pin_memory,
            drop_last=False,
        )

    # 3) 微调训练
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.transfer.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    criterion = nn.MSELoss()

    src_name = source_tag or Path(source_model_ckpt).stem
    dst_name = f"{src_name}_to_{target_tag}"
    best_path = ckpt_dir / f"adapt_{dst_name}_best.pt"
    last_path = ckpt_dir / f"adapt_{dst_name}_last.pt"

    best_mse = float("inf")
    for epoch in range(1, int(cfg.transfer.epochs) + 1):
        model.train()
        losses: List[float] = []
        for x, y in finetune_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            p = model(x)
            loss = criterion(p, y)
            loss.backward()
            if float(cfg.train.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        # 目标域上用“窗口级 MSE 均值”做 best 选择（因为没有单独 val，就用 target 本身）
        window_mse_now = _window_mse_on_loader(model, target_real_dataloader, device)
        mean_mse = float(np.mean(window_mse_now)) if window_mse_now.size else float("inf")

        save_checkpoint(last_path, model, optimizer, epoch, best_mse, extra={"mean_train_loss": float(np.mean(losses)) if losses else None})
        if mean_mse < best_mse:
            best_mse = mean_mse
            save_checkpoint(best_path, model, optimizer, epoch, best_mse, extra={"mean_train_loss": float(np.mean(losses)) if losses else None})

    # 4) 记录 pseudo-normal 误差分布（供阈值选择）
    window_mse_after = _window_mse_on_loader(model, target_real_dataloader, device)
    abs_error_after_stats = _abs_error_stats_on_loader(model, target_real_dataloader, device)
    dist_after = {
        "n": int(window_mse_after.size),
        "mse_mean": float(np.mean(window_mse_after)) if window_mse_after.size else float("nan"),
        "mse_std": float(np.std(window_mse_after)) if window_mse_after.size else float("nan"),
        "mse_p50": float(np.quantile(window_mse_after, 0.50)) if window_mse_after.size else float("nan"),
        "mse_p90": float(np.quantile(window_mse_after, 0.90)) if window_mse_after.size else float("nan"),
        "mse_p95": float(np.quantile(window_mse_after, 0.95)) if window_mse_after.size else float("nan"),
        "mse_p99": float(np.quantile(window_mse_after, 0.99)) if window_mse_after.size else float("nan"),
    }
    pseudo_info["after_dist"] = dist_after

    _save_json(
        metric_dir / f"pseudo_normal_mse_{dst_name}.json",
        {
            "source_model_ckpt": str(Path(source_model_ckpt).resolve()),
            "freeze_bilstm_layers": int(freeze_n),
            "pseudo_normal": pseudo_info,
            "abs_error_before_stats": abs_error_before_stats,
            "abs_error_after_stats": abs_error_after_stats,
        },
    )

    return {
        "adapt_best_checkpoint": str(best_path),
        "adapt_last_checkpoint": str(last_path),
        "adapt_best_target_mse": float(best_mse),
        "pseudo_normal_report": str((metric_dir / f"pseudo_normal_mse_{dst_name}.json").resolve()),
    }


# 保留旧接口，避免你之前的 run.py 断掉（内部复用新逻辑）
def finetune_transfer(
    model: BiLSTMRegressor,
    pretrained_ckpt: str,
    train_loader: DataLoader,
    val_loader: DataLoader,  # 旧接口保留，但这里不再强依赖
    cfg: ExperimentConfig,
    out_dir: str,
    extra_ckpt: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    兼容旧接口：用 train_loader 作为 target_real_dataloader 进行适配。
    """
    info = adapt_source_to_target(
        model=model,
        source_model_ckpt=pretrained_ckpt,
        target_real_dataloader=train_loader,
        cfg=cfg,
        out_dir=out_dir,
        target_tag="target",
    )
    return {
        "transfer_best_checkpoint": info["adapt_best_checkpoint"],
        "transfer_last_checkpoint": info["adapt_last_checkpoint"],
        "transfer_best_val_mse": info["adapt_best_target_mse"],
        "transfer_pseudo_report": info["pseudo_normal_report"],
    }
