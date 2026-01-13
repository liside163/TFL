from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig


def select_device(prefer: str) -> torch.device:
    """
    根据配置选择 device；若 cuda 不可用则自动回退到 cpu。
    """
    prefer = (prefer or "cpu").lower()
    if prefer.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")

        # 关键点（中文注释）：
        # - torch.cuda.is_available() 只能说明“能看到 GPU”，不代表当前 PyTorch wheel 编译了该 GPU 架构的 kernel
        # - 类似 sm_120（新架构）在旧版 PyTorch 上会触发：
        #   RuntimeError: no kernel image is available for execution on the device
        # - 因此这里做一次“架构兼容性”检查，不兼容则自动回退到 CPU，避免训练直接崩溃
        try:
            cap = torch.cuda.get_device_capability(0)  # (major, minor)
            arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()

            def _parse_sm(s: str) -> tuple[int, int] | None:
                if not s.startswith("sm_"):
                    return None
                tail = s[3:]
                digits = "".join(ch for ch in tail if ch.isdigit())
                if len(digits) < 2:
                    return None
                major = int(digits[:-1])
                minor = int(digits[-1])
                return major, minor

            supported = {t for t in (_parse_sm(a) for a in arch_list) if t is not None}
            if supported and tuple(cap) not in supported:
                # 保守策略：如果能拿到 arch_list 且不包含当前 cap，则认为不兼容，回退 CPU
                return torch.device("cpu")
        except Exception:
            # 拿不到 capability/arch_list 时不拦截，交给后续 .to('cuda') 自己报错
            pass

        # 支持 "cuda:0" 等写法
        return torch.device(prefer if ":" in prefer else "cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    for x, y in loader:
        # x: [B, L, F]，y: [B, C]
        x = x.to(device)
        y = y.to(device)
        p = model(x)  # p: [B, C]
        # 关键维度变化：模型将序列输入 [B, L, F] 映射为回归输出 [B, C]
        ys.append(y.detach().cpu().numpy())
        ps.append(p.detach().cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    err = y_pred - y_true
    mse = (err**2).mean(axis=0)
    mae = np.abs(err).mean(axis=0)
    return {
        "mse_per_channel": mse.tolist(),
        "mae_per_channel": mae.tolist(),
        "mse": float(mse.mean()),
        "mae": float(mae.mean()),
    }


def _save_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, str(path))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    out_dir: str,
    extra_ckpt: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    标准训练循环 + best checkpoint。

    关键点（中文注释）：
    - 每个 epoch 在验证集上评估并保存 best
    - 记录误差分布（按通道绝对误差的均值/方差/分位数）
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = select_device(cfg.train.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_path = out / "checkpoints" / "best.pt"
    last_path = out / "checkpoints" / "last.pt"

    history: List[Dict[str, float]] = []
    patience = int(cfg.train.early_stop_patience)
    bad_epochs = 0

    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        losses: List[float] = []
        for x, y in train_loader:
            # x: [B, L, F]，y: [B, C]
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            p = model(x)  # p: [B, C]
            # 关键维度变化：预测输出与标签在通道维度 C 上对齐
            loss = criterion(p, y)
            loss.backward()
            if float(cfg.train.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        yv, pv = predict(model, val_loader, device)
        val_metrics = compute_metrics(yv, pv)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss = float(val_metrics["mse"])
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        # 保存 last
        save_checkpoint(last_path, model, optimizer, epoch, best_val, extra_ckpt)

        # 误差分布统计（验证集）
        abs_err = np.abs(pv - yv)  # [N,C]
        dist = {
            "abs_err_mean": abs_err.mean(axis=0).tolist(),
            "abs_err_std": abs_err.std(axis=0).tolist(),
            "abs_err_p50": np.quantile(abs_err, 0.50, axis=0).tolist(),
            "abs_err_p90": np.quantile(abs_err, 0.90, axis=0).tolist(),
            "abs_err_p99": np.quantile(abs_err, 0.99, axis=0).tolist(),
        }
        _save_json(out / "metrics" / f"val_error_dist_epoch{epoch:03d}.json", dist)

        # 保存 best
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, best_val, extra_ckpt)
        else:
            bad_epochs += 1

        if patience > 0 and bad_epochs >= patience:
            break

    # 训练结束：导出 history
    _save_json(out / "metrics" / "history.json", {"history": history, "best_val": best_val})
    return {
        "device": str(device),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_val_mse": float(best_val),
        "history": history,
    }
