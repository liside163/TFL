from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _plots_dir(work_dir: str) -> Path:
    p = Path(work_dir) / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _downsample(y: np.ndarray, max_points: int) -> np.ndarray:
    if max_points is None or int(max_points) <= 0:
        return y
    n = int(y.shape[0])
    if n <= int(max_points):
        return y
    idx = np.linspace(0, n - 1, int(max_points), dtype=int)
    return y[idx]


def save_pred_vs_true(
    work_dir: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    channel_names: Optional[Sequence[str]] = None,
    max_points: int = 2000,
) -> str:
    """
    图 1：pred_vs_true（每通道一行）。
    """
    import matplotlib.pyplot as plt

    yt = _downsample(np.asarray(y_true), max_points)
    yp = _downsample(np.asarray(y_pred), max_points)
    c = int(yt.shape[1])
    names = list(channel_names) if channel_names is not None else [f"ch{i}" for i in range(c)]

    fig, axes = plt.subplots(c, 1, figsize=(11, 2.3 * c), sharex=True)
    if c == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(yt[:, i], label="true", linewidth=1)
        ax.plot(yp[:, i], label="pred", linewidth=1)
        ax.set_title(f"pred_vs_true / {names[i]}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()

    out = _plots_dir(work_dir) / "pred_vs_true.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def save_error_and_threshold(
    work_dir: str,
    error_e: np.ndarray,
    threshold_T: np.ndarray,
    *,
    channel_names: Optional[Sequence[str]] = None,
    max_points: int = 2000,
) -> str:
    """
    图 2：error_and_threshold。

    论文公式对应（中文注释）：
    - (6) e = p - yhat
    - (7) T = mu + alpha*sigma
    - (8) 1(|e|>T) 作为故障判据
    """
    import matplotlib.pyplot as plt

    e = _downsample(np.asarray(error_e), max_points)
    c = int(e.shape[1])
    names = list(channel_names) if channel_names is not None else [f"ch{i}" for i in range(c)]

    T = np.asarray(threshold_T)
    if T.ndim == 0:
        # 全局阈值：画聚合误差时更合适，这里用“每通道同一条线”方便直观
        tline = np.full((e.shape[0], c), float(T), dtype=float)
    else:
        t = T.reshape(1, -1)
        tline = np.repeat(t, e.shape[0], axis=0)

    fig, axes = plt.subplots(c, 1, figsize=(11, 2.3 * c), sharex=True)
    if c == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(e[:, i], label="error e", linewidth=1)
        ax.plot(tline[:, i], label="threshold T", linewidth=1)
        ax.set_title(f"error_and_threshold / {names[i]}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()

    out = _plots_dir(work_dir) / "error_and_threshold.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def save_fault_flags(
    work_dir: str,
    fault_flags: np.ndarray,
    *,
    max_points: int = 2000,
) -> str:
    """
    图 3：fault_flags（0/1 随时间）。
    """
    import matplotlib.pyplot as plt

    f = np.asarray(fault_flags)
    if f.ndim == 2:
        # per_channel 时：先画 any（更直观）
        f = np.any(f, axis=1).astype(int)
    else:
        f = f.astype(int)

    f = _downsample(f.reshape(-1), max_points)
    fig, ax = plt.subplots(1, 1, figsize=(11, 2.8))
    ax.step(np.arange(len(f)), f, where="post", linewidth=1)
    ax.set_title("fault_flags (0/1)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = _plots_dir(work_dir) / "fault_flags.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def save_rscore_and_weights(
    work_dir: str,
    r_scores: np.ndarray,
    weights: np.ndarray,
    *,
    labels: Optional[Sequence[str]] = None,
) -> str:
    """
    图 4：rscore_and_weights（条形图）。
    - r_scores：每个源域模型的 R
    - weights：对应权重 W
    """
    import matplotlib.pyplot as plt

    r = np.asarray(r_scores, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if r.shape[0] != w.shape[0]:
        raise ValueError("r_scores 与 weights 长度必须一致")

    x = np.arange(len(r))
    lab = list(labels) if labels is not None else [f"src{i}" for i in range(len(r))]

    fig, ax1 = plt.subplots(1, 1, figsize=(11, 3.2))
    ax1.bar(x - 0.2, r, width=0.4, label="R-score", alpha=0.8)
    ax1.set_ylabel("R")
    ax1.set_xticks(x)
    ax1.set_xticklabels(lab, rotation=0)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, w, width=0.4, label="weights", alpha=0.6, color="tab:orange")
    ax2.set_ylabel("W")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()

    out = _plots_dir(work_dir) / "rscore_and_weights.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)

