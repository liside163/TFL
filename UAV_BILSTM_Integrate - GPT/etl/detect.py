from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def compute_error(actual_p: np.ndarray, predicted_yhat: np.ndarray) -> np.ndarray:
    """
    (6) 误差定义：e = p - yhat
    - p：actual actuator（真实执行器输出）
    - yhat：predicted normal（模型预测的“正常输出”）
    """
    p = np.asarray(actual_p, dtype=float)
    y = np.asarray(predicted_yhat, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"actual_p 与 predicted_yhat shape 不一致：{p.shape} vs {y.shape}")
    return (p - y).astype(np.float32)


def _reduce_global(err_abs_or_signed: np.ndarray, reduce: str) -> np.ndarray:
    """
    把 [T,4] 误差聚合成 [T] 标量误差（global 模式用）。
    """
    e = np.asarray(err_abs_or_signed, dtype=float)
    r = (reduce or "mean").lower()
    if r == "mean":
        return np.mean(e, axis=1)
    if r == "max":
        return np.max(e, axis=1)
    if r == "l2":
        return np.sqrt(np.sum(e**2, axis=1) + 1e-12)
    raise ValueError(f"global_reduce 不支持：{reduce}（支持 mean|max|l2）")


def fit_error_stats(
    error_e: np.ndarray,
    *,
    error_mode: str = "per_channel",
    use_abs_error: bool = True,
    global_reduce: str = "mean",
) -> Dict[str, object]:
    """
    统计 mu/sigma（用于阈值 (7)）。

    说明（中文注释）：
    - per_channel：对每个通道分别统计，mu/sigma 为 [4]
    - global：先把通道聚合成标量，再统计，mu/sigma 为标量
    - any_channel：阈值仍按 per_channel 统计（输出 fault_flags[T] 时会 any 聚合）
    """
    e = np.asarray(error_e, dtype=float)
    if e.ndim != 2:
        raise ValueError("error_e 必须是二维 [T,C]")

    mode = (error_mode or "per_channel").lower()
    if use_abs_error:
        e2 = np.abs(e)
    else:
        e2 = e

    if mode in {"per_channel", "any_channel"}:
        mu = np.mean(e2, axis=0)
        sigma = np.std(e2, axis=0)
        return {
            "mode": mode,
            "use_abs_error": bool(use_abs_error),
            "mu": mu.astype(np.float32),
            "sigma": sigma.astype(np.float32),
        }

    if mode == "global":
        g = _reduce_global(e2, global_reduce)
        mu = float(np.mean(g))
        sigma = float(np.std(g))
        return {
            "mode": mode,
            "use_abs_error": bool(use_abs_error),
            "global_reduce": str(global_reduce),
            "mu": float(mu),
            "sigma": float(sigma),
        }

    raise ValueError(f"error_mode 不支持：{error_mode}（支持 per_channel/any_channel/global）")


def build_threshold(
    stats: Dict[str, object],
    *,
    alpha: float,
) -> Dict[str, object]:
    """
    (7) 阈值：T = mu + alpha*sigma
    """
    if "mu" not in stats or "sigma" not in stats:
        raise KeyError("stats 必须包含 mu/sigma")
    mu = stats["mu"]
    sigma = stats["sigma"]

    if isinstance(mu, np.ndarray):
        thr = np.asarray(mu, dtype=float) + float(alpha) * np.asarray(sigma, dtype=float)
        out = dict(stats)
        out["alpha"] = float(alpha)
        out["threshold"] = thr.astype(np.float32)
        return out

    thr = float(mu) + float(alpha) * float(sigma)
    out = dict(stats)
    out["alpha"] = float(alpha)
    out["threshold"] = float(thr)
    return out


def detect_faults(
    actual_p: np.ndarray,
    predicted_yhat: np.ndarray,
    threshold: Dict[str, object],
) -> Dict[str, np.ndarray]:
    """
    (8) 故障判据（简化实现）：
      fault = 1( |e| > T )

    输出：
    - e: [T,4]
    - T: [4] 或 标量（取决于 error_mode）
    - fault_flags: [T,4] 或 [T]（取决于 error_mode）
    """
    e = compute_error(actual_p, predicted_yhat)  # (6)
    mode = str(threshold.get("mode", "per_channel")).lower()
    use_abs = bool(threshold.get("use_abs_error", True))
    thr = threshold.get("threshold")

    if use_abs:
        e_cmp = np.abs(e)
    else:
        e_cmp = e

    if mode == "per_channel":
        t = np.asarray(thr, dtype=float).reshape(1, -1)
        flags = e_cmp > t
        return {
            "e": e.astype(np.float32),
            "T": np.asarray(thr, dtype=np.float32),
            "fault_flags": flags,
        }

    if mode == "any_channel":
        t = np.asarray(thr, dtype=float).reshape(1, -1)
        flags_ch = e_cmp > t
        flags_any = np.any(flags_ch, axis=1)
        return {
            "e": e.astype(np.float32),
            "T": np.asarray(thr, dtype=np.float32),
            "fault_flags": flags_any,
            "fault_flags_per_channel": flags_ch,
        }

    if mode == "global":
        reduce = str(threshold.get("global_reduce", "mean"))
        g = _reduce_global(e_cmp, reduce)
        t = float(thr)
        flags = g > t
        return {
            "e": e.astype(np.float32),
            "T": np.asarray(t, dtype=np.float32),
            "fault_flags": flags,
            "global_error": g.astype(np.float32),
        }

    raise ValueError(f"threshold.mode 不支持：{mode}")

