from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(-1)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        # 兜底：均匀权重
        return (np.ones_like(w) / float(len(w))).astype(np.float32)
    return (w / s).astype(np.float32)


def compute_weights_from_r(
    r_scores: np.ndarray,
    *,
    rscore_is_distance: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    根据 R-score 计算权重 Wi。

    关键点（中文注释）：
    - 若 rscore_is_distance=True：R 越小越相似，所以 Wi ∝ 1/(R^2 + eps)
    - 否则：按论文原式 Wi ∝ R^2（R 越大权重越大）
    """
    r = np.asarray(r_scores, dtype=float).reshape(-1)
    if rscore_is_distance:
        w = 1.0 / (r**2 + float(eps))
    else:
        w = r**2 + float(eps)
    return _normalize_weights(w)


def _pick_single_best(
    r_scores: np.ndarray,
    *,
    rscore_is_distance: bool,
) -> int:
    r = np.asarray(r_scores, dtype=float).reshape(-1)
    # “最相似”定义：
    # - distance：越小越相似 -> argmin
    # - similarity：越大越相似 -> argmax
    return int(np.argmin(r) if rscore_is_distance else np.argmax(r))


def fuse_predictions(pred_list: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """
    对多个模型预测做加权融合。
    pred_list: 每个元素形状 [T,C] 或 [N,C]
    weights: [M]
    """
    if len(pred_list) == 0:
        raise ValueError("pred_list 为空")
    preds = [np.asarray(p, dtype=float) for p in pred_list]
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(preds) != w.shape[0]:
        raise ValueError("weights 长度必须等于 pred_list 数量")

    # shape 对齐检查
    shape0 = preds[0].shape
    for i, p in enumerate(preds):
        if p.shape != shape0:
            raise ValueError(f"第 {i} 个预测 shape 不一致：{p.shape} vs {shape0}")

    stack = np.stack(preds, axis=0)  # [M,T,C]
    w2 = _normalize_weights(w).reshape(-1, 1, 1)
    out = np.sum(stack * w2, axis=0)
    return out.astype(np.float32)


def ensemble_predict(
    pred_list: List[np.ndarray],
    r_scores: np.ndarray,
    *,
    no_ensemble_threshold: float = 0.3,
    rscore_is_distance: bool = True,
    eps: float = 1e-8,
    two_segment: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输出：final_pred, weights, r_scores

    规则（中文注释）：
    1) 如果 min(R_i) < no_ensemble_threshold：只用最相似的那个源域模型（不集成）
    2) 否则按权重融合：
       - distance：Wi ∝ 1/(R_i^2 + eps)
       - similarity：Wi ∝ R_i^2（论文原式）
    3) two_segment=True：对一条 flight 按时间均分两段，分别计算权重并融合，最后拼回。

    注意：
    - two_segment 需要 r_scores 形状为 [2,M]（每段一组 R）；如果传 [M] 则两段共用一组。
    """
    if len(pred_list) == 0:
        raise ValueError("pred_list 为空")

    rs = np.asarray(r_scores, dtype=float)
    m = len(pred_list)

    if rs.ndim == 1:
        if rs.shape[0] != m:
            raise ValueError("r_scores 形状必须为 [M]，且与 pred_list 匹配")
        rs_seg = np.stack([rs, rs], axis=0)  # [2,M]（便于统一处理）
    elif rs.ndim == 2:
        if rs.shape != (2, m):
            raise ValueError("two_segment 模式下 r_scores 必须是 [2,M]")
        rs_seg = rs
    else:
        raise ValueError("r_scores 必须是 [M] 或 [2,M]")

    # 先处理不分段情况
    if not two_segment:
        r0 = rs_seg[0]
        # 不集成条件：只对“距离型 R”使用 min 判断（按你的描述）
        if float(np.min(r0)) < float(no_ensemble_threshold):
            idx = _pick_single_best(r0, rscore_is_distance=rscore_is_distance)
            w = np.zeros((m,), dtype=np.float32)
            w[idx] = 1.0
            return np.asarray(pred_list[idx], dtype=np.float32), w, r0.astype(np.float32)

        w = compute_weights_from_r(r0, rscore_is_distance=rscore_is_distance, eps=eps)
        fused = fuse_predictions(pred_list, w)
        return fused, w, r0.astype(np.float32)

    # two_segment：时间均分两段分别融合再拼回
    p0 = np.asarray(pred_list[0])
    t = int(p0.shape[0])
    mid = t // 2
    if mid <= 0 or mid >= t:
        raise ValueError("预测长度太短，无法 two_segment 切分")

    out_parts = []
    w_parts = []
    for seg_idx, sl in enumerate([slice(0, mid), slice(mid, t)]):
        rseg = rs_seg[seg_idx]
        # 仍然遵循“不集成条件”
        if float(np.min(rseg)) < float(no_ensemble_threshold):
            best = _pick_single_best(rseg, rscore_is_distance=rscore_is_distance)
            w = np.zeros((m,), dtype=np.float32)
            w[best] = 1.0
            out = np.asarray(pred_list[best][sl], dtype=np.float32)
        else:
            w = compute_weights_from_r(rseg, rscore_is_distance=rscore_is_distance, eps=eps)
            out = fuse_predictions([p[sl] for p in pred_list], w)
        out_parts.append(out)
        w_parts.append(w)

    final_pred = np.concatenate(out_parts, axis=0).astype(np.float32)
    weights = np.stack(w_parts, axis=0).astype(np.float32)  # [2,M]
    return final_pred, weights, rs_seg.astype(np.float32)

