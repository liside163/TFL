from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _try_minepy_mic(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    优先使用 minepy 的 MIC；如果 minepy 不可用则返回 None。
    """
    try:
        from minepy import MINE  # type: ignore
    except Exception:
        return None
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(x.astype(float), y.astype(float))
    return float(mine.mic())


def _mutual_info_regression(x: np.ndarray, y: np.ndarray) -> float:
    """
    mutual_info_regression 近似 MIC（注意：MI 本身不保证落在 [0,1]）。
    """
    from sklearn.feature_selection import mutual_info_regression

    x2 = x.reshape(-1, 1).astype(float)
    y2 = y.astype(float)
    mi = mutual_info_regression(x2, y2, random_state=0)
    return float(mi[0])


def _mic_like_1d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prefer_minepy: bool = True,
) -> float:
    """
    计算单个特征 x 与单个目标 y 的“MIC-like”分数。

    关键点（中文注释）：
    - minepy MIC 天然在 [0,1]
    - sklearn MI 没有上界，需要后续对整条向量做归一化
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x/y 长度不一致")
    if x.shape[0] < 5:
        return 0.0

    if prefer_minepy:
        mic = _try_minepy_mic(x, y)
        if mic is not None:
            return float(np.clip(mic, 0.0, 1.0))

    # MI 作为近似（可能很大）
    mi = _mutual_info_regression(x, y)
    return float(max(mi, 0.0))


def mic_vector(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    prefer_minepy: bool = True,
    normalize_to_01: bool = True,
) -> np.ndarray:
    """
    对一个域的数据计算相关性向量：
      v[j] = MIC(X_j, Y)

    支持 Y 为单输出或多输出：
    - 若 Y 是 [N,C]，则对每个通道分别算分数并取平均，得到 v[j]。

    归一化说明（中文注释）：
    - minepy MIC 已在 [0,1]；normalize_to_01=True 时仍可保持不变
    - MI 近似通常不在 [0,1]，因此会对整条向量做 min-max 归一化到 [0,1]
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 2:
        raise ValueError("X 必须是二维：[N,F]")
    if Y.ndim == 1:
        Y2 = Y.reshape(-1, 1)
    elif Y.ndim == 2:
        Y2 = Y
    else:
        raise ValueError("Y 必须是 [N] 或 [N,C]")
    if X.shape[0] != Y2.shape[0]:
        raise ValueError("X/Y 行数不一致")

    n, f = X.shape
    c = Y2.shape[1]
    v = np.zeros((f,), dtype=np.float64)
    for j in range(f):
        scores = []
        xj = X[:, j]
        for k in range(c):
            yk = Y2[:, k]
            scores.append(_mic_like_1d(xj, yk, prefer_minepy=prefer_minepy))
        v[j] = float(np.mean(scores)) if scores else 0.0

    if not normalize_to_01:
        return v.astype(np.float32)

    # min-max 归一化到 [0,1]（对 MI 近似很关键；对 MIC 也不会有坏处）
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmax - vmin < 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    vn = (v - vmin) / (vmax - vmin)
    return np.clip(vn, 0.0, 1.0).astype(np.float32)


def domain_mic_vectors(
    X: np.ndarray,
    Y: np.ndarray,
    domains: Sequence[str],
    *,
    prefer_minepy: bool = True,
    normalize_to_01: bool = True,
) -> Dict[str, np.ndarray]:
    """
    对每个域分别计算相关性向量 v_domain（MIC 向量）。

    domains：长度为 N 的域标签（例如 "SIL"/"HIL"/"Real"）。
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape[0] != len(domains):
        raise ValueError("domains 长度必须等于 X 的行数")

    out: Dict[str, np.ndarray] = {}
    uniq = sorted(set(str(d) for d in domains))
    for d in uniq:
        mask = np.array([str(x) == d for x in domains], dtype=bool)
        out[d] = mic_vector(X[mask], Y[mask], prefer_minepy=prefer_minepy, normalize_to_01=normalize_to_01)
    return out


def r_score(
    v_source: np.ndarray,
    v_target: np.ndarray,
    *,
    use_abs: bool = True,
    normalize_r: bool = False,
) -> float:
    """
    源域与目标域的 R-score（默认按“差异度/距离”实现）：
      R = sum_j abs(v_source[j] - v_target[j])

    可选参数（中文注释）：
    - use_abs：是否取 abs；默认 True（推荐用于“距离/差异度”）
    - normalize_r：是否除以维度 m，让 R 落在更稳定的尺度上
    """
    vs = np.asarray(v_source, dtype=float).reshape(-1)
    vt = np.asarray(v_target, dtype=float).reshape(-1)
    if vs.shape != vt.shape:
        raise ValueError("v_source/v_target 维度不一致")

    diff = vs - vt
    if use_abs:
        diff = np.abs(diff)
    r = float(np.sum(diff))
    if normalize_r and diff.size > 0:
        r = r / float(diff.size)
    return r


def r_scores_for_sources(
    v_sources: np.ndarray,
    v_target: np.ndarray,
    *,
    use_abs: bool = True,
    normalize_r: bool = False,
) -> np.ndarray:
    """
    多个源域向量与目标域向量的 R-score 批量计算。
    v_sources: [M,F]
    v_target: [F]
    """
    vs = np.asarray(v_sources, dtype=float)
    vt = np.asarray(v_target, dtype=float).reshape(1, -1)
    if vs.ndim != 2:
        raise ValueError("v_sources 必须是二维 [M,F]")
    if vs.shape[1] != vt.shape[1]:
        raise ValueError("v_sources 与 v_target 维度不匹配")
    diff = vs - vt
    if use_abs:
        diff = np.abs(diff)
    r = np.sum(diff, axis=1)
    if normalize_r and diff.shape[1] > 0:
        r = r / float(diff.shape[1])
    return r.astype(np.float32)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算皮尔逊相关系数（遇到常数列返回 0）。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size == 0:
        return 0.0
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rscore_table(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    feature_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
    prefer_minepy: bool = True,
    normalize_to_01: bool = True,
) -> "object":
    """
    兼容接口：生成 feature-target 的相关性表（用于保存 CSV/做快速可解释性分析）。

    注意（中文注释）：
    - 这是“工程输出/可视化用”的表格工具，不等价于你论文里“按域的 MIC 向量 + R 距离”
    - run.py 只需要能导出一个 CSV，所以这里保留旧函数名 rscore_table 防止 ImportError
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("rscore_table 需要 pandas；请安装 pandas 或在 run.py 中移除该输出步骤") from e

    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 2:
        raise ValueError("X 必须是二维：[N,F]")
    if Y.ndim == 1:
        Y2 = Y.reshape(-1, 1)
    elif Y.ndim == 2:
        Y2 = Y
    else:
        raise ValueError("Y 必须是 [N] 或 [N,C]")
    if X.shape[0] != Y2.shape[0]:
        raise ValueError("X/Y 行数不一致")

    n, f = X.shape
    c = Y2.shape[1]
    feature_names = feature_names or [f"f{i}" for i in range(f)]
    target_names = target_names or [f"y{i}" for i in range(c)]

    # 先算 raw mic-like（MI 可能不在 [0,1]）
    mic_raw = np.zeros((f, c), dtype=np.float64)
    corr = np.zeros((f, c), dtype=np.float64)
    for j in range(f):
        for k in range(c):
            mic_raw[j, k] = _mic_like_1d(X[:, j], Y2[:, k], prefer_minepy=prefer_minepy)
            corr[j, k] = _safe_corr(X[:, j], Y2[:, k])

    # 对每个 target 通道做归一化到 [0,1]（更便于比较/画图）
    if normalize_to_01:
        mic_01 = np.zeros_like(mic_raw, dtype=np.float64)
        for k in range(c):
            col = mic_raw[:, k]
            vmin = float(np.min(col))
            vmax = float(np.max(col))
            if vmax - vmin < 1e-12:
                mic_01[:, k] = 0.0
            else:
                mic_01[:, k] = (col - vmin) / (vmax - vmin)
        mic_used = mic_01
    else:
        mic_used = mic_raw

    rows = []
    for j in range(f):
        for k in range(c):
            # 一个常用的“综合分数”：|corr| * mic（非论文 R，仅用于排序展示）
            rs = abs(float(corr[j, k])) * float(mic_used[j, k])
            rows.append(
                {
                    "feature": feature_names[j],
                    "target": target_names[k],
                    "mic": float(mic_used[j, k]),
                    "mic_raw": float(mic_raw[j, k]),
                    "corr": float(corr[j, k]),
                    "rscore_display": float(rs),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["target", "rscore_display"], ascending=[True, False]).reset_index(drop=True)
