from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig


def _infer_columns(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    feature_cols: Optional[Sequence[str]],
    *,
    check_targets_exist: bool = True,
) -> Tuple[List[str], List[str]]:
    if check_targets_exist:
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            preview = list(df.columns[:30])
            raise KeyError(
                f"CSV 缺少 target_cols：{missing}\n"
                f"- 现有列预览(前30)：{preview}\n"
                "- 若你的目标是 4 维执行器输出，建议在 YAML 中配置 data.target_vector：\n"
                "  {base: \"_actuator_outputs_0_output\", dim: 4}（基列名按你的 CSV 实际列名修改）"
            )

    if feature_cols is not None:
        missing_f = [c for c in feature_cols if c not in df.columns]
        if missing_f:
            raise KeyError(f"CSV 缺少 feature_cols：{missing_f}")
        features = list(feature_cols)
    else:
        # 自动推断：选择数值列，剔除目标列（以及时间列由上层决定是否剔除）
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        features = [c for c in numeric_cols if c not in set(target_cols)]

    targets = list(target_cols)
    if not features:
        raise ValueError("未找到任何 feature 列；请检查 feature_cols 或 CSV 数据类型")
    return features, targets


def _resolve_vector_column_names(df: pd.DataFrame, base_name: str, dim: int) -> List[str]:
    """
    根据 df 实际列名，解析一个 vector(base,dim) 对应的“展开列名列表”。

    说明（中文注释）：
    - 若是展开列 base_0..base_{dim-1}，返回这些列名
    - 若是括号列 base[0]..base[dim-1]，返回这些列名
    - 若只有单列 base（字符串数组），这里返回空列表（自动特征推断不会把它当数值列）
    """
    cols = set(df.columns)
    expanded = [f"{base_name}_{i}" for i in range(int(dim))]
    if all(c in cols for c in expanded):
        return expanded
    bracketed = [f"{base_name}[{i}]" for i in range(int(dim))]
    if all(c in cols for c in bracketed):
        return bracketed
    if base_name in cols:
        return []
    # 不在这里报错，最终由 extract_vector 给出更清晰的“支持哪些列名形式”
    return []


def extract_vector(df: pd.DataFrame, base_name: str, dim: int) -> np.ndarray:
    """
    通用向量列抽取：自动识别三种 CSV 列风格并返回 [T,dim]。

    三种列风格：
    1) 单列存字符串数组：base_name -> "[a,b,c,d]"
    2) 多列展开：base_name_0 ... base_name_{dim-1}
    3) 列名包含 [i]：base_name[0] ... base_name[{dim-1}]

    缺失则抛出清晰异常。
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim 必须 > 0")

    cols = list(df.columns)
    expanded = [f"{base_name}_{i}" for i in range(dim)]
    bracketed = [f"{base_name}[{i}]" for i in range(dim)]

    # 2) 优先：展开列（数值列最直接）
    if all(c in cols for c in expanded):
        arr = df[expanded].to_numpy(dtype=np.float32, copy=True)
        return arr

    # 3) 次优先：带 [i] 的列名
    if all(c in cols for c in bracketed):
        arr = df[bracketed].to_numpy(dtype=np.float32, copy=True)
        return arr

    # 1) 单列字符串数组
    if base_name in cols:
        s = df[base_name]
        # 尝试把每行解析成 list/tuple
        out = np.zeros((len(df), dim), dtype=np.float32)
        for i, v in enumerate(s.to_list()):
            if isinstance(v, (list, tuple, np.ndarray)):
                seq = list(v)
            elif isinstance(v, str):
                try:
                    parsed = ast.literal_eval(v)
                except Exception as e:
                    raise ValueError(f"列 `{base_name}` 第 {i} 行无法解析为数组字符串：{v}") from e
                if not isinstance(parsed, (list, tuple)):
                    raise ValueError(f"列 `{base_name}` 第 {i} 行解析结果不是 list/tuple：{type(parsed).__name__}")
                seq = list(parsed)
            else:
                raise ValueError(f"列 `{base_name}` 第 {i} 行类型不支持：{type(v).__name__}（期望字符串数组或 list/tuple）")

            if len(seq) != dim:
                raise ValueError(f"列 `{base_name}` 第 {i} 行维度不匹配：期望 {dim}，实际 {len(seq)}")
            try:
                out[i] = np.asarray(seq, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"列 `{base_name}` 第 {i} 行无法转 float：{seq}") from e
        return out

    # 缺失：给出“你可以用哪些列名形式”
    raise KeyError(
        f"找不到向量列 `{base_name}`（dim={dim}）。支持列名："
        f"`{base_name}`(字符串数组) 或 {expanded} 或 {bracketed}"
    )


def _maybe_map_posvel_base(canonical_base: str, domain: Optional[str], features_mode: str) -> Optional[str]:
    """
    HIL vs Real 的 pos/vel 取列名映射：
    - Real: _vehicle_local_position_0_x / _vx -> actual
    - HIL/SIL: _vehicle_local_position_setpoint_0_x / _vx -> setpoint
    由 features_mode 决定是否纳入特征。

    关键点（中文注释）：
    - 为了保证“特征维度一致”，这里返回的是“读取用列名”，但特征名字仍建议用 canonical_base
    """
    fm = (features_mode or "all").lower()
    if fm in {"no_posvel", "no-position-vel", "exclude_posvel", "none"}:
        if "vehicle_local_position" in canonical_base:
            return None

    if domain is None:
        return canonical_base

    dom = str(domain)
    if dom in {"HIL", "SIL"}:
        if canonical_base.startswith("_vehicle_local_position_0_"):
            return canonical_base.replace("_vehicle_local_position_0_", "_vehicle_local_position_setpoint_0_")
    if dom == "Real":
        if canonical_base.startswith("_vehicle_local_position_setpoint_0_"):
            return canonical_base.replace("_vehicle_local_position_setpoint_0_", "_vehicle_local_position_0_")
    return canonical_base


def _extract_features_from_config(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    *,
    domain: Optional[str],
    features_mode: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    根据 scalar_features/vector_features 从 CSV 抽取特征矩阵 X。
    """
    feat_arrays: List[np.ndarray] = []
    feat_names: List[str] = []

    # 1) scalar_features：每个基列名对应 1 维
    for base in data_cfg.scalar_features:
        read_col = _maybe_map_posvel_base(base, domain, features_mode)
        if read_col is None:
            continue
        if read_col not in df.columns:
            if bool(data_cfg.allow_missing_features):
                continue
            raise KeyError(f"缺少特征列 `{read_col}`（由基列名 `{base}` 推导），请检查 CSV 或开启 allow_missing_features")
        v = pd.to_numeric(df[read_col], errors="coerce").astype(float).to_numpy(dtype=np.float32)
        if np.any(~np.isfinite(v)):
            # 缺失用插值补齐（避免直接 nan 进模型）
            v = pd.Series(v).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
        feat_arrays.append(v.reshape(-1, 1))
        feat_names.append(base)

    # 2) vector_features：每个基列名对应 dim 维（支持三种列风格）
    for item in data_cfg.vector_features:
        if not isinstance(item, dict) or "base" not in item or "dim" not in item:
            raise TypeError("data.vector_features 每项必须是 dict，且包含 base/dim，例如 {base: 'xxx', dim: 4}")
        base = str(item["base"])
        dim = int(item["dim"])
        required = bool(item.get("required", True))

        read_base = _maybe_map_posvel_base(base, domain, features_mode)
        if read_base is None:
            continue

        try:
            mat = extract_vector(df, read_base, dim)  # [T,dim]
        except KeyError as e:
            if bool(data_cfg.allow_missing_features) and not required:
                continue
            raise KeyError(f"缺少向量特征 `{read_base}`（由基列名 `{base}` 推导）：{e}") from e

        # 缺失值插值
        if np.any(~np.isfinite(mat)):
            mat2 = []
            for j in range(mat.shape[1]):
                col = mat[:, j].astype(float)
                if np.any(~np.isfinite(col)):
                    col = pd.Series(col).interpolate(limit_direction="both").to_numpy()
                mat2.append(col.astype(np.float32))
            mat = np.stack(mat2, axis=1)

        feat_arrays.append(mat)
        feat_names.extend([f"{base}[{i}]" for i in range(dim)])

    if not feat_arrays:
        raise ValueError(
            f"未抽取到任何特征：请在 data.scalar_features / data.vector_features 配置，或使用 data.feature_cols/自动推断。data_cfg={asdict(data_cfg)}"
        )
    x = np.concatenate(feat_arrays, axis=1).astype(np.float32, copy=False)
    return x, feat_names


def _resample_df(df: pd.DataFrame, time_col: str, target_hz: float) -> pd.DataFrame:
    """
    将不规则/高频序列按指定频率重采样（线性插值）。

    关键点（中文注释）：
    - 先把时间列转为 float 秒并排序
    - 用 numpy 构造均匀时间轴，再对每列做插值
    """
    if target_hz <= 0:
        raise ValueError("resample_hz 必须 > 0")
    t = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
    if np.any(~np.isfinite(t)):
        raise ValueError(f"time_col `{time_col}` 存在非数值/缺失，无法重采样")
    order = np.argsort(t)
    t = t[order]
    df_sorted = df.iloc[order].reset_index(drop=True)

    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0:
        return df_sorted

    dt = 1.0 / float(target_hz)
    t_new = np.arange(t0, t1 + 1e-9, dt, dtype=float)

    out = {}
    out[time_col] = t_new
    for col in df_sorted.columns:
        if col == time_col:
            continue
        y = pd.to_numeric(df_sorted[col], errors="coerce").astype(float).to_numpy()
        # 对缺失做简单前后填充，再插值
        if np.any(~np.isfinite(y)):
            y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
        out[col] = np.interp(t_new, t, y)
    return pd.DataFrame(out)


def load_flight_csv(
    path: str,
    data_cfg: DataConfig,
    *,
    domain: Optional[str] = None,
    features_mode: str = "all",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    读取单个 flight CSV，并抽取特征/目标矩阵。
    返回：
      X: [T, F] float32
      Y: [T, C] float32
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到 CSV：{path}")
    df = pd.read_csv(p)
    if data_cfg.time_col is not None and data_cfg.time_col not in df.columns:
        raise KeyError(f"CSV 缺少 time_col：{data_cfg.time_col}")

    if data_cfg.resample_hz is not None:
        if data_cfg.time_col is None:
            raise ValueError("指定了 resample_hz 但未提供 time_col；无法构建时间轴进行重采样")
        df = _resample_df(df, data_cfg.time_col, float(data_cfg.resample_hz))
        # 关键维度说明：重采样只改变时间长度 T，不改变特征/目标的列维度

    # 先处理目标规格（中文注释）：
    # - 若配置了 target_vector，则目标来自执行器输出向量列（不再要求 y1..y4 存在）
    # - 若未配置，则按 target_cols 读取
    target_cols_for_exclusion = list(data_cfg.target_cols)
    if data_cfg.target_vector is not None:
        tv = data_cfg.target_vector
        if not isinstance(tv, dict) or "base" not in tv or "dim" not in tv:
            raise TypeError("data.target_vector 必须是 dict 且包含 base/dim，例如 {base: 'xxx', dim: 4}")
        target_cols_for_exclusion = _resolve_vector_column_names(df, str(tv["base"]), int(tv["dim"]))

    # 特征抽取优先级：
    # 1) 显式 feature_cols（旧方式）
    # 2) scalar_features/vector_features（新方式）
    # 3) 自动推断数值列（兜底）
    if data_cfg.feature_cols is not None:
        # 旧方式：只校验 feature_cols 存在；target_cols 是否存在由 target_vector 决定
        feature_cols, target_cols = _infer_columns(
            df,
            data_cfg.target_cols,
            data_cfg.feature_cols,
            check_targets_exist=(data_cfg.target_vector is None),
        )
        if data_cfg.time_col is not None and data_cfg.time_col in feature_cols:
            feature_cols = [c for c in feature_cols if c != data_cfg.time_col]
        x = df[feature_cols].astype(float).to_numpy(dtype=np.float32)  # X: [T, F]
    elif data_cfg.scalar_features or data_cfg.vector_features:
        x, feature_cols = _extract_features_from_config(df, data_cfg, domain=domain, features_mode=features_mode)
        target_cols = list(data_cfg.target_cols)
    else:
        # 自动推断时，如果目标来自 target_vector，就不要强制要求 y1..y4 存在
        feature_cols, target_cols = _infer_columns(
            df,
            target_cols_for_exclusion,
            None,
            check_targets_exist=False,
        )
        if data_cfg.time_col is not None and data_cfg.time_col in feature_cols:
            feature_cols = [c for c in feature_cols if c != data_cfg.time_col]
        x = df[feature_cols].astype(float).to_numpy(dtype=np.float32)  # X: [T, F]

    # 目标抽取：优先 target_vector（适配向量列），否则用 target_cols（显式列名）
    if data_cfg.target_vector is not None:
        tv = data_cfg.target_vector
        base = str(tv["base"])
        dim = int(tv["dim"])
        y = extract_vector(df, base, dim)  # Y: [T, C]，此时 C=dim
        target_cols = [f"{base}[{i}]" for i in range(dim)]
    else:
        y = df[target_cols].astype(float).to_numpy(dtype=np.float32)  # Y: [T, C]
    return x, y, feature_cols, target_cols


def make_windows(length: int, window_size: int, stride: int) -> List[int]:
    if length < window_size:
        return []
    return list(range(0, length - window_size + 1, stride))


class WindowedFlightDataset(Dataset):
    """
    把多个 flight 串起来形成“窗口样本”的 Dataset。

    关键点（中文注释）：
    - 每个样本是一个窗口序列 X_win: [L, F]
    - 回归标签默认取窗口末端的目标值 y_last: [C]
    """

    def __init__(
        self,
        flights_x: List[np.ndarray],
        flights_y: List[np.ndarray],
        window_size: int,
        stride: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        self.flights_x = flights_x
        self.flights_y = flights_y
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.mean = mean
        self.std = std

        self._samples: List[Tuple[int, int]] = []
        for fi, x in enumerate(flights_x):
            for start in make_windows(len(x), self.window_size, self.stride):
                self._samples.append((fi, start))

        if not self._samples:
            raise ValueError("窗口化后没有任何样本；请检查 window_size/stride 或数据长度")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fi, start = self._samples[idx]
        # 关键维度变化：从整段序列 [T, F] 裁出窗口 [L, F]
        x = self.flights_x[fi][start : start + self.window_size]  # [L, F]
        # 关键维度变化：窗口标签取最后时刻目标 [C]，序列维度被压缩
        y = self.flights_y[fi][start + self.window_size - 1]  # [C]

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        return torch.from_numpy(x), torch.from_numpy(y)


def compute_normalizer(flights_x: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    用训练集统计特征归一化参数（逐特征 mean/std）。
    """
    cat = np.concatenate(flights_x, axis=0)
    mean = cat.mean(axis=0).astype(np.float32)
    std = cat.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def build_dataloaders(
    index_df: pd.DataFrame,
    data_cfg: DataConfig,
    seed: int,
    *,
    features_mode: str = "all",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """
    读取索引、划分 train/val/test，并构建 DataLoader。
    """
    if index_df.empty:
        raise ValueError("index_df 为空")

    rng = np.random.default_rng(int(seed))
    file_ids = index_df["file_id"].to_numpy()
    perm = rng.permutation(file_ids)

    n = len(perm)
    n_train = int(n * float(data_cfg.train_ratio))
    n_val = int(n * float(data_cfg.val_ratio))
    train_ids = set(perm[:n_train].tolist())
    val_ids = set(perm[n_train : n_train + n_val].tolist())
    test_ids = set(perm[n_train + n_val :].tolist())

    def _load_by_ids(ids: set[int]) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        feature_cols: Optional[List[str]] = None
        target_cols: Optional[List[str]] = None
        for _, row in index_df.iterrows():
            if int(row["file_id"]) not in ids:
                continue
            domain = str(row["domain"]) if "domain" in row else None
            x, y, f_cols, t_cols = load_flight_csv(row["path"], data_cfg, domain=domain, features_mode=features_mode)
            xs.append(x)
            ys.append(y)
            if feature_cols is None:
                feature_cols = f_cols
                target_cols = t_cols
            else:
                # 简单一致性检查：避免不同文件列顺序/列集不一致
                if f_cols != feature_cols or t_cols != target_cols:
                    raise ValueError(
                        "不同 CSV 的 feature/target 列不一致；请统一列名与顺序，或在 config 中显式指定 feature_cols/target_cols"
                    )
        if not xs:
            raise ValueError("划分后某个集合为空；请检查数据量或 split 参数")
        return xs, ys, feature_cols or [], target_cols or []

    train_x, train_y, feature_cols, target_cols = _load_by_ids(train_ids)
    val_x, val_y, _, _ = _load_by_ids(val_ids)
    test_x, test_y, _, _ = _load_by_ids(test_ids)

    mean, std = compute_normalizer(train_x)

    # Dataset 输出样本维度：X_win [L, F]，y_last [C]
    train_ds = WindowedFlightDataset(train_x, train_y, data_cfg.window_size, data_cfg.stride, mean, std)
    val_ds = WindowedFlightDataset(val_x, val_y, data_cfg.window_size, data_cfg.stride, mean, std)
    test_ds = WindowedFlightDataset(test_x, test_y, data_cfg.window_size, data_cfg.stride, mean, std)

    dl_kwargs = dict(
        batch_size=int(data_cfg.batch_size),
        num_workers=int(data_cfg.num_workers),
        pin_memory=bool(data_cfg.pin_memory),
        drop_last=False,
    )
    # DataLoader 批量维度：X [B, L, F]，y [B, C]
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    meta = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "normalizer": {"mean": mean, "std": std},
        "splits": {"train_ids": sorted(train_ids), "val_ids": sorted(val_ids), "test_ids": sorted(test_ids)},
        "data_cfg": asdict(data_cfg),
        "features_mode": features_mode,
    }
    return train_loader, val_loader, test_loader, meta
