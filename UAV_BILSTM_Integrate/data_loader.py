"""
data_loader.py
================
为 RflyMAD 数据集构建可复现、可扩展的 PyTorch 数据管线（面向 BiLSTM 回归）。

目标（与论文“Ensemble Transfer Learning Based Cross-Domain UAV Actuator Fault Detection”一致的处理阶段思路）：
1) 源域（SIL/HIL，Case_1/Case_2）：有“故障类型/工况”等标签，可用于训练/验证与评估；
2) 目标域（Real，Case_3）：在实际“新机型/新域”场景中通常缺少标签，因此支持 mode='unlabeled'
   ——即使文件名里包含 fault code，也在加载时忽略这些“分类标签”（但回归目标 PWM 依然来自传感器记录）。
3) 输入特征包含物理一致的派生量：四元数->欧拉角（roll/pitch）、地速 ground_speed（vx/vy 代理 airspeed）；
4) 输出为执行器 PWM（代理 aileron angle），做 MinMax 归一化到 [0,1] 或 [-1,1]，便于回归稳定。

注意：
- RflyMAD 不同版本/导出脚本可能导致列名存在差异（带 topic 前缀、[] 下标等），本文件尽量做“鲁棒列名匹配”。
- 若你能提供一份真实 CSV 的表头，本脚本还可以进一步“精确适配列名”。
"""

from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import yaml  # PyYAML
except Exception as exc:  # pragma: no cover
    raise ImportError("需要安装 PyYAML：pip install pyyaml") from exc

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

import torch
from torch.utils.data import Dataset, DataLoader


ArrayLike = Union[np.ndarray, torch.Tensor]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_name(name: str) -> str:
    """
    将列名归一化为便于匹配的形式：
    - 全小写
    - 将 ., /, 空格, [] 等统一替换为 '_'
    - 连续 '_' 压缩
    """
    name = name.strip().lower()
    name = re.sub(r"[\s\.\-\/\\\[\]\(\)\{\}:]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _build_normalized_column_map(columns: Sequence[str]) -> Dict[str, str]:
    """
    返回：normalized_name -> original_name（若冲突，优先保留更短、更“原始”的名字）。
    """
    mapping: Dict[str, str] = {}
    for col in columns:
        n = _normalize_name(col)
        if n not in mapping:
            mapping[n] = col
            continue
        if len(col) < len(mapping[n]):
            mapping[n] = col
    return mapping


def _first_existing_column(
    normalized_to_original: Dict[str, str],
    candidates: Sequence[str],
) -> Optional[str]:
    """
    在 normalized_to_original 中寻找第一个命中的列名（支持给出多种可能写法）。
    """
    for cand in candidates:
        key = _normalize_name(cand)
        if key in normalized_to_original:
            return normalized_to_original[key]
    return None


def _first_by_normalized_regex(
    normalized_to_original: Dict[str, str],
    patterns: Sequence[str],
) -> Optional[str]:
    """
    在“归一化列名”空间里做正则匹配，解决某些列名含 topic 下标（例如 _0_）或 [] 的情况。
    """
    keys = list(normalized_to_original.keys())
    for pat in patterns:
        rx = re.compile(pat)
        for k in keys:
            if rx.search(k):
                return normalized_to_original[k]
    return None


def quaternion_to_euler(
    q0: ArrayLike,
    q1: ArrayLike,
    q2: ArrayLike,
    q3: ArrayLike,
    *,
    order: Literal["wxyz", "xyzw"] = "wxyz",
    degrees: bool = False,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将四元数转换为欧拉角（roll, pitch, yaw），默认假设输入为 (w, x, y, z)。

    为什么要这样做：
    - 论文使用 roll/pitch 等姿态角作为“可解释”的物理特征；
    - 跨域迁移（SIL/HIL->Real）时，物理一致的特征往往比原始通道更稳健（减少域偏移的影响）。

    实现要点（鲁棒性）：
    - 对 asin 的输入进行 clamp，避免数值误差导致 NaN；
    - 支持 order='xyzw' 的情况（有些系统将 w 放在末尾）。
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q3 = np.asarray(q3, dtype=np.float64)

    if order == "xyzw":
        x, y, z, w = q0, q1, q2, q3
    else:
        w, x, y, z = q0, q1, q2, q3

    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1 + eps)

    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4 + eps)

    if degrees:
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)

    return roll.astype(np.float32), pitch.astype(np.float32), yaw.astype(np.float32)


def derive_ground_speed(vx: ArrayLike, vy: ArrayLike) -> np.ndarray:
    """
    用局部坐标系水平速度 (vx, vy) 构造 ground_speed 作为 airspeed 代理：
    - 论文需要 airspeed，但 RflyMAD 通常没有 pitot；
    - ground_speed 虽不等价（忽略风速），但在很多控制/动力学相关任务中可作为近似输入。
    """
    vx = np.asarray(vx, dtype=np.float32)
    vy = np.asarray(vy, dtype=np.float32)
    return np.sqrt(vx * vx + vy * vy).astype(np.float32)


def normalize_pwm(
    pwm: ArrayLike,
    *,
    pwm_min: float = 1000.0,
    pwm_max: float = 2000.0,
    out_range: Tuple[float, float] = (0.0, 1.0),
    clip: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    PWM 归一化：将 [pwm_min, pwm_max] 映射到 out_range（默认 [0,1]）。

    为什么必须做：
    - 论文预测 aileron angle，本质是连续量；PWM 是控制输出的原始量纲，范围较大且跨机型可能偏移；
    - 归一化能提升回归的数值稳定性，也更利于迁移学习（减少不同平台输出尺度差异）。
    """
    pwm = np.asarray(pwm, dtype=np.float32)
    if clip:
        pwm = np.clip(pwm, pwm_min, pwm_max)
    norm01 = (pwm - pwm_min) / (pwm_max - pwm_min + eps)
    lo, hi = out_range
    return (lo + (hi - lo) * norm01).astype(np.float32)


def sliding_window(
    x: np.ndarray,
    y: Optional[np.ndarray],
    *,
    seq_len: int,
    stride: int = 1,
    target_mode: Literal["last", "seq"] = "last",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    将 (N, F) 的表格数据切成 (M, seq_len, F) 的时序窗口。

    - seq_len：BiLSTM 需要的时间长度；
    - stride：滑窗步长，越大样本越少但冗余更低；
    - target_mode='last'：每个窗口对应最后一帧的回归目标（常用于“用过去预测当前/下一刻”）；
      target_mode='seq'：每个窗口对应整段目标序列（适用于 sequence-to-sequence 回归）。

    信号处理视角：
    - 滑窗相当于将长序列分解成局部平稳片段，利于 RNN 学习短期动力学规律；
    - stride 可减少相邻样本高度相关造成的“有效样本数虚高”问题。
    """
    if x.ndim != 2:
        raise ValueError(f"x 必须是二维数组 (N,F)，但得到 shape={x.shape}")
    if y is not None and y.shape[0] != x.shape[0]:
        raise ValueError("x 与 y 的时间长度不一致")
    if seq_len <= 0:
        raise ValueError("seq_len 必须 > 0")
    if stride <= 0:
        raise ValueError("stride 必须 > 0")

    n = x.shape[0]
    if n < seq_len:
        x_empty = np.empty((0, seq_len, x.shape[1]), dtype=np.float32)
        if y is None:
            return x_empty, None
        y = np.asarray(y)
        if target_mode == "seq":
            if y.ndim == 1:
                return x_empty, np.empty((0, seq_len), dtype=np.float32)
            return x_empty, np.empty((0, seq_len, y.shape[1]), dtype=np.float32)
        else:  # last
            if y.ndim == 1:
                return x_empty, np.empty((0,), dtype=np.float32)
            return x_empty, np.empty((0, y.shape[1]), dtype=np.float32)

    starts = range(0, n - seq_len + 1, stride)
    x_seq = np.stack([x[s : s + seq_len] for s in starts], axis=0).astype(np.float32)

    if y is None:
        return x_seq, None

    if target_mode == "seq":
        y_seq = np.stack([y[s : s + seq_len] for s in starts], axis=0).astype(np.float32)
    else:  # last
        y_seq = np.stack([y[s + seq_len - 1] for s in starts], axis=0).astype(np.float32)

    return x_seq, y_seq


def parse_rflymad_filename(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    解析 RflyMAD 文件名中的 Case 编码，提取元数据。

    论文设定中的关键：
    - Case_1 / Case_2：源域（SIL/HIL），通常标签完备；
    - Case_3：目标域（Real），在迁移学习任务里应当模拟“无标签”，因此 Dataset 支持忽略 fault 标签。

    文件名格式在不同整理方式下可能不同，本函数采用“尽量宽松”的正则策略：
    - 找到 'Case_' 后连续数字串，记为 code，例如 '31205'；
      A=3（域），B=1（飞行状态），CD=20（故障类型）……其余作为 run_id。
    """
    name = Path(file_path).name
    m = re.search(r"case[_-]?(\d+)", name, re.IGNORECASE)
    if not m:
        return {"filename": name, "case_code": None}

    code = m.group(1)
    domain_code = int(code[0]) if len(code) >= 1 else None
    flight_state = int(code[1]) if len(code) >= 2 else None
    fault_type = int(code[2:4]) if len(code) >= 4 else None
    run_id = int(code[4:]) if len(code) > 4 and code[4:].isdigit() else None

    return {
        "filename": name,
        "case_code": code,
        "domain_code": domain_code,  # 1/2/3
        "flight_state": flight_state,
        "fault_type": fault_type,
        "run_id": run_id,
    }


@dataclass
class StandardScaler:
    """
    简单的标准化器：x -> (x - mean) / std

    迁移学习视角：
    - 只用源域训练集拟合 mean/std，再应用到目标域，可以避免“窥探目标域分布”造成的评估乐观偏差。
    """

    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, eps: float = 1e-8) -> "StandardScaler":
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0) + eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler 还未 fit")
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    # 一些用户会把论文/Markdown/引用标记直接粘贴进 YAML（例如 [cite_start]、[cite: 12]），
    # 这会导致 YAML 解析失败。这里做一次“安全清洗”，只移除这类无语义标记，避免影响实验流程。
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    text = text.replace("[cite_start]", "").replace("[cite_end]", "")
    text = re.sub(r"\[cite:\s*[0-9,\s]+\]", "", text)
    cfg = yaml.safe_load(text)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml 顶层必须是 dict")
    return cfg


def _as_path_list(v: Any) -> List[Path]:
    if v is None:
        return []
    if isinstance(v, (str, Path)):
        return [Path(v)]
    return [Path(x) for x in v]


def discover_csv_files(
    roots: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    patterns: Sequence[str] = ("Case_*.csv",),
    recursive: bool = True,
) -> List[Path]:
    roots_list = _as_path_list(roots)
    files: List[Path] = []
    for root in roots_list:
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"数据根目录不存在：{root}")
        for pat in patterns:
            if recursive:
                files.extend(root.rglob(pat))
            else:
                files.extend(root.glob(pat))
    files = sorted({f.resolve() for f in files})
    return files


def _read_csv(path: Path) -> "pd.DataFrame":
    if pd is None:  # pragma: no cover
        raise ImportError("需要安装 pandas：pip install pandas")
    # 常见兼容性问题：不同导出工具可能含 BOM、不同分隔符；这里先用默认 read_csv，
    # 若你遇到分隔符/编码问题，可在 config 中扩展 read_csv_kwargs。
    return pd.read_csv(path)


def _coerce_numeric(df: "pd.DataFrame") -> "pd.DataFrame":
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def preprocess_rflymad_dataframe(
    df: "pd.DataFrame",
    *,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    从原始 DataFrame 提取模型输入 X 与回归目标 Y（PWM），并做物理特征工程。

    返回：
    - x: (N, F) float32
    - y: (N, T) float32（T 为 PWM 通道数）
    - fault_state: (N,) int64（来自 UAVState_data_fault_state，0=无故障，1=故障发生）
    - info: 记录实际使用了哪些列名，便于论文复现与排错
    """
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    norm_map = _build_normalized_column_map(df.columns)
    df = _coerce_numeric(df)

    cols_cfg = config.get("columns", {}) if isinstance(config.get("columns", {}), dict) else {}
    feats_cfg = config.get("features", {}) if isinstance(config.get("features", {}), dict) else {}
    norm_cfg = config.get("normalization", {}) if isinstance(config.get("normalization", {}), dict) else {}

    # 1) 时间戳列（可选）：用于排序/去重/重采样（如需）
    ts_col = None
    ts_candidates = cols_cfg.get(
        "timestamp",
        # RflyMAD 常见：Timestamp/trueTime
        ["timestamp", "truetime", "time_us", "time", "t", "hrt_absolute_time", "vehicle_local_position_timestamp"],
    )
    if isinstance(ts_candidates, (str, Path)):
        ts_candidates = [str(ts_candidates)]
    ts_col = _first_existing_column(norm_map, ts_candidates)
    if ts_col is not None:
        df = df.sort_values(ts_col).drop_duplicates(ts_col, keep="first")

    # 2) 四元数 -> roll/pitch/yaw
    add_euler = bool(feats_cfg.get("add_euler", True))
    quat_cols = cols_cfg.get("quaternion", ["q0", "q1", "q2", "q3"])
    if isinstance(quat_cols, (str, Path)):
        quat_cols = [str(quat_cols)]
    quat_order = str(feats_cfg.get("quat_order", "wxyz")).lower()
    used_quat: Optional[List[str]] = None

    if add_euler:
        # 常见列名候选：q0/q1/q2/q3 或 vehicle_attitude_q_0 等
        q0 = _first_existing_column(
            norm_map,
            [
                quat_cols[0],
                "_vehicle_attitude_0_q[0]",
                "vehicle_attitude_0_q[0]",
                "vehicle_attitude_q_0",
                "att_q_0",
                "q0",
                "q_0",
                "q[0]",
            ],
        ) or _first_by_normalized_regex(norm_map, [r"vehicle_attitude_\d+_q_0$"])
        q1 = _first_existing_column(
            norm_map,
            [
                quat_cols[1],
                "_vehicle_attitude_0_q[1]",
                "vehicle_attitude_0_q[1]",
                "vehicle_attitude_q_1",
                "att_q_1",
                "q1",
                "q_1",
                "q[1]",
            ],
        ) or _first_by_normalized_regex(norm_map, [r"vehicle_attitude_\d+_q_1$"])
        q2 = _first_existing_column(
            norm_map,
            [
                quat_cols[2],
                "_vehicle_attitude_0_q[2]",
                "vehicle_attitude_0_q[2]",
                "vehicle_attitude_q_2",
                "att_q_2",
                "q2",
                "q_2",
                "q[2]",
            ],
        ) or _first_by_normalized_regex(norm_map, [r"vehicle_attitude_\d+_q_2$"])
        q3 = _first_existing_column(
            norm_map,
            [
                quat_cols[3],
                "_vehicle_attitude_0_q[3]",
                "vehicle_attitude_0_q[3]",
                "vehicle_attitude_q_3",
                "att_q_3",
                "q3",
                "q_3",
                "q[3]",
            ],
        ) or _first_by_normalized_regex(norm_map, [r"vehicle_attitude_\d+_q_3$"])
        if all([q0, q1, q2, q3]):
            roll, pitch, yaw = quaternion_to_euler(
                df[q0].to_numpy(),
                df[q1].to_numpy(),
                df[q2].to_numpy(),
                df[q3].to_numpy(),
                order="xyzw" if quat_order == "xyzw" else "wxyz",
                degrees=bool(feats_cfg.get("euler_degrees", False)),
            )
            df["roll"] = roll
            df["pitch"] = pitch
            df["yaw"] = yaw
            used_quat = [q0, q1, q2, q3]
        else:
            # 某些记录可能缺失姿态 topic，导致无法由四元数构造欧拉角。
            # 为了保证跨域/跨文件的输入维度一致，这里补 0（而不是直接丢弃特征列）。
            # 研究复现时建议在 config.features.input 里显式指定特征，保证维度一致。
            df["roll"] = 0.0
            df["pitch"] = 0.0
            df["yaw"] = 0.0

    # 3) ground_speed（vx/vy）
    add_gs = bool(feats_cfg.get("add_ground_speed", True))
    used_vxy: Optional[List[str]] = None
    if add_gs:
        vxy_cols = cols_cfg.get("velocity_xy", ["vx", "vy"])
        if isinstance(vxy_cols, (str, Path)):
            vxy_cols = [str(vxy_cols)]
        vx = _first_existing_column(
            norm_map,
            [vxy_cols[0], "_vehicle_local_position_0_vx", "vehicle_local_position_0_vx", "vehicle_local_position_vx", "local_vx", "vx"],
        ) or _first_by_normalized_regex(norm_map, [r"vehicle_local_position_\d+_vx$"])
        vy = _first_existing_column(
            norm_map,
            [vxy_cols[1], "_vehicle_local_position_0_vy", "vehicle_local_position_0_vy", "vehicle_local_position_vy", "local_vy", "vy"],
        ) or _first_by_normalized_regex(norm_map, [r"vehicle_local_position_\d+_vy$"])
        if vx and vy:
            df["ground_speed"] = derive_ground_speed(df[vx].to_numpy(), df[vy].to_numpy())
            used_vxy = [vx, vy]
        else:
            # 同理：若缺速度 topic，则用 0 填充，保证输入维度一致
            df["ground_speed"] = 0.0

    # 4) PWM 目标列
    pwm_cols_cfg = cols_cfg.get("pwm", None)
    pwm_columns: List[str] = []
    if pwm_cols_cfg is None:
        # 尝试自动搜：常见的 actuator_outputs 输出列名会包含 pwm/out/output
        candidates = []
        for c in df.columns:
            n = _normalize_name(c)
            if re.search(r"(pwm|actuator|output)", n):
                candidates.append(c)
        # 进一步过滤：只保留看起来像通道（含数字下标/后缀）的列
        pwm_columns = sorted(
            [c for c in candidates if re.search(r"(\d+|_\d+|\[\d+\])", c)],
            key=lambda s: _normalize_name(s),
        )
    else:
        if isinstance(pwm_cols_cfg, (str, Path)):
            pwm_cols_cfg = [str(pwm_cols_cfg)]
        for c in pwm_cols_cfg:
            real = _first_existing_column(norm_map, [c])
            if real is not None:
                pwm_columns.append(real)

    if not pwm_columns:
        raise KeyError(
            "未找到 PWM 目标列。请在 config.yaml 的 columns.pwm 显式指定，例如：\n"
            "columns:\n  pwm: [output_0, output_1, output_2, output_3]\n"
        )

    pwm_min = float(norm_cfg.get("pwm_min", 1000.0))
    pwm_max = float(norm_cfg.get("pwm_max", 2000.0))
    out_range = tuple(norm_cfg.get("pwm_out_range", [0.0, 1.0]))
    if len(out_range) != 2:
        raise ValueError("normalization.pwm_out_range 必须是长度为2的列表/元组")
    y = normalize_pwm(df[pwm_columns].to_numpy(), pwm_min=pwm_min, pwm_max=pwm_max, out_range=(float(out_range[0]), float(out_range[1])))

    # 4.1) 故障发生标记（来自数据列，而不是文件名）
    # 用户数据集约定：UAVState_data_fault_state == 0 表示故障未发生，==1 表示故障发生。
    fault_cfg = cols_cfg.get("fault_state", ["UAVState_data_fault_state", "uavstate_data_fault_state"])
    if isinstance(fault_cfg, (str, Path)):
        fault_cfg = [str(fault_cfg)]
    fault_col = _first_existing_column(norm_map, list(fault_cfg))
    if fault_col is None:
        # 没有该列时默认全 0（等价于未知/无故障），保证流水线可运行
        fault_state = np.zeros((df.shape[0],), dtype=np.int64)
    else:
        fs = df[fault_col].to_numpy()
        fs = np.nan_to_num(fs, nan=0.0, posinf=0.0, neginf=0.0)
        fault_state = (fs.astype(np.int64) != 0).astype(np.int64)

    # 5) 输入特征列
    # 迁移学习实践：尽量使用“跨域更稳定”的状态量（姿态角、速度、位置、IMU 等），少用强依赖平台的控制内部量。
    input_cols_cfg = feats_cfg.get("input", None)
    x_columns: List[str] = []
    if input_cols_cfg is None:
        # 默认：优先选择我们新构造的物理特征 + 常见状态量
        default_candidates = [
            "roll",
            "pitch",
            "yaw",
            "ground_speed",
            # 兼容 RflyMAD 常见列名（带 _0_ 与 []）
            "_vehicle_local_position_0_vx",
            "_vehicle_local_position_0_vy",
            "_vehicle_local_position_0_vz",
            "_sensor_combined_0_gyro_rad[0]",
            "_sensor_combined_0_gyro_rad[1]",
            "_sensor_combined_0_gyro_rad[2]",
            "_sensor_combined_0_accelerometer_m_s2[0]",
            "_sensor_combined_0_accelerometer_m_s2[1]",
            "_sensor_combined_0_accelerometer_m_s2[2]",
            "x",
            "y",
            "z",
            "u",
            "v",
            "w",
            "p",
            "q",
            "r",
        ]
        for cand in default_candidates:
            real = _first_existing_column(norm_map, [cand])
            if real is not None and real not in pwm_columns:
                x_columns.append(real)
        # 我们构造的列在 df 中，不在 norm_map，需要额外添加
        for constructed in ["roll", "pitch", "yaw", "ground_speed"]:
            if constructed in df.columns and constructed not in x_columns:
                x_columns.insert(0, constructed)
    else:
        if isinstance(input_cols_cfg, (str, Path)):
            input_cols_cfg = [str(input_cols_cfg)]
        # 当用户显式指定输入列时，务必保证输出维度固定一致：
        # - 若列不存在，则创建该列并用 0 填充（避免不同文件/不同域出现特征维度不一致，影响迁移与 R-score）。
        for c in input_cols_cfg:
            if c in df.columns:  # 允许直接引用构造列（roll/pitch 等）
                x_columns.append(c)
                continue
            real = _first_existing_column(norm_map, [c])
            if real is not None:
                x_columns.append(real)
                continue
            df[c] = 0.0
            x_columns.append(c)

    x_columns = [c for c in x_columns if c in df.columns and c not in pwm_columns]
    if not x_columns:
        # 最后的保底策略：自动选择数值列（去掉 PWM/时间戳等），避免因为列名差异导致流程直接中断。
        # 研究复现中，保底策略可以先“跑通流程”，然后再根据表头精确指定 features.input 提升可解释性与一致性。
        exclude = set(pwm_columns)
        if ts_col is not None:
            exclude.add(ts_col)
        # 排除四元数原始列（我们更倾向用 roll/pitch 等可解释特征）
        if used_quat:
            exclude.update(used_quat)
        numeric_cols = []
        for c in df.columns:
            if c in exclude:
                continue
            if np.issubdtype(df[c].dtype, np.number):
                numeric_cols.append(c)
        max_auto = int(feats_cfg.get("max_auto_features", 20))
        x_columns = numeric_cols[:max_auto]
        if not x_columns:
            raise KeyError(
                "未找到输入特征列。请在 config.yaml 的 features.input 显式指定，或确保 CSV 中存在可识别的状态列。"
            )

    x = df[x_columns].to_numpy(dtype=np.float32)

    # 6) 清洗：删除 NaN/Inf
    clean_cfg = config.get("cleaning", {}) if isinstance(config.get("cleaning", {}), dict) else {}
    dropna = bool(clean_cfg.get("dropna", True))
    if dropna:
        mask = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x = x[mask]
        y = y[mask]
        fault_state = fault_state[mask]

    info = {
        "timestamp_col": ts_col,
        "used_quaternion_cols": used_quat,
        "used_velocity_xy_cols": used_vxy,
        "x_columns": x_columns,
        "pwm_columns": pwm_columns,
        "fault_state_col": fault_col,
    }
    return x, y, fault_state, info


class RflyMADDataset(Dataset):
    """
    RflyMAD 数据集（滑窗序列）：
    - 每个 CSV 文件是一个飞行记录；
    - 先提取并清洗为 (N,F)/(N,T)，再滑窗成 (M,seq_len,F)；
    - 返回 (x_seq, y_target, meta)；其中 meta 包含 case 信息与可选的 fault 标签。

    mode 说明：
    - mode='labeled'：保留 fault_type 等标签（用于评估/可视化）；
    - mode='unlabeled'：忽略 fault 标签（模拟目标域缺标签的迁移学习设置）。
      注意：回归目标 PWM 不是“分类标签”，因此仍会返回 y（用于残差/重构等自监督或回归任务）。
    """

    def __init__(
        self,
        files: Sequence[Union[str, Path]],
        *,
        config: Union[Dict[str, Any], str, Path],
        mode: Literal["labeled", "unlabeled"] = "labeled",
        scaler_x: Optional[StandardScaler] = None,
        fit_scaler_on_init: bool = False,
        return_meta: bool = False,
        return_fault_state: bool = False,
    ) -> None:
        super().__init__()
        self.files = [Path(f).expanduser().resolve() for f in files]
        self.config = load_yaml(config) if isinstance(config, (str, Path)) else dict(config)
        self.mode = mode
        self.return_meta = return_meta
        self.return_fault_state = return_fault_state

        win_cfg = self.config.get("window", {}) if isinstance(self.config.get("window", {}), dict) else {}
        self.seq_len = int(win_cfg.get("seq_len", 50))
        self.stride = int(win_cfg.get("stride", 1))
        self.target_mode = str(win_cfg.get("target_mode", "last")).lower()
        # fault_label_mode:
        # - last: 以窗口最后一帧的 fault_state 作为标签（与预测 y 的对齐方式一致）
        # - any : 只要窗口内出现过 fault_state==1 就视为故障（更偏“事件检测”）
        self.fault_label_mode = str(win_cfg.get("fault_label_mode", "last")).lower()
        if self.fault_label_mode not in ("last", "any"):
            raise ValueError("window.fault_label_mode 只能是 'last' 或 'any'")
        if self.target_mode not in ("last", "seq"):
            raise ValueError("window.target_mode 只能是 'last' 或 'seq'")

        self._scaler_x = scaler_x
        self._x_seq: List[np.ndarray] = []
        self._y_seq: List[np.ndarray] = []
        self._fault_seq: List[np.ndarray] = []
        self._meta: List[Dict[str, Any]] = []
        self._info_per_file: Dict[str, Dict[str, Any]] = {}

        self._build_cache(fit_scaler_on_init=fit_scaler_on_init)

    @property
    def scaler_x(self) -> Optional[StandardScaler]:
        return self._scaler_x

    @property
    def info_per_file(self) -> Dict[str, Dict[str, Any]]:
        return self._info_per_file

    def _build_cache(self, *, fit_scaler_on_init: bool) -> None:
        x_all_for_fit: List[np.ndarray] = []

        for f in self.files:
            df = _read_csv(f)
            x, y, fault_state, info = preprocess_rflymad_dataframe(df, config=self.config)

            if fit_scaler_on_init:
                x_all_for_fit.append(x)

            meta = parse_rflymad_filename(f)
            # 迁移学习设置：目标域“无标签”时，忽略 fault_type 等分类标签，避免训练/筛选时使用它。
            if self.mode == "unlabeled":
                meta["fault_type"] = None

            x_seq, y_seq = sliding_window(
                x,
                y,
                seq_len=self.seq_len,
                stride=self.stride,
                target_mode="seq" if self.target_mode == "seq" else "last",
            )
            # fault_state 的窗口标签（对齐到 y 的时间点）
            if x_seq.shape[0] > 0:
                n = x.shape[0]
                starts = range(0, n - self.seq_len + 1, self.stride)
                if self.fault_label_mode == "any":
                    f_seq = np.asarray(
                        [int(np.max(fault_state[s : s + self.seq_len])) for s in starts], dtype=np.int64
                    )
                else:
                    f_seq = np.asarray([int(fault_state[s + self.seq_len - 1]) for s in starts], dtype=np.int64)
            else:
                f_seq = np.empty((0,), dtype=np.int64)

            if x_seq.shape[0] == 0:
                continue

            # 先暂存，等 scaler fit 后再统一 transform（保证一致性）
            self._x_seq.append(x_seq)
            self._y_seq.append(y_seq if y_seq is not None else np.zeros((x_seq.shape[0],), dtype=np.float32))
            self._fault_seq.append(f_seq)
            self._meta.extend([meta] * x_seq.shape[0])
            self._info_per_file[str(f)] = info

        if not self._x_seq:
            raise RuntimeError("未从任何 CSV 构造出有效序列样本（可能 seq_len 太大或列名不匹配）")

        x_cat = np.concatenate(self._x_seq, axis=0)  # (M, L, F)
        y_cat = np.concatenate(self._y_seq, axis=0)
        fault_cat = np.concatenate(self._fault_seq, axis=0) if self._fault_seq else np.zeros((x_cat.shape[0],), dtype=np.int64)

        # 标准化输入 X（对每个特征维度做整体标准化；对时间维度视为样本集合）
        norm_cfg = self.config.get("normalization", {}) if isinstance(self.config.get("normalization", {}), dict) else {}
        x_norm_type = str(norm_cfg.get("x", "standard")).lower()
        if x_norm_type not in ("none", "standard"):
            raise ValueError("normalization.x 目前支持 'none' 或 'standard'")

        if x_norm_type == "standard":
            if self._scaler_x is None:
                self._scaler_x = StandardScaler()
                if fit_scaler_on_init:
                    # 用未滑窗前的 x 拟合也可以；这里用滑窗后的所有帧拟合，效果等价且实现更简单
                    self._scaler_x.fit(x_cat.reshape(-1, x_cat.shape[-1]))
                else:
                    # 若未要求 fit，则默认用当前数据 fit（方便快速跑通；严格迁移实验建议仅在源域训练集 fit）
                    self._scaler_x.fit(x_cat.reshape(-1, x_cat.shape[-1]))
            x_cat = self._scaler_x.transform(x_cat.reshape(-1, x_cat.shape[-1])).reshape(x_cat.shape)

        self._x = torch.from_numpy(x_cat)  # (M, L, F)
        self._y = torch.from_numpy(y_cat)  # (M, T) 或 (M, L, T)
        self._fault = torch.from_numpy(fault_cat.astype(np.int64))  # (M,)

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int):
        x = self._x[idx]
        y = self._y[idx]
        if self.return_meta and self.return_fault_state:
            return x, y, self._fault[idx], self._meta[idx]
        if self.return_fault_state:
            return x, y, self._fault[idx]
        if self.return_meta:
            return x, y, self._meta[idx]
        return x, y


def build_domain_filelists_from_config(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, List[Path]]:
    """
    根据 config.yaml 发现并按域划分文件列表：
    - source: Case_1 + Case_2
    - target: Case_3
    """
    cfg = load_yaml(config) if isinstance(config, (str, Path)) else dict(config)
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    root = data_cfg.get("root", ".")
    patterns = data_cfg.get("patterns", ["Case_*.csv"])
    recursive = bool(data_cfg.get("recursive", True))

    files = discover_csv_files(root, patterns=patterns, recursive=recursive)
    source: List[Path] = []
    target: List[Path] = []
    other: List[Path] = []
    for f in files:
        meta = parse_rflymad_filename(f)
        d = meta.get("domain_code", None)
        if d in (1, 2):
            source.append(f)
        elif d == 3:
            target.append(f)
        else:
            other.append(f)
    return {"source": source, "target": target, "other": other}


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )


def split_train_val(
    files: Sequence[Path],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(round(len(files) * val_ratio))
    val = files[:n_val]
    train = files[n_val:]
    return train, val


__all__ = [
    "RflyMADDataset",
    "build_domain_filelists_from_config",
    "discover_csv_files",
    "load_yaml",
    "make_dataloader",
    "normalize_pwm",
    "parse_rflymad_filename",
    "quaternion_to_euler",
    "set_global_seed",
    "sliding_window",
    "split_train_val",
]
