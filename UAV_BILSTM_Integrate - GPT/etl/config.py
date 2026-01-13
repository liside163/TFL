from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: PyYAML. Install with `pip install pyyaml`."
    ) from exc


@dataclass
class DataConfig:
    # 数据路径使用环境变量 (Docker容器中 DATA_ROOT=/data)
    dataset_root: str = field(default_factory=lambda: os.getenv('DATA_ROOT', 'D:\\DL_Learn\\Data\\ProcessData_test'))
    file_glob: str = "**/*.csv"
    time_col: Optional[str] = None
    # 兼容两种特征配置方式：
    # - feature_cols：直接给“真实列名”（旧方式）
    # - scalar_features/vector_features：给“基列名+维度”（新方式，适配 RflyMAD 三种列风格）
    feature_cols: Optional[List[str]] = None
    scalar_features: List[str] = field(default_factory=list)
    vector_features: List[Dict[str, Any]] = field(default_factory=list)  # 形如：[{base: "...", dim: 4, required: true}]
    allow_missing_features: bool = False
    target_vector: Optional[Dict[str, Any]] = None
    target_cols: List[str] = field(default_factory=lambda: ["y1", "y2", "y3", "y4"])
    resample_hz: Optional[float] = None
    window_size: int = 128
    stride: int = 32
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class ModelConfig:
    input_dim: Optional[int] = None
    hidden_dim: int = 128
    lstm_layers: int = 3
    dropout: float = 0.1
    mlp_hidden: int = 128
    output_dim: int = 4


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    device: str = "cuda"  # 自动降级由代码处理
    save_every: int = 1
    early_stop_patience: int = 10


@dataclass
class TransferConfig:
    enable: bool = False
    lr: float = 5e-4
    epochs: int = 10
    freeze_first_n_bilstm: int = 2
    pseudo_normal_enable: bool = False
    pseudo_keep_quantile: float = 0.7
    # 兼容你描述的新字段名（与旧字段等价）
    freeze_bilstm_layers: Optional[int] = None
    keep_low_error_quantile: Optional[float] = None


@dataclass
class EnsembleConfig:
    enable: bool = False
    temperature: float = 1.0


@dataclass
class DetectConfig:
    alpha: float = 3.0  # 阈值：T = mu + alpha * sigma


@dataclass
class FaultDetectionConfig:
    # 阈值系数：T = mu + alpha*sigma
    alpha: float = 3.0
    # 阈值统计来源
    # - source_train_error：用源域训练集误差统计（更稳）
    # - target_pseudo_normal_error：用目标域 pseudo-normal 误差统计
    threshold_from: str = "source_train_error"
    # 误差模式：
    # - per_channel：每个通道独立阈值，输出 fault_flags[T,4]
    # - any_channel：每通道阈值后做 any，输出 fault_flags[T]
    # - global：把 4 通道误差聚合成一个标量，再用全局阈值，输出 fault_flags[T]
    error_mode: str = "per_channel"
    # 是否对误差取绝对值再做统计/阈值（论文常用 |e|）
    use_abs_error: bool = True
    # global 模式下的通道聚合方式
    global_reduce: str = "mean"  # mean|max|l2


@dataclass
class VizConfig:
    enable: bool = True
    max_points: int = 2000


@dataclass
class ReproConfig:
    seed: int = 42
    deterministic: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    fault_detection: FaultDetectionConfig = field(default_factory=FaultDetectionConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    repro: ReproConfig = field(default_factory=ReproConfig)
    # 特征模式开关（用于 HIL/Real 的 pos/vel 取 actual or setpoint，以及是否纳入）
    # - "all"：纳入并按 domain 映射
    # - "no_posvel"：不纳入任何 vehicle_local_position* 相关特征
    features_mode: str = "all"
    output_dir: str = "runs"
    experiment_name: str = "rflymad_etl"


def _require_type(name: str, value: Any, expected: Tuple[type, ...]) -> None:
    if not isinstance(value, expected):
        exp = ", ".join(t.__name__ for t in expected)
        raise TypeError(f"配置项 `{name}` 类型错误：期望 {exp}，实际 {type(value).__name__}")


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(config_path: str) -> ExperimentConfig:
    """
    读取 YAML 配置并进行最基本的类型检查/默认值填充。

    关键点（中文注释）：
    - 不依赖 CLI/argparse，所有行为由 YAML 决定
    - 未提供的字段使用 dataclass 默认值
    - 对明显的类型错误尽早报错，避免训练中途才爆炸
    """
    path = Path(config_path)
    # 统一使用根目录的 rflymad_etl.yaml 作为唯一配置入口，
    # 确保所有超参数都集中在一个文件中修改。
    if path.name == "rflymad_etl.yaml" and path.parent.name == "configs":
        root_path = Path("rflymad_etl.yaml")
        if root_path.exists():
            path = root_path
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("YAML 顶层必须是 dict")

    base = asdict(ExperimentConfig())
    merged = _deep_update(base, raw)

    _require_type("output_dir", merged.get("output_dir"), (str,))
    _require_type("experiment_name", merged.get("experiment_name"), (str,))
    _require_type("data", merged.get("data"), (dict,))
    _require_type("model", merged.get("model"), (dict,))
    _require_type("train", merged.get("train"), (dict,))

    cfg = ExperimentConfig(
        data=DataConfig(**merged["data"]),
        model=ModelConfig(**merged["model"]),
        train=TrainConfig(**merged["train"]),
        transfer=TransferConfig(**merged.get("transfer", {})),
        ensemble=EnsembleConfig(**merged.get("ensemble", {})),
        detect=DetectConfig(**merged.get("detect", {})),
        fault_detection=FaultDetectionConfig(**merged.get("fault_detection", {})),
        viz=VizConfig(**merged.get("viz", {})),
        repro=ReproConfig(**merged.get("repro", {})),
        features_mode=str(merged.get("features_mode", "all")),
        output_dir=merged["output_dir"],
        experiment_name=merged["experiment_name"],
    )

    if not (0.0 < cfg.data.train_ratio < 1.0):
        raise ValueError("data.train_ratio 必须在 (0,1) 内")
    if not (0.0 <= cfg.data.val_ratio < 1.0) or cfg.data.train_ratio + cfg.data.val_ratio >= 1.0:
        raise ValueError("data.val_ratio 必须 >=0 且 train_ratio+val_ratio < 1")
    if cfg.data.window_size <= 1 or cfg.data.stride <= 0:
        raise ValueError("data.window_size 必须 >1 且 data.stride 必须 >0")
    if cfg.model.output_dim <= 0:
        raise ValueError("model.output_dim 必须 > 0")

    return cfg


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    return asdict(cfg)
