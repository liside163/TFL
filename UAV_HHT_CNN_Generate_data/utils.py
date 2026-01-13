import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _set_matplotlib_chinese_font() -> None:
    # 为什么：默认字体可能不含中文，导致标题标签显示为方块；这里按常见中文字体做降级链
    try:
        import matplotlib as mpl
    except Exception:  # pragma: no cover
        return

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    mpl.rcParams["font.sans-serif"] = candidates
    mpl.rcParams["axes.unicode_minus"] = False


def load_yaml(path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件并处理环境变量占位符
    
    支持格式：
        - ${VAR}           - 使用环境变量 VAR
        - ${VAR:default}   - 使用环境变量 VAR，未设置时使用 default
    
    为什么：把所有超参数集中到配置文件，避免命令行参数导致复现实验困难
    """
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 PyYAML，请先安装：pip install pyyaml") from exc
    
    # 导入路径工具模块处理环境变量
    try:
        from path_utils import process_config_paths
    except ImportError:
        # 如果 path_utils 不存在，直接返回原始配置
        process_config_paths = lambda x: x
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 处理配置中的环境变量占位符 (如 ${DATA_ROOT:/default/path})
    return process_config_paths(config)



def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def set_seed(seed: int) -> None:
    # 为什么：域自适应训练对随机性敏感，固定随机种子便于对比与复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_cfg: str) -> torch.device:
    # 为什么：配置里写 auto，代码自动选择可用 GPU，否则退回 CPU
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)



def build_stft_preprocess(cfg: Dict[str, Any], device: torch.device):
    data_cfg = cfg.get("data", {})
    if not bool(data_cfg.get("stft_on_gpu", False)):
        def _preprocess(x: torch.Tensor) -> torch.Tensor:
            return x.to(device, non_blocking=True)
        return _preprocess

    stft_cfg = cfg["stft"]
    n_fft = int(stft_cfg["n_fft"])
    hop_length = int(stft_cfg["hop_length"])
    win_length = int(stft_cfg["win_length"])
    center = bool(stft_cfg.get("center", False))
    log_mag = bool(stft_cfg.get("log_magnitude", True))
    eps = float(stft_cfg.get("eps", 1e-6))
    resize_cfg = stft_cfg.get("resize", {}) or {}
    resize_enabled = bool(resize_cfg.get("enabled", False))
    out_freq = int(resize_cfg.get("out_freq", 64))
    out_time = int(resize_cfg.get("out_time", 64))
    window = torch.hann_window(win_length, device=device)

    def _preprocess(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        b, c, t = x.shape
        x_flat = x.reshape(b * c, t)
        s = torch.stft(
            x_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            return_complex=True,
        )
        mag = torch.abs(s)
        if log_mag:
            mag = torch.log1p(mag)
        spec = mag.reshape(b, c, mag.shape[-2], mag.shape[-1])
        if resize_enabled:
            spec = F.interpolate(spec, size=(out_freq, out_time), mode="bilinear", align_corners=False)
        mean = spec.mean(dim=(2, 3), keepdim=True)
        std = spec.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        return (spec - mean) / std

    return _preprocess


def setup_logger(log_dir: str, name: str = "run") -> logging.Logger:
    # 为什么：训练与评估要可追溯，把关键指标写到文件里便于复盘
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(Path(log_dir) / "run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


@dataclass
class AverageMeter:
    # 为什么：把 loss/acc 的累计逻辑抽出来，训练循环更清晰
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, extra: Dict[str, Any]) -> None:

    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "extra": extra}
    torch.save(payload, path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("extra", {})


def plot_training_curves(out_path: str, history: Dict[str, list]) -> None:
    # 为什么：把 clf/mmd/total 曲线画出来，能快速判断训练是否收敛与是否过拟合
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return

    _set_matplotlib_chinese_font()
    plt.figure(figsize=(9, 4))
    for key in ["loss_total", "loss_clf", "loss_mmd", "val_loss"]:
        if key in history and len(history[key]) > 0:
            plt.plot(history[key], label=key)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix(out_path: str, y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> None:
    # 为什么：混淆矩阵能直观看到真实飞行数据上各类故障的可分性与混淆模式
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    except Exception:  # pragma: no cover
        return

    _set_matplotlib_chinese_font()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def try_import_umap():
    # 为什么：UMAP 不是标准库依赖，环境缺少时要能优雅跳过
    try:
        import umap
    except Exception:
        return None
    return umap


def plot_umap_domains(out_path: str, z: np.ndarray, y: np.ndarray, domain: np.ndarray, class_names: list, title: str) -> None:
    # 为什么：画 UMAP 看特征是否实现“类可分、域对齐”，这是 MK-MMD 的核心目的
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return

    _set_matplotlib_chinese_font()
    plt.figure(figsize=(8, 6))
    markers = {0: "o", 1: "^"}
    dom_names = {0: "source", 1: "target"}
    for dom_id in [0, 1]:
        m = domain == dom_id
        if not np.any(m):
            continue
        plt.scatter(
            z[m, 0],
            z[m, 1],
            c=y[m],
            s=8,
            marker=markers[dom_id],
            cmap="tab20",
            alpha=0.75,
            label=dom_names[dom_id],
        )
    plt.title(title)
    cb = plt.colorbar()
    cb.set_ticks(list(range(len(class_names))))
    cb.set_ticklabels(class_names)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    # 为什么：UMAP 可视化不需要全量点，抽样能显著降低内存与耗时
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)
