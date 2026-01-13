"""
train_evaluate.py
=================
本文件是整个“Ensemble Transfer Learning Based Cross-Domain UAV Actuator Fault Detection”复现工程的主流程脚本。

工作流（严格按用户要求的 5 个步骤组织）：
Step 1: 多源域预训练（源域：Case_1/Case_2）
Step 2: R-score 计算 + 集成权重计算（对应论文公式 (3)(4)(5)）
Step 3: 迁移学习微调（目标域：Case_3 的 Normal 子集）
Step 4: 集成推理 + 残差计算（对应论文式 (5) 的加权平均）
Step 5: 故障诊断（阈值 T=mu+2sigma，对应论文公式 (6)(8)）

重要说明：
- 本复现默认将“故障标签”从文件名 Case 编码解析得到（fault_type==0 视为正常，其余视为故障）。
  这符合论文中“真实域标签稀缺”的设定：训练/微调阶段可只用 Normal 子集，而测试阶段用故障文件评估检测效果。
- 若你的目标域确实完全没有 Y_T（执行器输出）可用，则无法按论文公式 (3) 计算 MIC[X_T^j, Y_T]，
  需要改为无监督相似度（本脚本不覆盖该变体）。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import warnings

from data_loader import (
    RflyMADDataset,
    build_domain_filelists_from_config,
    load_yaml,
    parse_rflymad_filename,
    set_global_seed,
    split_train_val,
    make_dataloader,
)
from models import BiLSTM_Predictor, Ensemble_Predictor, R_Score_Calculator, TransferLearningManager


@dataclass
class RunPaths:
    run_dir: Path
    weights_dir: Path
    figures_dir: Path


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def merge_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    给 config.yaml 填充默认值，避免用户遗漏字段导致脚本崩溃。
    """
    cfg = dict(cfg or {})
    cfg.setdefault("seed", 42)
    # device: "auto" | "cpu" | "cuda"
    # 说明：某些新显卡（例如 sm_120）在旧版本 PyTorch/CUDA wheel 中没有对应 kernel，
    # torch.cuda.is_available() 依然可能为 True，但实际运行会报 "no kernel image"。
    # 因此提供显式 device 配置，并在 auto 模式下做兼容性探测后自动回退到 CPU。
    cfg.setdefault("device", "auto")

    cfg.setdefault("log", {})
    cfg["log"].setdefault("log_dir", "runs")
    cfg["log"].setdefault("run_name", f"rflymad_ensemble_{_now_tag()}")

    cfg.setdefault("output", {})
    cfg["output"].setdefault("weights_dir", "weights")
    cfg["output"].setdefault("figures_dir", "figures")

    cfg.setdefault("task", {})
    cfg["task"].setdefault("pwm_index", 0)  # 如果 columns.pwm 给了多个通道，默认取第 0 个做回归

    cfg.setdefault("labels", {})
    cfg["labels"].setdefault("normal_fault_types", [0])  # fault_type==0 视为 Normal
    # 多分类诊断（11类故障类型）相关配置
    cfg.setdefault("diagnosis", {})
    cfg["diagnosis"].setdefault("enabled", True)
    # 是否把 normal(0) 也作为分类器类别一起训练；通常不需要，因为 normal 由阈值检测决定
    cfg["diagnosis"].setdefault("include_normal_in_classifier", False)
    # 为加速 CPU 训练，可采样部分样本训练诊断分类器
    cfg["diagnosis"].setdefault("max_train_samples", 20000)
    cfg["diagnosis"].setdefault("model", "logreg")  # logreg | knn

    cfg.setdefault("splits", {})
    cfg["splits"].setdefault("source_val_ratio", 0.2)
    cfg["splits"].setdefault("target_val_ratio", 0.2)

    cfg.setdefault("training", {})
    cfg["training"].setdefault("batch_size", 64)
    cfg["training"].setdefault("num_workers", 0)

    cfg["training"].setdefault("pretrain", {})
    cfg["training"]["pretrain"].setdefault("epochs", 10)
    cfg["training"]["pretrain"].setdefault("lr", 1e-3)

    cfg["training"].setdefault("transfer", {})
    cfg["training"]["transfer"].setdefault("epochs", 5)
    cfg["training"]["transfer"].setdefault("lr", 5e-4)
    cfg["training"]["transfer"].setdefault("freeze_bilstm_layers", 2)  # 用户要求冻结前 2 层

    cfg.setdefault("r_score", {})
    cfg["r_score"].setdefault("mine_alpha", 0.6)
    cfg["r_score"].setdefault("mine_c", 15)
    cfg["r_score"].setdefault("weight_mode", "inverse")  # inverse 或 softmax_neg

    cfg.setdefault("detection", {})
    cfg["detection"].setdefault("k_sigma", 2.0)  # T = mu + k*sigma

    # domains.sources 可选：不写就默认 SIL/HIL
    cfg.setdefault("domains", {})
    cfg["domains"].setdefault(
        "sources",
        [
            {"name": "SIL", "domain_code": 1},
            {"name": "HIL", "domain_code": 2},
        ],
    )
    cfg["domains"].setdefault("target", {"name": "Real", "domain_code": 3})
    return cfg


def setup_run(cfg: Dict[str, Any]) -> Tuple[torch.device, SummaryWriter, RunPaths]:
    set_global_seed(int(cfg["seed"]))
    device = resolve_device(cfg)

    log_dir = Path(cfg["log"]["log_dir"]).expanduser().resolve()
    run_name = str(cfg["log"]["run_name"])
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = Path(cfg["output"]["weights_dir"]).expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = Path(cfg["output"]["figures_dir"]).expanduser().resolve() / run_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 保存最终配置，便于论文复现实验（实验可追溯性非常关键）
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    writer = SummaryWriter(log_dir=str(run_dir))
    return device, writer, RunPaths(run_dir=run_dir, weights_dir=weights_dir, figures_dir=figures_dir)


def _cuda_arch_supported() -> bool:
    """
    判断当前 PyTorch CUDA wheel 是否包含本机 GPU 架构的 kernel。
    - 对应用户报错：sm_120 不在当前 torch 支持的 sm 列表里，导致 LSTM/CuDNN 调用报 no kernel image。
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
        want = f"sm_{major}{minor}"
        arch_list = torch.cuda.get_arch_list()  # 例如 ['sm_50','sm_60',...]
        if not arch_list:
            return False
        return want in set(arch_list)
    except Exception:
        return False


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    """
    设备选择策略：
    - cfg.device == "cpu"：强制 CPU
    - cfg.device == "cuda"：强制 CUDA（若不可用则报错）
    - cfg.device == "auto"：优先 CUDA，但若检测到架构不匹配/运行失败则自动回退 CPU
    """
    mode = str(cfg.get("device", "auto")).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求使用 CUDA，但 torch.cuda.is_available() 为 False")
        # 强制模式下也做一次轻量检查，尽早给出可读错误
        if not _cuda_arch_supported():
            major, minor = torch.cuda.get_device_capability()
            raise RuntimeError(
                f"配置要求使用 CUDA，但当前 PyTorch 不支持该 GPU 架构 sm_{major}{minor}；"
                "请安装支持该架构的 PyTorch CUDA 版本，或设置 device: cpu"
            )
        return torch.device("cuda")

    # auto：尽量使用 GPU，但必须保证“能跑”
    if torch.cuda.is_available() and _cuda_arch_supported():
        # 再做一次真正的 CUDA op 探测，避免某些环境 arch_list 误判
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def fault_label_from_meta(meta: Dict[str, Any], normal_fault_types: Sequence[int]) -> int:
    """
    将文件名解析得到的 fault_type 转为二分类标签：
    - 0: Normal
    - 1: Fault
    """
    # DataLoader 的默认 collate 可能把 meta 处理成 dict-of-lists 或其它结构；
    # 这里做一次鲁棒处理：若不是 dict，则尝试从文件名/字符串重新解析。
    if not isinstance(meta, dict):
        try:
            meta = parse_rflymad_filename(str(meta))
        except Exception:
            return 0
    ft = meta.get("fault_type", None)
    if ft is None:
        return 0
    try:
        ft_int = int(ft)
    except Exception:
        return 0
    return 0 if ft_int in set(int(x) for x in normal_fault_types) else 1


def fault_type_from_meta(meta: Any, *, default: int = 0) -> int:
    """
    获取多分类“故障类型”标签（整数）。
    - 优先使用 meta['fault_type']（来自文件名解析）
    - 若 meta 不是 dict，则尝试把它当成文件名再次解析
    """
    if not isinstance(meta, dict):
        try:
            meta = parse_rflymad_filename(str(meta))
        except Exception:
            return int(default)
    ft = meta.get("fault_type", None)
    if ft is None:
        return int(default)
    try:
        return int(ft)
    except Exception:
        return int(default)


def _build_diagnosis_features(x: np.ndarray, y_pred: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """
    构造“故障类型诊断”特征向量（用于 11 类分类器）：
    - 使用窗口最后一帧的状态特征 x_last（对应控制/动力学状态）
    - 加入预测值 y_hat 与残差 |y-y_hat|

    注：论文主要做“故障检测”（残差阈值），并未强制要求多类别诊断。
    这里的多分类属于工程扩展：在检测为 Fault 后进一步区分 fault_type。
    """
    x = np.asarray(x, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1, 1)
    residual = np.asarray(residual, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([x, y_pred, residual], axis=1).astype(np.float32)


def _fit_fault_type_classifier(
    cfg: Dict[str, Any],
    *,
    x_feat: np.ndarray,
    y_fault_type: np.ndarray,
) -> Any:
    """
    训练一个轻量级多分类诊断器（默认 LogisticRegression）。
    - 仅用于“Fault detected”后的类型判别
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
    except Exception as exc:  # pragma: no cover
        raise ImportError("需要安装 scikit-learn：conda install -c conda-forge scikit-learn") from exc

    model_type = str(cfg.get("diagnosis", {}).get("model", "logreg")).lower()
    if model_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        # 多类别逻辑回归：sklearn>=1.5 中 multi_class 参数已弃用，
        # 这里不显式设置 multi_class，交由 sklearn 自动选择（多类时会走 multinomial）。
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(x_feat, y_fault_type)
    return pipe


def _multiclass_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> np.ndarray:
    """
    生成多分类混淆矩阵（labels 指定类别顺序）。
    """
    idx = {int(c): i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if int(t) not in idx or int(p) not in idx:
            continue
        cm[idx[int(t)], idx[int(p)]] += 1
    return cm


def group_source_files(cfg: Dict[str, Any]) -> Dict[str, List[Path]]:
    """
    将 Case_1/Case_2 文件按源域划分。
    """
    files_by_domain = build_domain_filelists_from_config(cfg)
    sources = cfg["domains"]["sources"]
    out: Dict[str, List[Path]] = {}
    for s in sources:
        name = str(s.get("name"))
        code = int(s.get("domain_code"))
        out[name] = []
        for f in files_by_domain["source"]:
            meta = parse_rflymad_filename(f)
            if meta.get("domain_code") == code:
                out[name].append(f)
        out[name] = sorted(out[name])
    return out


def split_target_files(
    cfg: Dict[str, Any],
) -> Tuple[List[Path], List[Path]]:
    """
    目标域（Case_3）文件划分为：
    - normal_files：用于微调训练/阈值估计（仅使用正常数据，符合故障诊断范式）
    - test_files：用于最终检测评估（包含故障与正常）
    """
    files_by_domain = build_domain_filelists_from_config(cfg)
    target_files = files_by_domain["target"]
    normal_files: List[Path] = []
    test_files: List[Path] = []

    def _file_has_fault(path: Path) -> bool:
        """
        以 CSV 内的 UAVState_data_fault_state 为准：
        - 全 0：该文件属于 Normal（适合用于阈值估计与微调）
        - 出现过 1：该文件包含故障过程（适合作为测试）
        """
        try:
            import pandas as pd

            df = pd.read_csv(path, usecols=["UAVState_data_fault_state"])
            col = df["UAVState_data_fault_state"].to_numpy()
            col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
            return bool(np.any(col.astype(np.int64) != 0))
        except Exception:
            # 兜底：如果缺该列或读取失败，退化到文件名 fault_type（避免流程中断）
            meta = parse_rflymad_filename(path)
            return bool(meta.get("fault_type", 0) not in set(cfg["labels"]["normal_fault_types"]))

    for f in target_files:
        has_fault = _file_has_fault(f)
        if not has_fault:
            normal_files.append(f)
        test_files.append(f)
    return sorted(normal_files), sorted(test_files)


def _select_pwm_channel(y: torch.Tensor, pwm_index: int) -> torch.Tensor:
    """
    将 (B, C) 或 (B, T, C) 的 y 选出单通道 (B,1) 或 (B,T,1)，用于单执行器回归。
    """
    if y.ndim == 1:
        return y.view(-1, 1)
    if y.ndim == 2:
        if y.size(1) == 1:
            return y
        return y[:, int(pwm_index)].contiguous().view(-1, 1)
    if y.ndim == 3:
        if y.size(2) == 1:
            return y
        return y[:, :, int(pwm_index)].contiguous().unsqueeze(-1)
    raise ValueError(f"不支持的 y shape={tuple(y.shape)}")


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    device: torch.device,
    writer: SummaryWriter,
    tag: str,
    epochs: int,
    lr: float,
    pwm_index: int,
) -> Dict[str, List[float]]:
    """
    通用训练函数：用于源域预训练与目标域微调。
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(lr))

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            x, y = batch[0], batch[1]
            x = x.to(device).float()
            y = _select_pwm_channel(y.to(device).float(), pwm_index)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x)

            # 若 DataLoader 返回 y 是 seq-to-seq（B,T,1），而模型输出 (B,1)，则取最后一帧对齐
            if y.ndim > y_hat.ndim:
                y = y[:, -1, :]

            loss = criterion(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            total += float(loss.detach().cpu().item()) * x.size(0)
            n += int(x.size(0))

        train_loss = total / max(1, n)
        history["train_loss"].append(train_loss)
        writer.add_scalar(f"{tag}/train_mse", train_loss, epoch)

        if val_loader is not None:
            model.eval()
            v_total = 0.0
            v_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0], batch[1]
                    x = x.to(device).float()
                    y = _select_pwm_channel(y.to(device).float(), pwm_index)
                    y_hat = model(x)
                    if y.ndim > y_hat.ndim:
                        y = y[:, -1, :]
                    v_loss = criterion(y_hat, y)
                    v_total += float(v_loss.detach().cpu().item()) * x.size(0)
                    v_n += int(x.size(0))
            val_loss = v_total / max(1, v_n)
            history["val_loss"].append(val_loss)
            writer.add_scalar(f"{tag}/val_mse", val_loss, epoch)

    return history


def train_source_domains(
    cfg: Dict[str, Any],
    source_files: Dict[str, List[Path]],
    *,
    device: torch.device,
    writer: SummaryWriter,
    paths: RunPaths,
) -> Dict[str, Path]:
    """
    Step 1：对每个源域分别从头训练一个 BiLSTM_Predictor，并保存权重。
    """
    out_paths: Dict[str, Path] = {}
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"]["num_workers"])
    pre = cfg["training"]["pretrain"]
    pwm_index = int(cfg["task"]["pwm_index"])

    for name, files in source_files.items():
        if not files:
            raise RuntimeError(f"源域 {name} 没有找到任何 CSV 文件")
        train_f, val_f = split_train_val(files, val_ratio=float(cfg["splits"]["source_val_ratio"]), seed=int(cfg["seed"]))

        ds_train = RflyMADDataset(train_f, config=cfg, mode="labeled", return_meta=False)
        ds_val = RflyMADDataset(val_f, config=cfg, mode="labeled", scaler_x=ds_train.scaler_x, return_meta=False)

        train_loader = make_dataloader(
            ds_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda")
        )
        val_loader = make_dataloader(
            ds_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda")
        )

        input_dim = int(ds_train[0][0].shape[-1])
        model = BiLSTM_Predictor(input_dim=input_dim, output_dim=1)

        train_one_model(
            model,
            train_loader,
            val_loader,
            device=device,
            writer=writer,
            tag=f"source_pretrain/{name}",
            epochs=int(pre["epochs"]),
            lr=float(pre["lr"]),
            pwm_index=pwm_index,
        )

        save_path = paths.weights_dir / f"source_{name}.pth"
        torch.save(model.state_dict(), save_path)
        out_paths[name] = save_path
        writer.add_text("artifacts/source_weights", f"{name}: {save_path}", 0)

    return out_paths


def _first_batch_xy(loader: DataLoader, *, device: torch.device, pwm_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    取一个 batch 作为 R-score 计算所需的 (X,Y) 样本。
    """
    batch = next(iter(loader))
    x, y = batch[0], batch[1]
    x = x.to(device).float()
    y = _select_pwm_channel(y.to(device).float(), pwm_index)
    # 关键：我们的 Dataset 通常使用 sliding window + target_mode='last'，
    # 此时 y 是“窗口最后一帧”的输出 (B,1)，而 x 是整段序列 (B,T,F)。
    # 若直接把 x 展平到 (B*T,F) 会导致 x 与 y 样本数不一致，从而 MIC 计算报错。
    # 因此这里对齐为“同一时间点”的统计：取窗口最后一帧的特征 x_last 与 y 对齐，形状 (B,F)/(B,1)。
    if x.ndim == 3 and y.ndim == 2:
        x = x[:, -1, :]
    elif x.ndim == 3 and y.ndim == 3:
        x = x[:, -1, :]
        y = y[:, -1, :]
    return x.detach().cpu(), y.detach().cpu()


def calculate_r_scores_and_weights(
    cfg: Dict[str, Any],
    source_files: Dict[str, List[Path]],
    target_normal_files: List[Path],
    *,
    device: torch.device,
    writer: SummaryWriter,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Step 2：计算每个源域的 R_i，并据此计算集成权重 W_i。
    """
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"]["num_workers"])
    pwm_index = int(cfg["task"]["pwm_index"])

    if not target_normal_files:
        raise RuntimeError("目标域 normal_files 为空：无法用于 R-score/阈值估计/微调")

    # 目标域取一个 batch（建议用 Normal 子集，避免故障样本影响“相似性”度量）
    ds_t = RflyMADDataset(target_normal_files, config=cfg, mode="unlabeled", return_meta=False)
    loader_t = make_dataloader(ds_t, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda"))
    x_t, y_t = _first_batch_xy(loader_t, device=device, pwm_index=pwm_index)

    calc = R_Score_Calculator(mine_alpha=float(cfg["r_score"]["mine_alpha"]), mine_c=int(cfg["r_score"]["mine_c"]))

    r_scores: Dict[str, float] = {}
    for name, files in source_files.items():
        ds_s = RflyMADDataset(files, config=cfg, mode="labeled", return_meta=False)
        loader_s = make_dataloader(
            ds_s, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda")
        )
        x_s, y_s = _first_batch_xy(loader_s, device=device, pwm_index=pwm_index)
        r = calc.r_score(x_s, y_s, x_t, y_t)  # 对应论文公式 (3)
        r_scores[name] = float(r)
        writer.add_scalar(f"r_score/{name}", float(r), 0)

    # 权重：论文文字倾向“R 越小越相似”，因此这里默认采用 inverse 或 softmax(-R)
    mode = str(cfg["r_score"]["weight_mode"]).lower()
    names = list(r_scores.keys())
    r_vec = np.asarray([r_scores[n] for n in names], dtype=np.float64)

    if mode == "softmax_neg":
        # W_i = softmax(-R_i)：R 越小 -> -R 越大 -> 权重越大
        z = -r_vec
        z = z - np.max(z)
        w = np.exp(z) / (np.sum(np.exp(z)) + 1e-12)
    else:
        # W_i ∝ 1/(|R_i|+eps)：相比 1/R 更稳定（避免 R 接近 0 产生爆炸）
        w = 1.0 / (np.abs(r_vec) + 1e-8)
        w = w / (np.sum(w) + 1e-12)

    weights = {n: float(w[i]) for i, n in enumerate(names)}
    for n, wi in weights.items():
        writer.add_scalar(f"ensemble_weight/{n}", wi, 0)

    return r_scores, weights


def transfer_finetune(
    cfg: Dict[str, Any],
    source_weight_paths: Dict[str, Path],
    target_normal_files: List[Path],
    *,
    device: torch.device,
    writer: SummaryWriter,
    paths: RunPaths,
) -> Dict[str, Path]:
    """
    Step 3：对每个源域模型做迁移微调（目标域 Normal 子集），保存 transfer 权重。
    """
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"]["num_workers"])
    pwm_index = int(cfg["task"]["pwm_index"])
    tr = cfg["training"]["transfer"]

    train_f, val_f = split_train_val(target_normal_files, val_ratio=float(cfg["splits"]["target_val_ratio"]), seed=int(cfg["seed"]))
    ds_train = RflyMADDataset(train_f, config=cfg, mode="unlabeled", return_meta=False)
    ds_val = RflyMADDataset(val_f, config=cfg, mode="unlabeled", scaler_x=ds_train.scaler_x, return_meta=False)
    train_loader = make_dataloader(
        ds_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda")
    )
    val_loader = make_dataloader(
        ds_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda")
    )

    input_dim = int(ds_train[0][0].shape[-1])

    out_paths: Dict[str, Path] = {}
    for name, src_path in source_weight_paths.items():
        model = BiLSTM_Predictor(input_dim=input_dim, output_dim=1)
        model.load_state_dict(torch.load(src_path, map_location="cpu"))

        # 冻结前 2 层 BiLSTM（用户明确要求）；注意：本实现只有 2 层，因此将只微调 Dense head
        TransferLearningManager.freeze_layers(model, num_layers_to_freeze=int(tr["freeze_bilstm_layers"]))

        train_one_model(
            model,
            train_loader,
            val_loader,
            device=device,
            writer=writer,
            tag=f"target_finetune/{name}",
            epochs=int(tr["epochs"]),
            lr=float(tr["lr"]),
            pwm_index=pwm_index,
        )

        save_path = paths.weights_dir / f"transfer_{name}.pth"
        torch.save(model.state_dict(), save_path)
        out_paths[name] = save_path
        writer.add_text("artifacts/transfer_weights", f"{name}: {save_path}", 0)

    return out_paths


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    pwm_index: int,
    return_meta: bool = False,
    return_fault_state: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
    """
    在一个 DataLoader 上做推理，返回 y_true / y_pred（都为 (N,1)）。
    """
    model.eval()
    model.to(device)
    ys: List[np.ndarray] = []
    yhs: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    faults: List[np.ndarray] = []

    def _decollate_meta(meta_batch: Any, batch_size: int) -> List[Dict[str, Any]]:
        """
        将 DataLoader 默认 collate 后的 meta 还原成 List[Dict]。
        - 若 Dataset 返回 dict，默认 collate 可能变为 Dict[str, List/Tensor]
        - 若 Dataset 返回 List[Dict]，则直接返回
        """
        if meta_batch is None:
            return []
        if isinstance(meta_batch, list):
            # 可能已经是 List[Dict]
            if meta_batch and isinstance(meta_batch[0], dict):
                return meta_batch  # type: ignore[return-value]
            # 也可能是 list[str]（例如误操作），尽量兜底
            return [parse_rflymad_filename(str(x)) for x in meta_batch]

        if isinstance(meta_batch, dict):
            keys = list(meta_batch.keys())
            values_list: Dict[str, List[Any]] = {}
            for k in keys:
                v = meta_batch[k]
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().tolist()
                elif isinstance(v, np.ndarray):
                    v = v.tolist()
                elif not isinstance(v, list):
                    v = [v] * batch_size
                # 标量类型转 python 原生，避免 json/打印麻烦
                out_v = []
                for item in v:
                    if isinstance(item, (np.generic,)):
                        item = item.item()
                    out_v.append(item)
                values_list[k] = out_v

            out: List[Dict[str, Any]] = []
            for i in range(batch_size):
                d: Dict[str, Any] = {}
                for k in keys:
                    seq = values_list.get(k, [])
                    d[k] = seq[i] if i < len(seq) else None
                out.append(d)
            return out

        # 其它类型：当作 filename/标量兜底
        return [parse_rflymad_filename(str(meta_batch)) for _ in range(batch_size)]

    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("DataLoader batch 必须至少包含 (x,y)")

        x, y = batch[0], batch[1]
        fault_state = None
        meta = None

        # 支持 (x,y,meta) / (x,y,fault_state) / (x,y,fault_state,meta)
        if len(batch) == 3:
            if isinstance(batch[2], dict) or isinstance(batch[2], list):
                meta = batch[2]
            else:
                fault_state = batch[2]
        elif len(batch) >= 4:
            fault_state = batch[2]
            meta = batch[3]

        if return_meta and meta is not None:
            metas.extend(_decollate_meta(meta, int(x.size(0))))
        if return_fault_state and fault_state is not None:
            if isinstance(fault_state, torch.Tensor):
                faults.append(fault_state.detach().cpu().numpy().reshape(-1))
            else:
                faults.append(np.asarray(fault_state).reshape(-1))

        x = x.to(device).float()
        y = _select_pwm_channel(y.to(device).float(), pwm_index)
        y_hat = model(x)
        if y.ndim > y_hat.ndim:
            y = y[:, -1, :]
        ys.append(y.detach().cpu().numpy())
        yhs.append(y_hat.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yhs, axis=0)
    fault_arr = np.concatenate(faults, axis=0).astype(np.int32) if (return_fault_state and faults) else None
    return y_true, y_pred, fault_arr, metas if return_meta else None


def evaluate_ensemble_and_detect(
    cfg: Dict[str, Any],
    transfer_weight_paths: Dict[str, Path],
    weights: Dict[str, float],
    target_normal_files: List[Path],
    target_test_files: List[Path],
    *,
    device: torch.device,
    writer: SummaryWriter,
    paths: RunPaths,
) -> Dict[str, float]:
    """
    Step 4 + Step 5：
    - 集成推理得到 y_final
    - 计算残差 e=|y-y_hat|
    - 用 Normal 验证集估计阈值 T=mu+k*sigma，并在 Test 上做故障判定与指标计算
    - 生成论文 Fig.7 类似图像并保存
    """
    import matplotlib.pyplot as plt

    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"]["num_workers"])
    pwm_index = int(cfg["task"]["pwm_index"])
    k_sigma = float(cfg["detection"]["k_sigma"])

    # 1) 构造数据集：Normal 用于阈值估计，Test 用于检测评估
    ds_normal = RflyMADDataset(
        target_normal_files, config=cfg, mode="labeled", return_meta=True, return_fault_state=True
    )
    ds_test = RflyMADDataset(
        target_test_files,
        config=cfg,
        mode="labeled",
        return_meta=True,
        return_fault_state=True,
        scaler_x=ds_normal.scaler_x,
    )
    loader_normal = make_dataloader(
        ds_normal, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda")
    )
    loader_test = make_dataloader(
        ds_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda")
    )

    input_dim = int(ds_test[0][0].shape[-1])

    # 2) 加载迁移后的模型并组装集成器
    model_list: List[nn.Module] = []
    weight_list: List[float] = []
    for name, w_path in transfer_weight_paths.items():
        m = BiLSTM_Predictor(input_dim=input_dim, output_dim=1)
        m.load_state_dict(torch.load(w_path, map_location="cpu"))
        model_list.append(m)
        weight_list.append(float(weights.get(name, 0.0)))

    if np.sum(weight_list) <= 0:
        weight_list = [1.0 / len(model_list)] * len(model_list)
    else:
        s = float(np.sum(weight_list))
        weight_list = [float(x) / s for x in weight_list]

    ensemble = Ensemble_Predictor(model_list, weights=weight_list, weight_mode="similarity")

    # 3) Normal 残差分布 -> 阈值（对应论文公式 (6)）
    y_n, yhat_n, fs_n, meta_n = predict_dataset(
        ensemble, loader_normal, device=device, pwm_index=pwm_index, return_meta=True, return_fault_state=True
    )
    e_n = np.abs(y_n - yhat_n).reshape(-1)
    mu = float(np.mean(e_n))
    sigma = float(np.std(e_n))
    threshold = mu + k_sigma * sigma
    writer.add_scalar("detection/normal_mu", mu, 0)
    writer.add_scalar("detection/normal_sigma", sigma, 0)
    writer.add_scalar("detection/threshold", threshold, 0)

    # 4) Test 检测与指标（先做二分类检测）
    y_t, yhat_t, fs_t, meta_t = predict_dataset(
        ensemble, loader_test, device=device, pwm_index=pwm_index, return_meta=True, return_fault_state=True
    )
    e_t = np.abs(y_t - yhat_t).reshape(-1)
    pred_fault = (e_t > threshold).astype(np.int32)  # 对应论文公式 (8)
    if fs_t is None:
        raise RuntimeError("未获取到 fault_state 标签：请确认 CSV 中存在 UAVState_data_fault_state 列")
    gt_fault = fs_t.reshape(-1).astype(np.int32)

    if gt_fault.size != pred_fault.size:
        # 理论上应一一对应；若 Dataset 返回 meta 结构不一致，则给出明确错误方便排查
        raise RuntimeError(f"meta 与预测样本数量不一致：gt={gt_fault.size}, pred={pred_fault.size}")

    tp = int(np.sum((pred_fault == 1) & (gt_fault == 1)))
    tn = int(np.sum((pred_fault == 0) & (gt_fault == 0)))
    fp = int(np.sum((pred_fault == 1) & (gt_fault == 0)))
    fn = int(np.sum((pred_fault == 0) & (gt_fault == 1)))

    acc = float((tp + tn) / max(1, (tp + tn + fp + fn)))
    tpr = float(tp / max(1, (tp + fn)))
    fpr = float(fp / max(1, (fp + tn)))

    writer.add_scalar("metrics/accuracy", acc, 0)
    writer.add_scalar("metrics/tpr", tpr, 0)
    writer.add_scalar("metrics/fpr", fpr, 0)

    # 4.1) 二分类混淆矩阵输出（Normal=0, Fault=1）
    # 按二分类惯例矩阵为：
    #   [[TN, FP],
    #    [FN, TP]]
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    cm_dict = {
        "labels": ["Normal(0)", "Fault(1)"],
        "matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    (paths.run_dir / "confusion_matrix.json").write_text(
        json.dumps(cm_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 保存混淆矩阵图（便于论文/报告直接引用）
    fig_cm, ax_cm = plt.subplots(1, 1, figsize=(5.5, 5.0))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title("Confusion Matrix (Target Test)")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Ground Truth")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Normal", "Fault"])
    ax_cm.set_yticklabels(["Normal", "Fault"])
    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j, i, str(int(v)), ha="center", va="center", color="black")
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    fig_cm.tight_layout()
    cm_path = paths.figures_dir / "confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)
    writer.add_text("artifacts/confusion_matrix", str(cm_path), 0)

    # 4.2) 多分类“故障类型”混淆矩阵（你的数据集有 11 种 fault_type）
    diag_cfg = cfg.get("diagnosis", {}) if isinstance(cfg.get("diagnosis", {}), dict) else {}
    if bool(diag_cfg.get("enabled", True)):
        try:
            # 1) 组织训练数据：使用源域（Case_1/Case_2）的“类型标签”训练一个诊断器
            #    注意：这里仅训练“Fault -> fault_type”的分类器；Normal 由阈值决定。
            source_files = group_source_files(cfg)
            x_train_list: List[np.ndarray] = []
            y_train_list: List[np.ndarray] = []
            max_samples = int(diag_cfg.get("max_train_samples", 20000))
            include_normal = bool(diag_cfg.get("include_normal_in_classifier", False))

            for domain_name, files in source_files.items():
                if not files:
                    continue
                ds_src = RflyMADDataset(
                    files, config=cfg, mode="labeled", return_meta=True, return_fault_state=True, scaler_x=ds_normal.scaler_x
                )
                dl_src = make_dataloader(ds_src, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda"))
                # 取一部分 batch 加速（CPU 上全量会较慢）
                n_collected = 0
                for batch in dl_src:
                    # (x,y,fault_state,meta)
                    x_b, y_b, fs_b, meta_b = batch[0], batch[1], batch[2], batch[3]
                    x_b = x_b.to(device).float()
                    # 取最后一帧特征用于诊断（与 r-score 一致的“对齐”逻辑）
                    x_last = x_b[:, -1, :].detach().cpu().numpy()
                    y_b = _select_pwm_channel(y_b.to(device).float(), pwm_index)
                    y_hat_b = ensemble(x_b).detach().cpu().numpy()
                    if y_b.ndim > 2:
                        y_b = y_b[:, -1, :]
                    y_true_b = y_b.detach().cpu().numpy()
                    res_b = np.abs(y_true_b - y_hat_b).reshape(-1)

                    meta_list = []
                    # 复用 predict_dataset 的解包策略：这里 batch[2] 已经是 collate 后的结构
                    # 为简洁起见，借用 fault_type_from_meta 的兜底解析
                    if isinstance(meta_b, list):
                        meta_list = meta_b
                    elif isinstance(meta_b, dict):
                        # dict-of-lists -> list-of-dicts
                        keys = list(meta_b.keys())
                        tmp = []
                        for i in range(int(x_b.size(0))):
                            d = {}
                            for k in keys:
                                v = meta_b[k]
                                if isinstance(v, torch.Tensor):
                                    v = v.detach().cpu().tolist()
                                if isinstance(v, list):
                                    d[k] = v[i] if i < len(v) else None
                                else:
                                    d[k] = v
                            tmp.append(d)
                        meta_list = tmp
                    else:
                        meta_list = [meta_b] * int(x_b.size(0))

                    ft = np.asarray([fault_type_from_meta(m, default=0) for m in meta_list], dtype=np.int32)
                    fsb = fs_b.detach().cpu().numpy().reshape(-1).astype(np.int32)
                    if not include_normal:
                        # 只用故障发生时刻（fault_state==1）来训练“故障类型”分类器
                        mask = fsb == 1
                        if np.any(mask):
                            x_last = x_last[mask]
                            y_hat_b = y_hat_b[mask]
                            res_b = res_b[mask]
                            ft = ft[mask]
                        else:
                            continue

                    feat_b = _build_diagnosis_features(x_last, y_hat_b.reshape(-1), res_b)
                    x_train_list.append(feat_b)
                    y_train_list.append(ft)
                    n_collected += feat_b.shape[0]
                    if n_collected >= max_samples:
                        break

            if x_train_list and y_train_list:
                x_train = np.concatenate(x_train_list, axis=0)
                y_train = np.concatenate(y_train_list, axis=0)
                # 2) 训练诊断分类器
                clf = _fit_fault_type_classifier(cfg, x_feat=x_train, y_fault_type=y_train)

                # 3) 在目标域 Test 上预测 fault_type：未检测到故障 -> 预测为 0（Normal）
                #    detected fault -> 用分类器在 11 类中判别类型
                x_last_test = ds_test._x[:, -1, :].detach().cpu().numpy()  # (N,F)，使用已缓存的张量
                feat_test = _build_diagnosis_features(x_last_test, yhat_t.reshape(-1), e_t.reshape(-1))
                pred_ft = np.zeros((feat_test.shape[0],), dtype=np.int32)
                detected_idx = np.where(pred_fault.reshape(-1) == 1)[0]
                if detected_idx.size > 0:
                    pred_ft_detected = clf.predict(feat_test[detected_idx])
                    pred_ft[detected_idx] = pred_ft_detected.astype(np.int32)

                # GT fault_type：只有 fault_state==1 才有意义；否则强制为 0（Normal）
                gt_ft_raw = np.asarray([fault_type_from_meta(m, default=0) for m in (meta_t or [])], dtype=np.int32)
                gt_ft = np.where(gt_fault.reshape(-1) == 1, gt_ft_raw, 0).astype(np.int32)

                # 类别集合：0 + 所有出现过的 fault_type（保证 11 类在矩阵里）
                classes = sorted(set(gt_ft.tolist()) | set(pred_ft.tolist()))
                if 0 not in classes:
                    classes = [0] + classes
                cm_mc = _multiclass_confusion_matrix(gt_ft, pred_ft, labels=classes)

                mc_dict = {
                    "labels": classes,
                    "matrix": cm_mc.tolist(),
                    "note": "多分类混淆矩阵：0=Normal，其它为 fault_type（来自文件名解析）",
                }
                (paths.run_dir / "confusion_matrix_multiclass.json").write_text(
                    json.dumps(mc_dict, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                # 画多分类混淆矩阵（类别多时图会更大）
                fig_m, ax_m = plt.subplots(1, 1, figsize=(8.5, 7.5))
                im2 = ax_m.imshow(cm_mc, cmap="Blues")
                ax_m.set_title("Multi-class Confusion Matrix (Fault Type)")
                ax_m.set_xlabel("Predicted fault_type")
                ax_m.set_ylabel("Ground Truth fault_type")
                ax_m.set_xticks(range(len(classes)))
                ax_m.set_yticks(range(len(classes)))
                ax_m.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
                ax_m.set_yticklabels([str(c) for c in classes])
                # 只在矩阵不太大时标数值，避免图太密
                if len(classes) <= 20:
                    for (i, j), v in np.ndenumerate(cm_mc):
                        ax_m.text(j, i, str(int(v)), ha="center", va="center", fontsize=7, color="black")
                fig_m.colorbar(im2, ax=ax_m, fraction=0.046, pad=0.04)
                fig_m.tight_layout()
                cm_mc_path = paths.figures_dir / "confusion_matrix_multiclass.png"
                fig_m.savefig(cm_mc_path, dpi=150)
                plt.close(fig_m)
                writer.add_text("artifacts/confusion_matrix_multiclass", str(cm_mc_path), 0)
            else:
                warnings.warn("多分类诊断器训练数据为空：请确认源域文件名包含 fault_type 或启用 include_normal。")
        except Exception as exc:
            warnings.warn(f"多分类故障类型混淆矩阵生成失败：{exc}")

    # 5) 画图（Fig.7 风格）：True vs Pred，以及 Residual vs Threshold
    n_plot = min(2000, y_t.shape[0])
    xs = np.arange(n_plot)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(xs, y_t[:n_plot].reshape(-1), label="True", linewidth=1.2)
    ax[0].plot(xs, yhat_t[:n_plot].reshape(-1), label="Pred(Ensemble)", linewidth=1.2)
    ax[0].set_title("True Value vs Predicted Value (Actuator PWM normalized)")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(xs, e_t[:n_plot], label="Residual |y-ŷ|", linewidth=1.2)
    ax[1].axhline(threshold, color="r", linestyle="--", label=f"Threshold T=μ+{k_sigma}σ")
    ax[1].set_title("Residual vs Threshold (Fault Detection)")
    ax[1].legend(loc="best")
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xlabel("Sample Index")

    fig.tight_layout()
    fig_path = paths.figures_dir / "fig7_like_plot.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    writer.add_text("artifacts/figures", str(fig_path), 0)

    return {
        "threshold": threshold,
        "mu": mu,
        "sigma": sigma,
        "accuracy": acc,
        "tpr": tpr,
        "fpr": fpr,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = merge_defaults(load_yaml(args.config))

    device, writer, paths = setup_run(cfg)
    writer.add_text("env/device", str(device), 0)

    # Step 0: 文件发现与划分
    source_files = group_source_files(cfg)
    target_normal_files, target_test_files = split_target_files(cfg)

    # Step 1: 源域预训练
    source_weight_paths = train_source_domains(cfg, source_files, device=device, writer=writer, paths=paths)

    # Step 2: R-score + 权重
    r_scores, weights = calculate_r_scores_and_weights(
        cfg, source_files, target_normal_files, device=device, writer=writer
    )
    (paths.run_dir / "r_scores.json").write_text(json.dumps(r_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths.run_dir / "ensemble_weights.json").write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 3: 迁移微调
    transfer_weight_paths = transfer_finetune(
        cfg, source_weight_paths, target_normal_files, device=device, writer=writer, paths=paths
    )

    # Step 4+5: 集成推理 + 故障检测评估 + 画图
    metrics = evaluate_ensemble_and_detect(
        cfg,
        transfer_weight_paths,
        weights,
        target_normal_files,
        target_test_files,
        device=device,
        writer=writer,
        paths=paths,
    )
    (paths.run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    writer.close()


if __name__ == "__main__":
    main()
