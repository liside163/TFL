from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from etl.config import ExperimentConfig, config_to_dict, load_config
from etl.dataset import build_dataloaders
from etl.detect import build_threshold, detect_faults, fit_error_stats
from etl.ensemble import ensemble_predict
from etl.indexer import build_index
from etl.model import BiLSTMRegressor
from etl.rscore import rscore_table
from etl.train import compute_metrics, predict, select_device, train_model
from etl.transfer import finetune_transfer
from etl.viz import save_error_and_threshold, save_fault_flags, save_pred_vs_true, save_rscore_and_weights


def _set_seed(seed: int, deterministic: bool = True) -> None:
    """
    设置随机种子，尽量保证可复现。

    关键点（中文注释）：
    - Python / NumPy / PyTorch 全部设置
    - deterministic=True 时，开启更严格的确定性（可能略慢）
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # 部分版本/算子不支持，忽略即可
            pass


def _make_run_dir(cfg: ExperimentConfig) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(cfg.output_dir) / cfg.experiment_name / ts
    out.mkdir(parents=True, exist_ok=True)
    return str(out.resolve())


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.no_grad()
def _collect_last_step_xy(loader, device: torch.device, max_batches: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 window 数据中提取用于 R-score 的 (x_last, y_last)。
    """
    xs = []
    ys = []
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device)
        xs.append(x[:, -1, :].detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def run_experiment(config_path: str) -> Dict[str, Any]:
    """
    实验函数入口：可在 notebook 中直接调用：
      run_experiment("rflymad_etl.yaml")

    返回 dict（便于 notebook 展示/序列化）。
    """
    cfg = load_config(config_path)
    _set_seed(cfg.repro.seed, deterministic=bool(cfg.repro.deterministic))

    run_dir = _make_run_dir(cfg)
    _save_json(Path(run_dir) / "config.json", config_to_dict(cfg))

    # 1) 建索引
    index_df = build_index(cfg.data)
    index_path = Path(run_dir) / "index.csv"
    index_df.to_csv(index_path, index=False, encoding="utf-8-sig")

    # 2) 构建数据管线
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        index_df, cfg.data, seed=cfg.repro.seed, features_mode=str(cfg.features_mode)
    )
    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]
    # 关键维度说明：
    # - DataLoader 输出 X: [B, L, F]，y: [B, C]
    # - B 为 batch size，L 为窗口长度，F 为特征维度，C 为目标维度

    # 3) 建模
    input_dim = cfg.model.input_dim or len(feature_cols)
    model = BiLSTMRegressor(
        input_dim=int(input_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        lstm_layers=int(cfg.model.lstm_layers),
        dropout=float(cfg.model.dropout),
        mlp_hidden=int(cfg.model.mlp_hidden),
        output_dim=int(cfg.model.output_dim),
    )
    # 关键维度变化：
    # - 输入序列 [B, L, F] 经过 BiLSTM 变为 [B, L, 2H]
    # - 取最后时间步得到 [B, 2H]
    # - MLP 映射到输出 [B, C]

    # 4) 训练
    train_info = train_model(
        model,
        train_loader,
        val_loader,
        cfg,
        out_dir=run_dir,
        extra_ckpt={"meta": meta},
    )
    # 训练阶段的预测维度始终保持：
    # - 预测 p: [B, C]
    # - 标签 y: [B, C]

    final_ckpt = train_info["best_checkpoint"]

    # 5) 迁移学习（可选）
    transfer_info: Dict[str, Any] = {}
    if bool(cfg.transfer.enable):
        transfer_info = finetune_transfer(
            model=model,
            pretrained_ckpt=final_ckpt,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            out_dir=run_dir,
            extra_ckpt={"meta": meta},
        )
        final_ckpt = transfer_info["transfer_best_checkpoint"]

    # 6) 评估（加载最终 best）
    device = select_device(cfg.train.device)
    payload = torch.load(final_ckpt, map_location="cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)

    yt_train, yp_train = predict(model, train_loader, device)
    yt_val, yp_val = predict(model, val_loader, device)
    yt_test, yp_test = predict(model, test_loader, device)
    # 关键维度变化：批量输出被拼接为全量数组
    # - yt_*/yp_*: [N, C]，N 为样本数（窗口数）

    metrics = {
        "train": compute_metrics(yt_train, yp_train),
        "val": compute_metrics(yt_val, yp_val),
        "test": compute_metrics(yt_test, yp_test),
    }
    _save_json(Path(run_dir) / "metrics" / "final_metrics.json", metrics)

    # 7) 阈值检测（论文公式：(6)(7)(8)）
    # (6) e = p - yhat，这里 p=yt（真实执行器输出），yhat=yp（预测正常输出）
    fd = cfg.fault_detection
    e_train = (yt_train - yp_train).astype(np.float32)

    # 阈值统计来源：源域训练误差 or 目标域 pseudo-normal 误差
    stats = fit_error_stats(
        e_train,
        error_mode=str(fd.error_mode),
        use_abs_error=bool(fd.use_abs_error),
        global_reduce=str(fd.global_reduce),
    )

    if str(fd.threshold_from).lower() == "target_pseudo_normal_error":
        # 如果有迁移学习步骤，尝试读取 transfer 输出的 pseudo-normal 误差统计；否则回退到 source_train_error
        pseudo_report_path = transfer_info.get("transfer_pseudo_report") if isinstance(transfer_info, dict) else None
        if pseudo_report_path and Path(str(pseudo_report_path)).exists():
            try:
                rep = json.loads(Path(str(pseudo_report_path)).read_text(encoding="utf-8"))
                # 约定：若存在 abs_error_after_stats 就用它（窗口级别标签误差分布）
                abs_after = rep.get("abs_error_after_stats")
                if isinstance(abs_after, dict) and "mu" in abs_after and "sigma" in abs_after:
                    # mu/sigma 可能是 list，转回 ndarray
                    mu = np.asarray(abs_after["mu"], dtype=np.float32)
                    sigma = np.asarray(abs_after["sigma"], dtype=np.float32)
                    stats = {
                        "mode": str(fd.error_mode).lower(),
                        "use_abs_error": bool(fd.use_abs_error),
                        "global_reduce": str(fd.global_reduce),
                        "mu": mu if mu.ndim > 0 else float(mu),
                        "sigma": sigma if sigma.ndim > 0 else float(sigma),
                    }
            except Exception:
                pass

    thresholds = build_threshold(stats, alpha=float(fd.alpha))
    det = detect_faults(yt_test, yp_test_used, thresholds)

    # 8) R-score（用于特征解释/后续集成的“打分”）
    #    这里用窗口末端的特征与目标做相关性度量（简单但可用）
    x_last, y_last = _collect_last_step_xy(train_loader, device=device, max_batches=50)
    rtab = rscore_table(x_last, y_last, feature_names=feature_cols, target_names=target_cols)
    rtab_path = Path(run_dir) / "metrics" / "rscore_table.csv"
    rtab.to_csv(rtab_path, index=False, encoding="utf-8-sig")

    # 9) 集成（按你的论文规则实现的 ensemble_predict 需要多个 source->target 模型与对应 R_i）
    #    当前 run_experiment 默认只训练/得到一个模型，因此这里做“安全兜底”：
    #    - 如果只有一个模型，final_pred 就是它本身，weights=[1]
    ensemble_info: Dict[str, Any] = {}
    if bool(cfg.ensemble.enable):
        # 这里没有多源域模型的 R_i 输入，因此仅对“单模型”做兼容输出
        rs = np.array([0.0], dtype=np.float32)
        final_pred, w, _ = ensemble_predict([yp_test], rs, rscore_is_distance=True, two_segment=False)
        ensemble_info = {"r_scores": rs.tolist(), "weights": w.tolist()}
        yp_test_used = final_pred
    else:
        yp_test_used = yp_test

    # 10) 可视化（自动保存到 work_dir/plots）
    if bool(cfg.viz.enable):
        save_pred_vs_true(run_dir, yt_test, yp_test_used, channel_names=target_cols, max_points=int(cfg.viz.max_points))
        save_error_and_threshold(run_dir, det["e"], det["T"], channel_names=target_cols, max_points=int(cfg.viz.max_points))
        save_fault_flags(run_dir, det["fault_flags"], max_points=int(cfg.viz.max_points))
        if ensemble_info.get("weights") is not None:
            save_rscore_and_weights(
                run_dir,
                np.asarray(ensemble_info.get("r_scores", [])),
                np.asarray(ensemble_info.get("weights", [])),
            )

    summary = {
        "run_dir": run_dir,
        "config_path": str(Path(config_path).resolve()),
        "index_csv": str(index_path),
        "checkpoints": {
            "final": final_ckpt,
            "best": train_info["best_checkpoint"],
            "last": train_info["last_checkpoint"],
        },
        "metrics": metrics,
        "thresholds": {
            "alpha": float(thresholds["alpha"]),
            "mode": str(thresholds.get("mode")),
            "use_abs_error": bool(thresholds.get("use_abs_error", True)),
            "mu": thresholds["mu"].tolist() if isinstance(thresholds["mu"], np.ndarray) else float(thresholds["mu"]),
            "sigma": thresholds["sigma"].tolist() if isinstance(thresholds["sigma"], np.ndarray) else float(thresholds["sigma"]),
            "threshold": thresholds["threshold"].tolist()
            if isinstance(thresholds["threshold"], np.ndarray)
            else float(thresholds["threshold"]),
        },
        "fault_rate": float(
            np.mean(
                (np.any(det["fault_flags"], axis=1) if np.asarray(det["fault_flags"]).ndim == 2 else det["fault_flags"]).astype(float)
            )
        ),
        "meta": {
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "splits": meta["splits"],
        },
        "transfer": transfer_info,
        "ensemble": ensemble_info,
    }
    _save_json(Path(run_dir) / "summary.json", summary)
    return summary
