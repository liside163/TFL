from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import build_stft_preprocess, plot_confusion_matrix, plot_umap_domains, sample_indices, try_import_umap


@torch.no_grad()
def predict_all(model: torch.nn.Module, loader: DataLoader, device: torch.device, preprocess) -> Tuple[np.ndarray, np.ndarray]:
    # 为什么：对真实飞行数据做全量推理，才能画混淆矩阵并验证整体效果
    model.eval()
    y_true = []
    y_pred = []
    for x, y, _domain in loader:
        x = preprocess(x)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.append(pred)
        y_true.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)


@torch.no_grad()
def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device, preprocess) -> Tuple[np.ndarray, np.ndarray]:
    # 为什么：UMAP 需要特征向量与标签（用于着色，来观察“类聚合 + 域对齐”）
    model.eval()
    feats = []
    ys = []
    for x, y, _domain in loader:
        x = preprocess(x)
        _logits, z = model(x, return_feat=True)
        feats.append(z.cpu().numpy())
        ys.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(feats, axis=0), np.concatenate(ys, axis=0)


def _umap_plot_for_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    source_loader: DataLoader,
    target_loader: DataLoader,
    device: torch.device,
    preprocess,
    out_path: str,
    cfg: Dict,
    class_names: list,
    seed: int,
    title: str,
    logger,
) -> None:
    # 对比“训练前/训练后”的特征分布，验证域对齐效果
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.to(device)

    umap_mod = try_import_umap()
    if umap_mod is None:
        logger.info("环境缺少 umap-learn，跳过UMAP可视化")
        return

    z_s, y_s = extract_features(model, source_loader, device, preprocess)
    z_t, y_t = extract_features(model, target_loader, device, preprocess)

    max_n = int(cfg["eval"].get("max_umap_points_per_domain", 4000))
    idx_s = sample_indices(z_s.shape[0], max_n, seed)
    idx_t = sample_indices(z_t.shape[0], max_n, seed + 1)

    z = np.concatenate([z_s[idx_s], z_t[idx_t]], axis=0)
    y = np.concatenate([y_s[idx_s], y_t[idx_t]], axis=0)
    domain = np.concatenate(
        [np.zeros(idx_s.shape[0], dtype=np.int64), np.ones(idx_t.shape[0], dtype=np.int64)],
        axis=0,
    )

    reducer = umap_mod.UMAP(
        n_neighbors=int(cfg["eval"].get("umap_n_neighbors", 30)),
        min_dist=float(cfg["eval"].get("umap_min_dist", 0.1)),
        metric=str(cfg["eval"].get("umap_metric", "euclidean")),
        # 设置 random_state 时 UMAP 会强制 n_jobs=1；显式指定可避免警告
        n_jobs=int(cfg["eval"].get("umap_n_jobs", 1)),
        random_state=seed,
    )
    z2 = reducer.fit_transform(z)
    plot_umap_domains(out_path, z2, y, domain, class_names, title=title)

def run_analysis(
    model: torch.nn.Module,
    init_ckpt_path: Optional[str],
    best_ckpt_path: str,
    source_vis_loader: DataLoader,
    target_vis_loader: DataLoader,
    real_eval_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    cfg: Dict,
    class_names: list,
    logger,
) -> None:
    # 为什么：训练完成后输出曲线、混淆矩阵/UMAP，让结果对齐论文并便于写复现报告
    payload = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    preprocess = build_stft_preprocess(cfg, device)
    y_true, y_pred = predict_all(model, real_eval_loader, device, preprocess)

    acc = float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0
    logger.info(f"real_eval_acc(仅用于验证)={acc:.4f} n={y_true.size}")
    plot_confusion_matrix(str(Path(out_dir) / "confusion_matrix_real.png"), y_true, y_pred, class_names)

    seed = int(cfg["project"].get("seed", 42))
    if init_ckpt_path is not None and Path(init_ckpt_path).exists():
        _umap_plot_for_checkpoint(
            model=model,
            ckpt_path=init_ckpt_path,
            source_loader=source_vis_loader,
            target_loader=target_vis_loader,
            device=device,
            preprocess=preprocess,
            out_path=str(Path(out_dir) / "umap_features_before.png"),
            cfg=cfg,
            class_names=class_names,
            seed=seed,
            title="UMAP(训练前): 颜色=类别, 形状=域",
            logger=logger,
        )

    _umap_plot_for_checkpoint(
        model=model,
        ckpt_path=best_ckpt_path,
        source_loader=source_vis_loader,
        target_loader=target_vis_loader,
        device=device,
        preprocess=preprocess,
        out_path=str(Path(out_dir) / "umap_features_after.png"),
        cfg=cfg,
        class_names=class_names,
        seed=seed + 7,
        title="UMAP(训练后): 颜色=类别, 形状=域",
        logger=logger,
    )

