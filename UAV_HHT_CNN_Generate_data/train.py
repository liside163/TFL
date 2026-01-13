import itertools
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losses import MKMMDLoss
from utils import AverageMeter, build_stft_preprocess, save_checkpoint


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, preprocess) -> Tuple[float, float]:
    # 为什么：源域验证集用来选最优权重，避免训练过程震荡导致模型回退
    model.eval()
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    for x, y, _domain in loader:
        x = preprocess(x)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_meter.update(loss.item(), n=int(x.size(0)))
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    acc = correct / max(1, total)
    return loss_meter.avg, acc


def train_mkmmd(
    model: torch.nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
    out_dir: str,
    logger,
) -> Dict[str, list]:
    # 为什么：分类损失保证源域判别能力，MKMMD 约束源/目标特征分布对齐以提升真实飞行泛化
    train_cfg = cfg["train"]
    mmd_cfg = cfg["mkmmd"]

    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    mmd_loss_fn = MKMMDLoss(
        kernel_mul=float(mmd_cfg.get("kernel_mul", 2.0)),
        kernel_num=int(mmd_cfg.get("kernel_num", 5)),
        fix_sigma=mmd_cfg.get("fix_sigma", None),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    history = {"loss_total": [], "loss_clf": [], "loss_mmd": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_path = str(Path(out_dir) / "best.pt")

    target_iter = itertools.cycle(target_loader)
    preprocess = build_stft_preprocess(cfg, device)

    for epoch in range(int(train_cfg["num_epochs"])):
        model.train()
        loss_total = AverageMeter()
        loss_clf = AverageMeter()
        loss_mmd = AverageMeter()

        for step, batch_s in enumerate(source_loader):
            x_s, y_s, _dom_s = batch_s
            x_t, _dom_t = next(target_iter)

            x_s = preprocess(x_s)
            y_s = y_s.to(device, non_blocking=True)
            x_t = preprocess(x_t)

            optimizer.zero_grad(set_to_none=True)

            logits_s, feat_s = model(x_s, return_feat=True)
            _logits_t, feat_t = model(x_t, return_feat=True)

            l_clf = criterion(logits_s, y_s)
            l_mmd = mmd_loss_fn(feat_s, feat_t)
            total = l_clf + float(train_cfg["mmd_lambda"]) * l_mmd

            total.backward()
            if float(train_cfg.get("grad_clip_norm", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["grad_clip_norm"]))
            optimizer.step()

            bs = int(x_s.size(0))
            loss_total.update(total.item(), n=bs)
            loss_clf.update(l_clf.item(), n=bs)
            loss_mmd.update(l_mmd.item(), n=bs)

            if (step + 1) % int(train_cfg.get("log_interval", 20)) == 0:
                logger.info(
                    f"epoch={epoch+1:03d} step={step+1:04d} "
                    f"loss_total={loss_total.avg:.4f} loss_clf={loss_clf.avg:.4f} loss_mmd={loss_mmd.avg:.4f}"
                )

        v_loss, v_acc = evaluate(model, val_loader, device, preprocess)
        history["loss_total"].append(loss_total.avg)
        history["loss_clf"].append(loss_clf.avg)
        history["loss_mmd"].append(loss_mmd.avg)
        history["val_loss"].append(float(v_loss))
        history["val_acc"].append(float(v_acc))

        logger.info(
            f"epoch={epoch+1:03d} train_total={loss_total.avg:.4f} train_clf={loss_clf.avg:.4f} "
            f"train_mmd={loss_mmd.avg:.4f} val_loss={v_loss:.4f} val_acc={v_acc:.4f}"
        )

        extra = {"epoch": epoch + 1, "history": history, "best_acc": best_acc}
        save_checkpoint(str(Path(out_dir) / "last.pt"), model, optimizer, extra)

        if v_acc > best_acc:
            best_acc = float(v_acc)
            extra = {"epoch": epoch + 1, "history": history, "best_acc": best_acc}
            save_checkpoint(best_path, model, optimizer, extra)

    logger.info(f"best_val_acc={best_acc:.4f} best_ckpt={best_path}")
    return history
