from pathlib import Path

import torch
from torch.utils.data import DataLoader

from analyze import run_analysis
from data_loader import RflyMADDataset, build_label_map, discover_files, split_source_files
from model import DomainAdaptNet
from train import train_mkmmd
from utils import ensure_dir, get_device, load_yaml, now_str, plot_training_curves, save_json, set_seed, setup_logger


def main() -> None:
    cfg = load_yaml("config.yaml")

    seed = int(cfg["project"].get("seed", 42))
    set_seed(seed)

    run_dir = Path(cfg["project"]["output_root"]) / cfg["project"]["exp_name"] / now_str()
    ensure_dir(str(run_dir))
    logger = setup_logger(str(run_dir), name="rflymad")
    save_json(str(run_dir / "config_resolved.json"), cfg)

    device = get_device(str(cfg["train"].get("device", "auto")))
    logger.info(f"device={device}")

    infos = discover_files(cfg["data"]["root_dir"], cfg["data"]["file_glob"])
    if len(infos) == 0:
        logger.info("未发现任何 Case_*.csv 文件，请检查 config.yaml 的 data.root_dir")
        return

    label_map = build_label_map(infos)
    inv = {v: k for k, v in label_map.items()}
    class_names = [inv[i] for i in range(len(inv))]
    logger.info(f"发现类别数={len(class_names)} 类别={class_names}")

    source_As = set(int(x) for x in cfg["data"]["source_A_values"])
    target_As = set(int(x) for x in cfg["data"]["target_A_values"])
    source_files = [fi for fi in infos if fi.A in source_As]
    target_files = [fi for fi in infos if fi.A in target_As]
    if len(source_files) == 0 or len(target_files) == 0:
        logger.info(
            f"源域文件数={len(source_files)} 目标域文件数={len(target_files)}，请检查 A 的划分规则"
        )
        return

    train_files, val_files = split_source_files(source_files, cfg["train"]["source_val_ratio"], seed)

    ds_source_train = RflyMADDataset(train_files, label_map, cfg, domain_id=0, return_label=True)
    ds_source_val = RflyMADDataset(val_files, label_map, cfg, domain_id=0, return_label=True)
    ds_target_train = RflyMADDataset(target_files, label_map, cfg, domain_id=1, return_label=False)
    ds_target_eval = RflyMADDataset(target_files, label_map, cfg, domain_id=1, return_label=True)

    logger.info(
        f"样本数(source_train)={len(ds_source_train)} 样本数(source_val)={len(ds_source_val)} "
        f"样本数(target_train)={len(ds_target_train)} 样本数(target_eval)={len(ds_target_eval)}"
    )

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 0))
    pin = bool(cfg["train"].get("pin_memory", True))

    source_loader = DataLoader(ds_source_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=True)
    target_loader = DataLoader(ds_target_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(ds_source_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, drop_last=False)
    real_eval_loader = DataLoader(ds_target_eval, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, drop_last=False)

    model = DomainAdaptNet(
        num_classes=len(class_names),
        base_channels=int(cfg["model"]["base_channels"]),
        feat_dim=int(cfg["model"]["feat_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    # 为什么：保存随机初始化权重，用于对比 UMAP“训练前/训练后”的域对齐效果
    init_path = str(run_dir / "init.pt")
    torch.save({"model": model.state_dict()}, init_path)

    history = train_mkmmd(
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        out_dir=str(run_dir),
        logger=logger,
    )

    plot_training_curves(str(run_dir / "training_curves.png"), history)

    run_analysis(
        model=model,
        init_ckpt_path=init_path,
        best_ckpt_path=str(run_dir / "best.pt"),
        source_vis_loader=val_loader,
        target_vis_loader=real_eval_loader,
        real_eval_loader=real_eval_loader,
        device=device,
        out_dir=str(run_dir),
        cfg=cfg,
        class_names=class_names,
        logger=logger,
    )


if __name__ == "__main__":
    main()
