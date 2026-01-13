"""Training entrypoint for UAV fault diagnosis models.

This script wires together data loading, model construction, strategy selection,
optimizer/scheduler setup, and training execution based on a YAML configuration
file.
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import create_dataset
from src.data.transforms import build_augmentation
import src.data.transforms_signal  # noqa: F401  # ensure signal transforms are registered
import src.data.transforms_image  # noqa: F401  # ensure image transforms are registered
from src.features import create_feature_extractor
from src.models.backbones import create_backbone
from src.models.factory import TransferModel
from src.models.heads import create_head
from src.strategies import create_strategy
from src.training.trainer import Trainer
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments containing config path and
        optional seed/device overrides.
    """

    parser = argparse.ArgumentParser(description="UAV fault diagnosis training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use, e.g., 'cuda' or 'cpu'")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across common libraries."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_data_index(
    data_cfg: Dict[str, Any],
) -> Tuple[List[Tuple[str, int, Dict[str, Any]]], List[Tuple[str, int, Dict[str, Any]]]]:
    """Construct dataset indices for training and validation.

    This helper expects the configuration to optionally contain pre-built index
    lists under ``train_index`` and ``val_index``. Each entry should follow the
    convention ``(path, label, meta_dict)``. In real projects, this function can
    be replaced with logic that scans directories or reads split files.

    Args:
        data_cfg (Dict[str, Any]): Data configuration dictionary.

    Returns:
        Tuple[List[Tuple[str, int, Dict[str, Any]]], List[Tuple[str, int, Dict[str, Any]]]]:
            Training and validation indices.

    Raises:
        ValueError: If required indices are missing.
    """

    train_index = data_cfg.get("train_index")
    val_index = data_cfg.get("val_index")
    if train_index is None or val_index is None:
        raise ValueError(
            "Data configuration must provide 'train_index' and 'val_index'. "
            "Each should be a list of (path, label, meta) tuples."
        )
    return train_index, val_index


def create_dataloaders(
    cfg: Dict[str, Any],
    device: str,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders based on configuration.

    Args:
        cfg (Dict[str, Any]): Global configuration.
        device (str): Target device (currently unused but reserved for future use).

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders.
    """

    data_cfg = cfg.get("data", {})

    # Build data indices
    train_index, val_index = build_data_index(data_cfg)

    # Augmentations and feature extractor
    augmentation = build_augmentation(data_cfg.get("augmentation", []))
    feature_extractor = create_feature_extractor(
        data_cfg.get("feature_extractor"), **data_cfg.get("feature_extractor_params", {})
    )

    dataset_name = data_cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("data.dataset_name is required in config")

    train_ds = create_dataset(
        dataset_name,
        data_index=train_index,
        transform=augmentation,
        feature_extractor=feature_extractor,
    )
    val_ds = create_dataset(
        dataset_name,
        data_index=val_index,
        transform=augmentation,
        feature_extractor=feature_extractor,
    )

    train_cfg = cfg.get("train", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", os.cpu_count() or 2)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def create_optimizer_and_scheduler(
    model: torch.nn.Module, cfg: Dict[str, Any]
) -> Tuple[torch.optim.Optimizer, Any]:
    """Create optimizer and (optional) scheduler from configuration.

    Supports Adam and AdamW optimizers. Scheduler configuration follows a simple
    convention: ``train.lr_scheduler = {"name": "steplr", "step_size": 10, "gamma": 0.1}`` or
    ``{"name": "plateau", "mode": "min", "factor": 0.1, "patience": 5}``.
    """

    train_cfg = cfg.get("train", {})
    optim_name = train_cfg.get("optimizer", "adam").lower()
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.0)

    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

    scheduler_cfg = train_cfg.get("lr_scheduler")
    scheduler = None
    if scheduler_cfg:
        name = scheduler_cfg.get("name", "").lower()
        if name == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_cfg.get("step_size", 10),
                gamma=scheduler_cfg.get("gamma", 0.1),
            )
        elif name in ("plateau", "reducelronplateau"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_cfg.get("mode", "min"),
                factor=scheduler_cfg.get("factor", 0.1),
                patience=scheduler_cfg.get("patience", 5),
            )
        else:
            raise ValueError(f"Unsupported scheduler: {name}")

    return optimizer, scheduler


def create_model(cfg: Dict[str, Any]) -> TransferModel:
    """Build TransferModel from configuration."""

    model_cfg = cfg.get("model", {})
    backbone_cfg = model_cfg.get("backbone", {})
    head_cfg = model_cfg.get("head", {})

    if not backbone_cfg:
        raise ValueError("model.backbone configuration is required")
    if not head_cfg:
        raise ValueError("model.head configuration is required")

    backbone = create_backbone(**backbone_cfg)
    head = create_head(**head_cfg)

    freeze_backbone = model_cfg.get("freeze_backbone", False)
    model = TransferModel(backbone, head, freeze_backbone=freeze_backbone)
    return model


def main() -> None:
    """Main training routine."""

    args = parse_args()
    cfg = load_config(args.config)

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build data loaders
    train_loader, val_loader = create_dataloaders(cfg, device)

    # Build model and strategy
    model = create_model(cfg)
    model.to(device)
    strategy = create_strategy(cfg.get("strategy", {}).get("name", "finetune"), model=model, config=cfg)

    # Optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, cfg)

    # Trainer
    trainer = Trainer(
        model=model,
        strategy=strategy,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=cfg,
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
