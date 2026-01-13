#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Randomly sample CSV files and visualize time/frequency patterns by condition/fault.

Example:
  python scripts/eda_random_visualize.py --config config/config.yaml \
    --domain HIL --group_by condition_fault --per_group 3 --max_len 2000 \
    --output_dir logs/eda
"""

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402


CONDITION_NAMES = {
    0: "hover",
    1: "waypoint",
    2: "velocity",
    3: "circling",
    4: "acce",
    5: "dece",
}

DOMAIN_CODE = {
    "1": "SIL",
    "2": "HIL",
    "3": "REAL",
    "4": "SIL_ROS",
    "5": "HIL_ROS",
}


@dataclass
class SampleInfo:
    path: str
    domain: str
    condition: int
    fault_code: str
    fault_label: int


@dataclass
class SampleData:
    time: np.ndarray
    values: np.ndarray
    dt: float


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_case_filename(path: str) -> Optional[Tuple[str, int, str]]:
    filename = os.path.basename(path)
    match = re.match(r"Case_(\d)(\d)(\d{2})(\d{6})", filename)
    if not match:
        return None
    dataset_code = match.group(1)
    condition = int(match.group(2))
    fault_code = match.group(3)
    return dataset_code, condition, fault_code


def map_fault_label(config: Dict, fault_code: str) -> Optional[int]:
    code_to_label = config["fault_types"]["code_to_label"]
    if fault_code in code_to_label:
        return int(code_to_label[fault_code])
    return None


def list_csv_files(data_root: str, domain: str) -> List[str]:
    domain_dir = os.path.join(data_root, domain)
    if not os.path.isdir(domain_dir):
        return []
    return sorted(glob.glob(os.path.join(domain_dir, "*.csv")))


def filter_samples(
    files: List[str],
    config: Dict,
    domain_filter: Optional[str],
    conditions: Optional[List[int]],
    fault_labels: Optional[List[int]],
    fault_codes: Optional[List[str]],
    include_skipped: bool,
) -> List[SampleInfo]:
    samples = []
    skip_codes = set(config["fault_types"].get("skip_codes", []))
    for path in files:
        parsed = parse_case_filename(path)
        if parsed is None:
            continue
        dataset_code, condition, fault_code = parsed
        domain = DOMAIN_CODE.get(dataset_code, "UNKNOWN")

        if domain_filter and domain != domain_filter:
            continue
        if conditions and condition not in conditions:
            continue
        if fault_codes and fault_code not in fault_codes:
            continue
        if (fault_code in skip_codes) and not include_skipped:
            continue

        fault_label = map_fault_label(config, fault_code)
        if fault_label is None:
            continue
        if fault_labels and fault_label not in fault_labels:
            continue

        samples.append(
            SampleInfo(
                path=path,
                domain=domain,
                condition=condition,
                fault_code=fault_code,
                fault_label=fault_label,
            )
        )
    return samples


def load_sample(
    sample: SampleInfo,
    features: List[str],
    max_len: Optional[int],
    normalize: bool,
    fault_state_only: bool,
) -> Optional[SampleData]:
    df = pd.read_csv(sample.path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        return None

    if fault_state_only and "UAVState_data_fault_state" in df.columns:
        df = df[df["UAVState_data_fault_state"] != 0]
        if df.empty:
            return None

    time_col = None
    if "trueTime" in df.columns:
        time_col = "trueTime"
    elif "Timestamp" in df.columns:
        time_col = "Timestamp"

    values = df[features].astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.interpolate(method="linear", limit_direction="both").fillna(0.0)
    values = values.to_numpy()

    if max_len is not None and values.shape[0] > max_len:
        values = values[:max_len]

    if normalize:
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        values = (values - mean) / std

    if time_col:
        time = df[time_col].to_numpy()
        if max_len is not None and time.shape[0] > max_len:
            time = time[:max_len]
        diffs = np.diff(time)
        dt = float(np.median(diffs)) if diffs.size > 0 else 1.0
        if dt <= 0:
            dt = 1.0
    else:
        time = np.arange(values.shape[0], dtype=float)
        dt = 1.0

    return SampleData(time=time, values=values, dt=dt)


def compute_fft(sample: SampleData) -> Tuple[np.ndarray, np.ndarray]:
    values = sample.values - sample.values.mean(axis=0, keepdims=True)
    n = values.shape[0]
    freqs = np.fft.rfftfreq(n, d=sample.dt)
    spec = np.abs(np.fft.rfft(values, axis=0))
    return freqs, spec


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def plot_time_domain(
    group_name: str,
    samples: List[SampleData],
    features: List[str],
    output_dir: str,
    overlay_samples: bool,
    max_plots_per_fig: int,
) -> None:
    feature_to_idx = {name: idx for idx, name in enumerate(features)}
    min_len = min(s.values.shape[0] for s in samples)
    time = samples[0].time[:min_len]
    stacked = np.stack([s.values[:min_len] for s in samples], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    for idx, feature_chunk in enumerate(chunk_list(features, max_plots_per_fig)):
        cols = 3
        rows = int(np.ceil(len(feature_chunk) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for i, feature in enumerate(feature_chunk):
            ax = axes[i]
            f_idx = feature_to_idx[feature]
            if overlay_samples:
                for s in samples:
                    ax.plot(time, s.values[:min_len, f_idx], alpha=0.15, linewidth=0.8)
            ax.plot(time, mean[:, f_idx], color="black", linewidth=1.4)
            ax.fill_between(
                time,
                mean[:, f_idx] - std[:, f_idx],
                mean[:, f_idx] + std[:, f_idx],
                color="gray",
                alpha=0.25,
            )
            ax.set_title(feature, fontsize=9)
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Time Domain - {group_name} (chunk {idx + 1})", fontsize=12)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        out_path = os.path.join(output_dir, f"time_{group_name}_chunk{idx + 1}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_frequency_domain(
    group_name: str,
    samples: List[SampleData],
    features: List[str],
    output_dir: str,
    max_plots_per_fig: int,
) -> None:
    feature_to_idx = {name: idx for idx, name in enumerate(features)}
    specs = []
    freqs_ref = None

    for sample in samples:
        freqs, spec = compute_fft(sample)
        if freqs_ref is None:
            freqs_ref = freqs
            specs.append(spec)
        else:
            if freqs.shape != freqs_ref.shape or not np.allclose(freqs, freqs_ref):
                aligned = np.zeros((freqs_ref.shape[0], spec.shape[1]))
                for i in range(spec.shape[1]):
                    aligned[:, i] = np.interp(freqs_ref, freqs, spec[:, i])
                specs.append(aligned)
            else:
                specs.append(spec)

    stacked = np.stack(specs, axis=0)
    mean_spec = stacked.mean(axis=0)
    std_spec = stacked.std(axis=0)

    for idx, feature_chunk in enumerate(chunk_list(features, max_plots_per_fig)):
        cols = 3
        rows = int(np.ceil(len(feature_chunk) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for i, feature in enumerate(feature_chunk):
            ax = axes[i]
            f_idx = feature_to_idx[feature]
            ax.plot(freqs_ref, mean_spec[:, f_idx], color="darkblue", linewidth=1.2)
            ax.fill_between(
                freqs_ref,
                mean_spec[:, f_idx] - std_spec[:, f_idx],
                mean_spec[:, f_idx] + std_spec[:, f_idx],
                color="steelblue",
                alpha=0.25,
            )
            ax.set_title(feature, fontsize=9)
            ax.set_ylabel("FFT Magnitude")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Frequency Domain - {group_name} (chunk {idx + 1})", fontsize=12)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        out_path = os.path.join(output_dir, f"freq_{group_name}_chunk{idx + 1}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def compute_summary_features(sample: SampleData) -> np.ndarray:
    values = sample.values
    time_mean = values.mean(axis=0)
    time_std = values.std(axis=0)

    freqs, spec = compute_fft(sample)
    max_freq = freqs[-1] if freqs.size > 0 else 1.0
    edges = [0.0, max_freq * 0.33, max_freq * 0.66, max_freq]

    band_features = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (freqs >= start) & (freqs <= end)
        if not np.any(mask):
            band_energy = np.zeros(values.shape[1])
        else:
            band_energy = np.mean(spec[mask] ** 2, axis=0)
        band_features.append(band_energy)

    return np.concatenate([time_mean, time_std] + band_features, axis=0)


def compute_separability(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    label_names: Dict[int, str],
    output_dir: str,
    label_by: str,
    seed: int,
) -> Optional[Dict]:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return None

    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(feature_matrix)

    try:
        score = silhouette_score(projected, labels)
    except ValueError:
        score = None

    centroids = []
    within = []
    for label in unique_labels:
        group = projected[labels == label]
        centroid = group.mean(axis=0)
        centroids.append(centroid)
        within.append(np.mean(np.linalg.norm(group - centroid, axis=1)))
    centroids = np.vstack(centroids)
    within_mean = float(np.mean(within)) if within else 0.0
    between = np.mean(
        np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    )
    ratio = float(between / within_mean) if within_mean > 0 else None

    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=unique_labels.size)
    for idx, label in enumerate(unique_labels):
        pts = projected[labels == label]
        plt.scatter(pts[:, 0], pts[:, 1], label=label_names.get(label, str(label)), s=40, alpha=0.7)
    plt.legend(loc="best", fontsize=8)
    plt.title(f"Separability (PCA) by {label_by}")
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(output_dir, f"separability_{label_by}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return {
        "label_by": label_by,
        "num_samples": int(feature_matrix.shape[0]),
        "num_labels": int(unique_labels.size),
        "silhouette_score": score,
        "between_within_ratio": ratio,
    }


def build_group_name(key: Tuple, config: Dict, group_by: str) -> str:
    fault_names = config["fault_types"]["labels"]
    if group_by == "condition":
        condition = key[0]
        return f"cond{condition}_{CONDITION_NAMES.get(condition, 'unknown')}"
    if group_by == "fault":
        fault = key[0]
        return f"fault{fault}_{fault_names.get(fault, 'unknown')}"
    condition, fault = key
    return (
        f"cond{condition}_{CONDITION_NAMES.get(condition, 'unknown')}"
        f"__fault{fault}_{fault_names.get(fault, 'unknown')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Random CSV EDA for UAV dataset")
    parser.add_argument("--config", default="./config/config.yaml", help="Config path")
    parser.add_argument("--domain", default=None, choices=["HIL", "REAL"], help="Domain filter")
    parser.add_argument("--data_root", default=None, help="Override data root from config")
    parser.add_argument("--group_by", default="condition_fault", choices=["condition", "fault", "condition_fault"])
    parser.add_argument("--conditions", nargs="*", type=int, default=None, help="Filter conditions")
    parser.add_argument("--fault_labels", nargs="*", type=int, default=None, help="Filter fault labels")
    parser.add_argument("--fault_codes", nargs="*", default=None, help="Filter fault codes (e.g. 00 05 10)")
    parser.add_argument("--per_group", type=int, default=3, help="Samples per group")
    parser.add_argument("--max_groups", type=int, default=12, help="Limit number of groups")
    parser.add_argument("--max_len", type=int, default=2000, help="Max timesteps per sample")
    parser.add_argument("--normalize", action="store_true", help="Z-score per sample")
    parser.add_argument("--fault_state_only", action="store_true", help="Use fault_state != 0 if available")
    parser.add_argument("--overlay_samples", action="store_true", help="Overlay raw samples")
    parser.add_argument("--features", nargs="*", default=None, help="Feature list override")
    parser.add_argument("--max_plots_per_fig", type=int, default=9, help="Max subplots per figure")
    parser.add_argument("--output_dir", default="./logs/eda", help="Output dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(project_root, config_path)
        if os.path.exists(candidate):
            config_path = candidate
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    features = args.features or config["preprocessing"]["selected_features"]

    data_root = args.data_root or config["data"]["data_root"]
    domains = ["HIL", "REAL"] if args.domain is None else [args.domain]

    all_files = []
    for domain in domains:
        domain_files = list_csv_files(data_root, domain)
        print(f"[scan] {domain}: {len(domain_files)} files under {os.path.join(data_root, domain)}")
        all_files.extend(domain_files)

    samples = filter_samples(
        files=all_files,
        config=config,
        domain_filter=args.domain,
        conditions=args.conditions,
        fault_labels=args.fault_labels,
        fault_codes=args.fault_codes,
        include_skipped=False,
    )

    print(f"[filter] matched samples: {len(samples)}")
    if not samples:
        print("No samples found with current filters.")
        return

    groups = defaultdict(list)
    for s in samples:
        if args.group_by == "condition":
            key = (s.condition,)
        elif args.group_by == "fault":
            key = (s.fault_label,)
        else:
            key = (s.condition, s.fault_label)
        groups[key].append(s)

    rng = random.Random(args.seed)
    group_keys = sorted(groups.keys())
    if args.max_groups and len(group_keys) > args.max_groups:
        group_keys = rng.sample(group_keys, args.max_groups)

    os.makedirs(args.output_dir, exist_ok=True)

    feature_matrix = []
    label_fault = []
    label_condition = []

    summary = {"groups": []}

    for key in group_keys:
        group_samples = groups[key]
        selected = rng.sample(group_samples, min(args.per_group, len(group_samples)))

        loaded = []
        for s in selected:
            data = load_sample(
                s,
                features=features,
                max_len=args.max_len,
                normalize=args.normalize,
                fault_state_only=args.fault_state_only,
            )
            if data is None:
                continue
            loaded.append(data)

            feature_matrix.append(compute_summary_features(data))
            label_fault.append(s.fault_label)
            label_condition.append(s.condition)

        if not loaded:
            continue

        group_name = build_group_name(key, config, args.group_by)
        plot_time_domain(
            group_name=group_name,
            samples=loaded,
            features=features,
            output_dir=args.output_dir,
            overlay_samples=args.overlay_samples,
            max_plots_per_fig=args.max_plots_per_fig,
        )
        plot_frequency_domain(
            group_name=group_name,
            samples=loaded,
            features=features,
            output_dir=args.output_dir,
            max_plots_per_fig=args.max_plots_per_fig,
        )

        summary["groups"].append(
            {
                "group": group_name,
                "num_files": len(loaded),
            }
        )

    if feature_matrix:
        features_np = np.vstack(feature_matrix)
        labels_fault = np.array(label_fault)
        labels_condition = np.array(label_condition)

        fault_names = {int(k): v for k, v in config["fault_types"]["labels"].items()}
        condition_names = {k: v for k, v in CONDITION_NAMES.items()}

        sep_fault = compute_separability(
            feature_matrix=features_np,
            labels=labels_fault,
            label_names=fault_names,
            output_dir=args.output_dir,
            label_by="fault",
            seed=args.seed,
        )
        sep_condition = compute_separability(
            feature_matrix=features_np,
            labels=labels_condition,
            label_names=condition_names,
            output_dir=args.output_dir,
            label_by="condition",
            seed=args.seed,
        )

        summary["separability"] = {
            "fault": sep_fault,
            "condition": sep_condition,
        }

    summary_path = os.path.join(args.output_dir, "eda_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
