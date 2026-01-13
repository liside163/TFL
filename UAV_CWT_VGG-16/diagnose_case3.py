"""Inference script: diagnose real flight Case_3 CSV files with trained VGG-16 model."""

# === Configuration ===
INPUT_CSV = "ProcessData"  # File or folder; if folder, all Case_3*.csv will be processed
MODEL_PATH = "best_vgg16_fault_diagnosis.pth"
IMG_SIZE = (224, 224)
WINDOW_SIZE = 256
STEP_SIZE = 64
WAVELET = "morl"
SCALES = list(range(1, 129))

from pathlib import Path
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib

# Use non-interactive backend to avoid GUI issues in headless/remote runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

FEATURE_COL = "_sensor_combined_0_accelerometer_m_s2[2]"  # Z-axis accelerometer
MODE_COL = "_rfly_ctrl_lxl_0_mode"  # Fault mode indicator
CLASS_NAMES = ["Normal", "Fault"]


def window_signal(signal: np.ndarray):
    """Yield (start_idx, window_array) with sliding windows."""
    n = len(signal)
    if n < WINDOW_SIZE:
        return
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        yield start, signal[start : start + WINDOW_SIZE]


def window_to_rgb(window: np.ndarray) -> Image.Image:
    """Convert 1D window to RGB scalogram resized for VGG-16."""
    coef, _ = pywt.cwt(window, SCALES, WAVELET)
    power = np.abs(coef)
    power -= power.min()
    power /= power.max() + 1e-9
    cmap = plt.get_cmap("jet")
    rgba = cmap(power)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    return img.resize(IMG_SIZE, Image.BILINEAR)


def build_model(num_classes: int) -> models.VGG:
    model = models.vgg16(weights=None)
    # Freeze features by default
    for p in model.features.parameters():
        p.requires_grad = False
    # Recreate classifier to match training
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model


def load_signal(csv_path: Path) -> tuple[np.ndarray, str]:
    df = pd.read_csv(csv_path)
    if FEATURE_COL not in df.columns:
        raise KeyError(f"Missing required column {FEATURE_COL} in {csv_path.name}")
    signal = df[FEATURE_COL].to_numpy()

    mode_tag = "ModeNA"
    if MODE_COL in df.columns:
        modes = df[MODE_COL].dropna().to_numpy()
        if len(modes) > 0:
            mode_tag = f"Mode{int(modes[0])}"
    return signal, mode_tag


def diagnose(csv_path: Path, model, device: torch.device):
    signal, mode_tag = load_signal(csv_path)
    to_tensor = transforms.ToTensor()  # Training used Resize + ToTensor; we already resize

    probs = []
    starts = []
    for start_idx, window in window_signal(signal):
        img = window_to_rgb(window)
        tensor = to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            soft = torch.softmax(logits, dim=1)
            fault_prob = soft[0, 1].item()
        probs.append(fault_prob)
        starts.append(start_idx)

    return np.array(starts), np.array(probs), signal, mode_tag


def plot_report(csv_path: Path, starts: np.ndarray, probs: np.ndarray, signal: np.ndarray, mode_tag: str, output_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    axes[0].plot(signal, color="steelblue")
    axes[0].set_title(f"Raw Z-Accel: {csv_path.name} ({mode_tag})")
    axes[0].set_ylabel("Accel (m/s^2)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(starts, probs, color="firebrick")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fault Probability")
    axes[1].set_xlabel("Sample Index (window start)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Diagnosis report saved to {output_path}")


def main():
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    input_path = (script_dir / INPUT_CSV).resolve()
    model_path = (script_dir / MODEL_PATH).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Collect Case_3 files
    if input_path.is_dir():
        case_files = sorted(input_path.glob("Case_3*.csv"))
    elif input_path.exists():
        case_files = [input_path]
    else:
        parent = input_path.parent if input_path.parent != Path(".") else Path(".")
        case_files = sorted(parent.glob("Case_3*.csv"))
        if case_files:
            print(f"[INFO] Provided path missing. Using Case_3 files in {parent}")

    if not case_files:
        raise FileNotFoundError(f"No Case_3*.csv files found for path {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    summary = []
    for csv_path in case_files:
        print(f"[INFO] Diagnosing {csv_path.name}")
        starts, probs, signal, mode_tag = diagnose(csv_path, model, device)
        if len(probs) == 0:
            print(f"[WARN] No windows processed for {csv_path.name}; skipping.")
            continue

        output_path = csv_path.with_name(f"Diagnosis_Report_{csv_path.stem}.png")
        plot_report(csv_path, starts, probs, signal, mode_tag, output_path)

        summary.append(
            {
                "file": csv_path.name,
                "mode": mode_tag,
                "num_windows": len(probs),
                "mean_fault_prob": float(np.mean(probs)),
                "max_fault_prob": float(np.max(probs)),
                "max_fault_at": int(starts[int(np.argmax(probs))]),
            }
        )

    if summary:
        summary_path = case_files[0].parent / "Diagnosis_Summary_Case3.csv"
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
        for row in summary:
            print(
                f"{row['file']} ({row['mode']}): windows={row['num_windows']}, "
                f"mean_fault_prob={row['mean_fault_prob']:.3f}, "
                f"max_fault_prob={row['max_fault_prob']:.3f} at start={row['max_fault_at']}"
            )


if __name__ == "__main__":
    main()
