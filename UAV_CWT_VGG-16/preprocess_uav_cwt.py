from __future__ import annotations

"""Preprocess RflyMAD UAV CSV logs into CWT scalogram images for VGG-16."""

# === Configuration ===
# Update these paths before running the script.
# On Windows, prefer forward slashes or raw strings to avoid backslash escapes.
INPUT_DIR = "E:/DL_Learn/FD_BSAEON_DATA/UAV_CWT_VGG-16/ProcessData"
OUTPUT_DIR = "E:/DL_Learn/FD_BSAEON_DATA/UAV_CWT_VGG-16/Image"
IMG_SIZE = (224, 224)  # Target image size for VGG-16
WINDOW_SIZE = 256  # Points per window/segment
STEP_SIZE = 64  # Sliding window step (50% overlap by default)

from pathlib import Path
import numpy as np
import pandas as pd
import pywt
from PIL import Image
from tqdm import tqdm
from matplotlib import cm
import os
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None


FEATURE_COL = "_sensor_combined_0_accelerometer_m_s2[2]"  # Z-axis accelerometer
LABEL_COL = "UAVState_data_fault_state"  # 0 = Normal, >0 = Fault code/type
MODE_COL = "_rfly_ctrl_lxl_0_mode"  # Fault type; take first value per window for naming
WAVELET = "morl"  # Morlet works well for time-frequency analysis
SCALES = np.arange(1, 129)  # Tune if you want higher/lower frequency resolution


def ensure_output_dirs(base: Path) -> dict[str, Path]:
    """Create Train/Normal and Train/Fault folders."""
    train_base = base / "Train"
    normal_dir = train_base / "Normal"
    fault_dir = train_base / "Fault"
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(fault_dir, exist_ok=True)
    return {"Train_Normal": normal_dir, "Train_Fault": fault_dir}


def load_columns(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | tuple[None, None, None]:
    """Read required columns from CSV; returns (signal, labels, modes) or (None, None, None) if missing."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[WARN] Failed to read {csv_path}: {exc}")
        return None, None, None

    missing = [col for col in (FEATURE_COL, LABEL_COL) if col not in df.columns]
    if missing:
        print(f"[WARN] Skipping {csv_path.name}; missing columns: {missing}")
        return None, None, None

    modes_array: np.ndarray | None
    if MODE_COL in df.columns:
        modes_array = df[MODE_COL].to_numpy()
    else:
        print(f"[WARN] {csv_path.name} missing {MODE_COL}; filenames will use ModeNA.")
        modes_array = None

    return df[FEATURE_COL].to_numpy(), df[LABEL_COL].to_numpy(), modes_array


def slice_windows(signal: np.ndarray, labels: np.ndarray, modes: np.ndarray | None = None):
    """Yield (window_signal, class_name, mode_tag) for clean windows of length WINDOW_SIZE."""
    total_len = len(signal)
    if total_len < WINDOW_SIZE:
        return  # Nothing to do if file is shorter than one window

    stride = max(1, STEP_SIZE)
    # Sliding windows with configurable stride to avoid missing short fault intervals
    for start in range(0, total_len - WINDOW_SIZE + 1, stride):
        end = start + WINDOW_SIZE
        label_window = labels[start:end]

        if np.all(label_window == 0):
            class_name = "Normal"
        else:
            unique_vals = np.unique(label_window)
            # Discard mixed labels; keep only pure fault windows by code
            if len(unique_vals) != 1:
                continue
            fault_code = unique_vals[0]
            if fault_code == 0:
                class_name = "Normal"
            else:
                class_name = f"Fault_{int(fault_code)}"  # Include fault type in label

        if modes is not None and len(modes) > start:
            mode_val = modes[start]
            mode_tag = f"Mode{mode_val}"
        else:
            mode_tag = "ModeNA"

        yield signal[start:end], class_name, mode_tag


def window_to_image(window: np.ndarray) -> np.ndarray:
    """Convert 1D signal window to a RGB scalogram image sized for VGG-16."""
    coef, _ = pywt.cwt(window, SCALES, WAVELET)
    power = np.abs(coef)

    # Normalize to [0, 1] to stabilize color mapping
    power -= power.min()
    power /= power.max() + 1e-9

    cmap = cm.get_cmap("jet")
    rgba = cmap(power)  # Returns values in [0, 1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    if cv2 is not None:
        resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    else:
        resized = np.array(Image.fromarray(rgb).resize(IMG_SIZE, Image.BILINEAR))
    return resized


def process_training_file(csv_path: Path, out_dirs: dict[str, Path]) -> int:
    """Process one labeled CSV file; returns number of images saved."""
    signal, labels, modes = load_columns(csv_path)
    if signal is None or labels is None:
        return 0

    saved = 0
    segment_idx = 0
    for window, class_name, mode_tag in slice_windows(signal, labels, modes):
        img = window_to_image(window)
        filename = f"Source_{csv_path.stem}_{mode_tag}_Window{segment_idx:05d}.png"
        if class_name == "Normal":
            out_path = out_dirs["Train_Normal"] / filename
        else:
            out_path = out_dirs["Train_Fault"] / filename
        Image.fromarray(img).save(out_path)
        segment_idx += 1
        saved += 1
    return saved


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {input_dir}")

    out_dirs = ensure_output_dirs(output_dir)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    total_saved = 0
    for csv_path in tqdm(csv_files, desc="Processing CSV files"):
        stem = csv_path.stem
        if stem.startswith("Case_3"):
            print(f"Skipping Real Flight data (Case_3...) to avoid schema mismatch: {csv_path.name}")
            continue

        if stem.startswith(("Case_1", "Case_2")):
            saved = process_training_file(csv_path, out_dirs)
            total_saved += saved
        else:
            # Unknown prefix; skip quietly
            continue
    print(f"Done. Saved {total_saved} training images to {output_dir}")


if __name__ == "__main__":
    main()
