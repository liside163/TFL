import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CASE_RE = re.compile(r"^Case_(\d)(\d)(\d{2})(\d+)\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class FileInfo:
    path: str
    A: int
    B: int
    CD: str


class _LRUFileCache:
    # 为什么：STFT 的 __getitem__ 里会反复访问同一文件，缓存能显著减少 IO 开销
    def __init__(self, max_items: int):
        self.max_items = int(max_items)
        self._data: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)


def parse_case_filename(name: str) -> Optional[Tuple[int, int, str]]:
    """Parse filename Case_ABC...csv into (A, B, CD)."""
    m = CASE_RE.match(name)
    if not m:
        return None
    A = int(m.group(1))
    B = int(m.group(2))
    CD = m.group(3)
    return A, B, CD

def discover_files(root_dir: str, file_glob: str) -> List[FileInfo]:
    root = Path(root_dir)
    if not root.exists():
        return []
    infos: List[FileInfo] = []
    for p in sorted(root.rglob(file_glob)):
        parsed = parse_case_filename(p.name)
        if parsed is None:
            continue
        A, B, CD = parsed
        infos.append(FileInfo(path=str(p), A=A, B=B, CD=CD))
    return infos


def build_label_map(infos: List[FileInfo]) -> Dict[str, int]:
    """Build label map from FileInfo list."""
    unique = sorted({fi.CD for fi in infos})
    return {cd: i for i, cd in enumerate(unique)}

def read_csv_six_channels(path: str, columns: List[str]) -> Optional[np.ndarray]:
    """Read only the 6 required columns; tolerate missing/empty files."""
    try:
        import pandas as pd
    except Exception:
        pd = None

    try:
        if pd is not None:
            df = pd.read_csv(path, usecols=columns)
            if df.shape[0] == 0:
                return None
            x = df.to_numpy(dtype=np.float32, copy=True)
        else:
            data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
            if data.size == 0:
                return None
            x = np.stack([data[c].astype(np.float32) for c in columns], axis=1)
    except Exception:
        return None

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 2 or x.shape[1] != 6:
        return None
    return x

def _expected_stft_shape(t: int, n_fft: int, hop_length: int, win_length: int, center: bool) -> Tuple[int, int]:
    # 为什么：当文件损坏时仍要返回固定形状张量，DataLoader 才不会崩溃
    freq = n_fft // 2 + 1
    if center:
        return freq, max(1, int(np.ceil(t / hop_length)))
    if t < win_length:
        return freq, 1
    frames = 1 + (t - win_length) // hop_length
    return freq, max(1, int(frames))


class RflyMADDataset(Dataset):
    def __init__(
        self,
        files: List[FileInfo],
        label_map: Dict[str, int],
        cfg: Dict,
        domain_id: int,
        return_label: bool,
    ):
        super().__init__()
        self.files = list(files)
        self.label_map = dict(label_map)
        self.domain_id = int(domain_id)
        self.return_label = bool(return_label)

        data_cfg = cfg["data"]
        stft_cfg = cfg["stft"]
        self.columns = list(data_cfg["feature_columns"])
        self.win_len = int(data_cfg["window_length"])
        self.stride = int(data_cfg["window_stride"])
        self.drop_last = bool(data_cfg.get("drop_last_window", True))
        self.max_windows_per_file = data_cfg.get("max_windows_per_file", None)

        self.n_fft = int(stft_cfg["n_fft"])
        self.hop_length = int(stft_cfg["hop_length"])
        self.win_length = int(stft_cfg["win_length"])
        self.center = bool(stft_cfg.get("center", False))
        self.log_mag = bool(stft_cfg.get("log_magnitude", True))
        self.eps = float(stft_cfg.get("eps", 1e-6))
        self.stft_on_gpu = bool(data_cfg.get("stft_on_gpu", False))
        resize_cfg = stft_cfg.get("resize", {}) or {}
        self.resize_enabled = bool(resize_cfg.get("enabled", False))
        self.out_freq = int(resize_cfg.get("out_freq", 64))
        self.out_time = int(resize_cfg.get("out_time", 64))

        cache_size = int(data_cfg.get("cache_size_files", 64))
        self.cache = _LRUFileCache(cache_size)
        self.cache_in_memory = bool(data_cfg.get("cache_in_memory", True))

        self._window_index: List[Tuple[int, int]] = []
        self._file_lengths: List[int] = []
        self._build_index()

        self._stft_window = torch.hann_window(self.win_length)

    def _build_index(self) -> None:
        # 为什么：把“文件×窗口起点”预展开成索引表，迭代器就能 O(1) 定位样本
        self._window_index.clear()
        self._file_lengths.clear()

        for file_idx, fi in enumerate(self.files):
            arr = read_csv_six_channels(fi.path, self.columns)
            if arr is None or arr.shape[0] < self.win_len:
                self._file_lengths.append(0)
                continue
            if self.cache_in_memory:
                self.cache.put(fi.path, arr)
            n = arr.shape[0]
            self._file_lengths.append(n)

            starts = list(range(0, n - self.win_len + 1, self.stride))
            if not self.drop_last:
                last = n - self.win_len
                if len(starts) == 0 or starts[-1] != last:
                    starts.append(max(0, last))

            if self.max_windows_per_file is not None:
                starts = starts[: int(self.max_windows_per_file)]

            for s in starts:
                self._window_index.append((file_idx, int(s)))

    def __len__(self) -> int:
        return len(self._window_index)

    def _get_file_array(self, file_idx: int) -> Optional[np.ndarray]:
        fi = self.files[file_idx]
        arr = self.cache.get(fi.path)
        if arr is not None:
            return arr
        arr = read_csv_six_channels(fi.path, self.columns)
        if arr is None:
            return None
        if self.cache_in_memory:
            self.cache.put(fi.path, arr)
        return arr

    def _window_to_spectrogram(self, x_win: np.ndarray) -> torch.Tensor:
        # 为什么：论文用“二维谱图”做输入，这里用 STFT 把 1D 序列转成 2D 时频表示
        x = torch.from_numpy(x_win).float()  # (T, 6)
        x = x.transpose(0, 1)  # (6, T)

        specs = []
        for c in range(x.size(0)):
            s = torch.stft(
                x[c],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self._stft_window,
                center=self.center,
                return_complex=True,
            )
            mag = torch.abs(s)
            if self.log_mag:
                mag = torch.log1p(mag)
            specs.append(mag)
        spec = torch.stack(specs, dim=0)  # (6, F, TT)

        if self.resize_enabled:
            spec = F.interpolate(
                spec.unsqueeze(0),
                size=(self.out_freq, self.out_time),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # 为什么：按窗口做标准化能减少不同飞行段幅值差异对域对齐的干扰
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        spec = (spec - mean) / std
        return spec

    def __getitem__(self, idx: int):
        file_idx, start = self._window_index[idx]
        fi = self.files[file_idx]
        arr = self._get_file_array(file_idx)

        if self.resize_enabled:
            out_f, out_t = self.out_freq, self.out_time
        else:
            out_f, out_t = _expected_stft_shape(self.win_len, self.n_fft, self.hop_length, self.win_length, self.center)

        if arr is None:
            # fallback for missing files
            if self.stft_on_gpu:
                x = torch.zeros((6, self.win_len), dtype=torch.float32)
            else:
                x = torch.zeros((6, out_f, out_t), dtype=torch.float32)
            y = -1
        else:
            x_win = arr[start : start + self.win_len, :]
            if self.stft_on_gpu:
                x = torch.from_numpy(x_win).float().transpose(0, 1)
            else:
                x = self._window_to_spectrogram(x_win)
            y = self.label_map.get(fi.CD, -1)
        if self.return_label:
            return x, int(y), int(self.domain_id)
        return x, int(self.domain_id)


def split_source_files(files: List[FileInfo], val_ratio: float, seed: int) -> Tuple[List[FileInfo], List[FileInfo]]:
    # split source files into train/val subsets
    if len(files) == 0:
        return [], []
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = int(round(len(files) * float(val_ratio)))
    val_idx = set(idx[:n_val].tolist())
    train_files = [f for i, f in enumerate(files) if i not in val_idx]
    val_files = [f for i, f in enumerate(files) if i in val_idx]
    return train_files, val_files
