# ????: ???????????????????????
from collections import deque
from typing import Optional
import numpy as np


class SlidingWindowBuffer:
    def __init__(self, window_size: int = 100, feature_dim: int = 21) -> None:
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=window_size)
        self.is_ready = False

    def add_sample(self, sample: np.ndarray) -> None:
        if sample.shape != (self.feature_dim,):
            raise ValueError(f"Sample shape {sample.shape} does not match ({self.feature_dim},)")
        self.buffer.append(sample)
        self.is_ready = len(self.buffer) >= self.window_size

    def get_window(self) -> Optional[np.ndarray]:
        if not self.is_ready:
            return None
        return np.stack(self.buffer, axis=0)

    def reset(self) -> None:
        self.buffer.clear()
        self.is_ready = False
