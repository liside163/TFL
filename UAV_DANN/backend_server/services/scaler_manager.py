# ????: ???????????????????????
import pickle
import sys
import numpy as np


def _ensure_numpy_core_alias() -> None:
    """
    兼容早期在 Windows 环境生成的 pickle，内部引用 numpy._core.*
    新版 numpy 不再暴露该私有模块，这里做一次模块别名映射以便正常反序列化。
    """
    try:
        import numpy.core as np_core
    except Exception:
        return

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)


class ScalerManager:
    def __init__(self) -> None:
        self.scaler = None

    def load_scaler(self, scaler_path: str) -> None:
        _ensure_numpy_core_alias()
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("Scaler is not loaded")

        original_shape = data.shape
        data_2d = data.reshape(-1, original_shape[-1])
        normalized = self.scaler.transform(data_2d)
        return normalized.reshape(original_shape)
