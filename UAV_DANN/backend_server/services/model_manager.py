# ????: ???????????????????????
import threading
import time
from pathlib import Path
import sys
from typing import Optional, Dict, Any

import numpy as np
import torch
import yaml

from backend_server.config import settings
from backend_server.services.scaler_manager import ScalerManager


PROJECT_ROOT = Path(settings.project_root).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dann_deep import DANNDeep  # noqa: E402


class ModelManager:
    def __init__(self, scaler_manager: Optional[ScalerManager] = None) -> None:
        self.model: Optional[DANNDeep] = None
        self.device = torch.device("cuda" if settings.device == "cuda" and torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.scaler_manager = scaler_manager

    def load_model(self, checkpoint_path: str, config_path: str) -> None:
        config = self._load_config(config_path)
        hp = config.get("training", {}).get("model_hyperparameters", {})

        cnn_cfg = hp.get("cnn", {})
        lstm_cfg = hp.get("lstm", {})
        classifier_cfg = hp.get("classifier", {})
        discriminator_cfg = hp.get("discriminator", {})

        model = DANNDeep(
            n_features=settings.feature_dim,
            seq_len=settings.window_size,
            num_classes=settings.num_classes,
            cnn_layers=cnn_cfg.get("num_layers", 2),
            cnn_channels=cnn_cfg.get("channels", [64, 128]),
            cnn_kernel_size=cnn_cfg.get("kernel_size", 5),
            lstm_hidden=lstm_cfg.get("hidden_size", 128),
            lstm_layers=lstm_cfg.get("num_layers", 2),
            lstm_dropout=lstm_cfg.get("dropout", 0.3),
            lstm_bidirectional=lstm_cfg.get("bidirectional", False),
            classifier_layers=classifier_cfg.get("num_layers", 2),
            classifier_hidden=classifier_cfg.get("hidden_dim", 64),
            classifier_dropout=classifier_cfg.get("dropout", 0.5),
            discriminator_layers=discriminator_cfg.get("num_layers", 2),
            discriminator_hidden=discriminator_cfg.get("hidden_dim", 64),
            use_batchnorm=hp.get("use_batchnorm", False),
            use_layernorm=hp.get("use_layernorm", True),
            use_attention=hp.get("use_attention", True),
            use_residual=hp.get("use_residual", True),
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model

    def predict(self, window_data: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        if window_data.shape != (settings.window_size, settings.feature_dim):
            raise ValueError("Invalid window shape")

        with self.lock:
            start = time.perf_counter()

            data = window_data.astype(np.float32)
            if self.scaler_manager is not None:
                data = self.scaler_manager.transform(data)

            tensor = torch.from_numpy(data).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)
                logits = outputs["class_logits"]
                probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

            fault_class = int(np.argmax(probs))
            confidence = float(np.max(probs))

            latency_ms = (time.perf_counter() - start) * 1000.0
            if latency_ms > settings.max_latency_ms:
                print(f"[ModelManager] Warning: latency {latency_ms:.2f}ms exceeds threshold")

            return {
                "fault_class": fault_class,
                "confidence": confidence,
                "probabilities": probs.tolist(),
                "latency_ms": latency_ms,
            }

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
