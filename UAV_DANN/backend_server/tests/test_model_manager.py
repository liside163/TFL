# ????: ???????????????????????
import unittest
from pathlib import Path
import numpy as np

from backend_server.config import settings
from backend_server.services.model_manager import ModelManager
from backend_server.services.scaler_manager import ScalerManager


class TestModelManager(unittest.TestCase):
    def test_model_predict(self) -> None:
        if not Path(settings.model_path).exists():
            self.skipTest("Model checkpoint not found")
        if not Path(settings.scaler_path).exists():
            self.skipTest("Scaler not found")

        scaler = ScalerManager()
        scaler.load_scaler(settings.scaler_path)

        mgr = ModelManager(scaler_manager=scaler)
        mgr.load_model(settings.model_path, settings.config_path)

        dummy_input = np.random.randn(settings.window_size, settings.feature_dim).astype(np.float32)
        result = mgr.predict(dummy_input)

        self.assertIn("fault_class", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("latency_ms", result)
        self.assertGreaterEqual(result["fault_class"], 0)
        self.assertLessEqual(result["fault_class"], settings.num_classes - 1)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
