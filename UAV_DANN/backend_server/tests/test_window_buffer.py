# ????: ???????????????????????
import unittest
import numpy as np

from backend_server.services.window_buffer import SlidingWindowBuffer


class TestSlidingWindowBuffer(unittest.TestCase):
    def test_window_ready(self) -> None:
        buffer = SlidingWindowBuffer(window_size=100, feature_dim=21)
        for _ in range(99):
            buffer.add_sample(np.random.randn(21).astype(np.float32))
        self.assertFalse(buffer.is_ready)

        buffer.add_sample(np.random.randn(21).astype(np.float32))
        self.assertTrue(buffer.is_ready)

        window = buffer.get_window()
        self.assertIsNotNone(window)
        self.assertEqual(window.shape, (100, 21))


if __name__ == "__main__":
    unittest.main()
