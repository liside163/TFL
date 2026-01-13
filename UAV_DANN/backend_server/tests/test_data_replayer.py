# ????: ???????????????????????
import tempfile
import unittest
import numpy as np
import pandas as pd

from backend_server.services.data_replayer import DataReplayer, FEATURE_COLUMNS


class TestDataReplayer(unittest.TestCase):
    def test_load_data(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data = np.random.randn(10, len(FEATURE_COLUMNS))
            df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
            df.to_csv(f.name, index=False)
            temp_path = f.name

        replayer = DataReplayer(csv_path=temp_path, speed_factor=1.0)
        replayer.load_data()

        self.assertIsNotNone(replayer.data)
        self.assertEqual(replayer.data.shape, (10, len(FEATURE_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
