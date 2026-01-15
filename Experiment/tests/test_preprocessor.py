"""
预处理器测试
"""

import pytest
import numpy as np
from data.preprocessor import Preprocessor
from config import Config

def test_window_slice():
    """测试时间窗口切片"""
    preprocessor = Preprocessor()

    # 创建模拟数据: [1000, 27]
    data = np.random.randn(1000, 27)
    labels = np.random.randint(0, 11, 1000)

    windows, window_labels = preprocessor.create_windows(
        data, labels[0], window_size=100, step=50
    )

    assert windows.shape[0] > 0, "应该生成窗口"
    assert windows.shape[1] == 100, f"窗口大小应为100，实际为{windows.shape[1]}"
    assert windows.shape[2] == 27, f"特征维度应为27，实际为{windows.shape[2]}"
    print(f"✓ 生成{windows.shape[0]}个时间窗口")


def test_normalization():
    """测试数据标准化"""
    preprocessor = Preprocessor()

    # 创建模拟数据
    data = np.random.randn(100, 27) * 10 + 5  # 均值5，标准差10

    # 拟合标准化参数
    preprocessor.fit_normalizer(data)

    # 应用标准化
    normalized = preprocessor.normalize(data)

    # 验证标准化效果 (均值接近0，标准差接近1)
    mean = normalized.mean(axis=0)
    std = normalized.std(axis=0)

    assert np.all(np.abs(mean) < 1e-10), "均值应接近0"
    assert np.all(np.abs(std - 1.0) < 1e-10), "标准差应接近1"
    print("✓ 标准化测试通过")


if __name__ == '__main__':
    test_window_slice()
    test_normalization()
    print("\n所有测试通过!")
