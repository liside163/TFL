"""
数据划分模块测试
"""

import numpy as np
from data.data_split import DataSplitter

def test_split_data():
    """测试数据划分"""
    splitter = DataSplitter()

    # 模拟数据: 100个样本
    X = np.random.randn(100, 100, 27)
    y = np.random.randint(0, 11, 100)

    # 划分
    train, test = splitter.split_target_domain(X, y, test_ratio=0.8)

    assert len(train[0]) == 20, "训练集应为20个样本"
    assert len(test[0]) == 80, "测试集应为80个样本"
    print("OK - Data split test passed")


if __name__ == '__main__':
    test_split_data()
