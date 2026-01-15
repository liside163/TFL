"""
HIL数据集加载器测试
"""

import pytest
import numpy as np
from data.dataset_loader import HILDatasetLoader

def test_parse_filename():
    """测试文件名解析功能"""
    loader = HILDatasetLoader()

    # 测试Hover状态的motor故障
    state_id, fault_type, case_id = loader.parse_filename("Case_2000012345.csv")

    assert state_id == 0, "Hover状态解析错误"
    assert fault_type == 0, "Motor故障类型解析错误"
    assert case_id == 12345, "案例ID解析错误"
    print("✓ 文件名解析测试通过")


def test_load_hil_data():
    """测试HIL数据加载"""
    loader = HILDatasetLoader()
    config = loader.config

    # 假设数据集存在
    cases, labels = loader.load_hil_data(state_id=0)

    assert len(cases) > 0, "未加载到数据"
    assert len(cases) == len(labels), "数据和标签数量不匹配"
    assert isinstance(cases[0], np.ndarray), "案例应该是numpy数组"
    print(f"✓ 数据加载测试通过: 加载了{len(cases)}个案例")


if __name__ == '__main__':
    test_parse_filename()
    test_load_hil_data()
    print("\n所有测试通过!")
