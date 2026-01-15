"""
1D CNN模型测试
"""

import torch
from models.cnn_1d import CNN1D

def test_cnn_forward():
    """测试CNN前向传播"""
    model = CNN1D(input_dim=27, num_classes=11)

    # 创建输入: [batch, time, features]
    x = torch.randn(32, 100, 27)

    output = model(x)

    assert output.shape == (32, 11), f"输出shape应为(32, 11)，实际为{output.shape}"
    print(f"OK - CNN forward test passed: output shape={output.shape}")


def test_cnn_parameters():
    """测试CNN参数量"""
    model = CNN1D(input_dim=27, num_classes=11)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"CNN total parameters: {total_params:,}")

    # 参数量应该在180K左右
    assert 150000 < total_params < 200000, "参数量超出预期范围"


if __name__ == '__main__':
    test_cnn_forward()
    test_cnn_parameters()
    print("\nAll tests passed!")
