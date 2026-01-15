"""
LSTM模型测试
"""

import torch
from models.lstm_model import LSTMModel

def test_lstm_forward():
    """测试LSTM前向传播"""
    model = LSTMModel(input_dim=27, num_classes=11)
    x = torch.randn(32, 100, 27)

    output = model(x)
    assert output.shape == (32, 11), f"输出shape应为(32, 11)"
    print("OK - LSTM forward test passed")


if __name__ == '__main__':
    test_lstm_forward()
