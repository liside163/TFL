from __future__ import annotations

import torch
from torch import nn


class BiLSTMRegressor(nn.Module):
    """
    3 层 BiLSTM + 2 层 MLP 的多输出回归器。

    关键点（中文注释）：
    - 输入形状：[B, L, F]
    - BiLSTM 输出形状：[B, L, 2H]
    - 回归头默认取最后一个时间步的特征进行回归
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        lstm_layers: int = 3,
        dropout: float = 0.1,
        mlp_hidden: int = 128,
        output_dim: int = 4,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim 必须 > 0")
        if lstm_layers < 1:
            raise ValueError("lstm_layers 必须 >= 1")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.lstm_layers = int(lstm_layers)
        self.output_dim = int(output_dim)

        # 注意：PyTorch LSTM 的 dropout 只在 num_layers>1 时生效
        self.bilstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            dropout=float(dropout) if self.lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        feat_dim = 2 * self.hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, int(mlp_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden), int(mlp_hidden)),
            nn.ReLU(),
            nn.Linear(int(mlp_hidden), self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        # B: batch size，L: 时间步/窗口长度，F: 输入特征维度
        out, _ = self.bilstm(x)  # [B, L, 2H]
        # 关键维度变化：双向 LSTM 会把隐藏维度从 H 变为 2H（正向+反向拼接）
        last = out[:, -1, :]  # [B, 2H]
        # 关键维度变化：从序列维度 L 中取最后一个时间步，序列维度被“压缩掉”
        y = self.mlp(last)  # [B, C]
        # 关键维度变化：MLP 把 2H 映射到输出通道数 C
        return y


def freeze_first_n_bilstm_layers(model: BiLSTMRegressor, n: int) -> None:
    """
    冻结 BiLSTM 的前 n 层（用于迁移学习/微调）。
    """
    n = int(n)
    if n <= 0:
        return
    if n > model.lstm_layers:
        n = model.lstm_layers

    # PyTorch LSTM 的参数命名形如：weight_ih_l{k} / weight_hh_l{k}，
    # bidirectional 会额外出现 *_reverse。
    for name, param in model.bilstm.named_parameters():
        for layer_idx in range(n):
            if f"_l{layer_idx}" in name:
                param.requires_grad = False
                break
