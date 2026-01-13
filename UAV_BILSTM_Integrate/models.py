"""
models.py
=========
复现论文 “Ensemble Transfer Learning Based Cross-Domain UAV Actuator Fault Detection” 的核心模型与集成/迁移逻辑。

包含组件：
1) BiLSTM_Predictor：两层堆叠 BiLSTM + 全连接回归头（预测执行器输出的“正常行为”）
2) TransferLearningManager：冻结层/微调训练循环（面向目标域）
3) R_Score_Calculator：对应论文公式 (3)，使用 MIC（minepy）计算源域/目标域相关性差异
4) Ensemble_Predictor：对应论文公式 (4)(5) 的加权集成，并在注释中说明权重逻辑的潜在矛盾

说明（很重要）：
- 本复现任务中，“目标域无标签”通常指故障类型等分类标签缺失；但回归目标 PWM 来自传感器记录，
  因此依然可以用于“预测/残差”范式的训练或自监督校准（具体取决于你的实验设定）。
- 若你的目标域确实完全没有可用的 Y_T（例如只拿到状态量，没有 actuator 输出），则无法按论文公式 (3)
  直接计算 MIC[X_T^j, Y_T]；需要改成无监督相似度度量（不在本文实现范围内）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings


class BiLSTM_Predictor(nn.Module):
    """
    对应论文的 BiLSTM 回归预测器（用于预测执行器“正常输出”）。

    结构（按用户给定参数）：
    - 输入维度：input_dim=7
    - BiLSTM Layer 1：hidden=64, bidirectional=True
    - BiLSTM Layer 2：hidden=64, bidirectional=True
    - Dense：输出维度 output_dim=1（单个执行器通道）

    迁移学习需要“可冻结/可微调”的层级访问能力：
    - self.bilstm_layers 是 ModuleList，便于冻结底部若干层（例如只微调上层/FC）。
    """

    def __init__(
        self,
        *,
        input_dim: int = 7,
        hidden_size: int = 64,
        output_dim: int = 1,
        dropout: float = 0.0,
        head_hidden: int = 64,
        pooling: Literal["last", "mean"] = "last",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.output_dim = int(output_dim)
        self.pooling = pooling

        # 论文中的“堆叠 BiLSTM”，这里显式用两层 LSTM（num_layers=1）串联，便于冻结其中某层。
        self.bilstm_layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=self.input_dim,
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                ),
                nn.LSTM(
                    input_size=self.hidden_size * 2,
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                ),
            ]
        )

        # Dropout 放在时序输出上，等价于对特征做随机失活，降低过拟合（尤其源域->目标域时）。
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        # 回归头：将 BiLSTM 的双向输出（2*hidden）映射到 actuator 输出
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, int(head_hidden)),
            nn.ReLU(),
            nn.Linear(int(head_hidden), self.output_dim),
        )

    def get_lstm_layers(self) -> List[nn.Module]:
        return list(self.bilstm_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        return: (B, output_dim)
        """
        if x.ndim != 3:
            raise ValueError(f"BiLSTM_Predictor 期望输入 (B,T,F)，但得到 shape={tuple(x.shape)}")
        out = x
        for lstm in self.bilstm_layers:
            out, _ = lstm(out)  # out: (B,T,2H)
        out = self.dropout(out)

        if self.pooling == "mean":
            feat = out.mean(dim=1)  # (B,2H)
        else:  # last
            feat = out[:, -1, :]  # (B,2H)，对应“窗口末端时刻”的预测

        y_hat = self.head(feat)  # (B,output_dim)
        return y_hat


class TransferLearningManager:
    """
    迁移学习管理器：
    - freeze_layers：冻结底部 BiLSTM 层（对应论文“冻结部分层，仅微调高层”思想）
    - fine_tune：标准监督回归训练循环（用于目标域微调/校准）
    """

    @staticmethod
    def freeze_layers(model: BiLSTM_Predictor, num_layers_to_freeze: int = 1) -> None:
        """
        冻结底部若干 BiLSTM 层的参数（requires_grad=False）。

        注意论文表述可能存在歧义：
        - 有的版本写“冻结两层 BiLSTM，微调最后一层 BiLSTM + Dense”，这暗示至少 3 层；
        - 本复现模型为 2 层 BiLSTM，因此推荐：冻结第 1 层，仅微调第 2 层 + head。
        """
        layers = model.get_lstm_layers()
        k = max(0, min(int(num_layers_to_freeze), len(layers)))
        for layer in layers[:k]:
            for p in layer.parameters():
                p.requires_grad = False

    @staticmethod
    def unfreeze_all(model: nn.Module) -> None:
        for p in model.parameters():
            p.requires_grad = True

    @staticmethod
    def fine_tune(
        model: nn.Module,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        criterion: Optional[nn.Module] = None,
        grad_clip_norm: Optional[float] = 1.0,
        log_every: int = 0,
    ) -> Dict[str, List[float]]:
        """
        标准训练循环（回归任务）：
        - 默认 loss 使用 MSE；也可传入 SmoothL1Loss 等更鲁棒的损失。
        - 返回 history，便于论文画图（loss 曲线/过拟合分析）。
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        model = model.to(device)
        model.train()

        if criterion is None:
            criterion = nn.MSELoss()

        # 只优化 requires_grad=True 的参数（冻结层不会更新）
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(int(epochs)):
            model.train()
            total = 0.0
            n = 0
            for step, batch in enumerate(train_loader):
                # 兼容 Dataset 返回 (x,y) 或 (x,y,meta)
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    raise ValueError("train_loader batch 必须至少包含 (x,y)")

                x = x.to(device).float()
                y = y.to(device).float()

                optimizer.zero_grad(set_to_none=True)
                y_hat = model(x)

                # y 可能是 (B,1) 或 (B,T,1)（若你做 seq-to-seq）；这里默认与 y_hat 同形状
                if y.ndim > y_hat.ndim:
                    # 常见情况：y 是窗口序列，但模型输出单点 -> 默认取最后一帧对齐
                    y = y[:, -1, :]

                loss = criterion(y_hat, y)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, float(grad_clip_norm))
                optimizer.step()

                total += float(loss.detach().cpu().item()) * x.size(0)
                n += int(x.size(0))
                if log_every and (step % int(log_every) == 0):
                    pass

            train_loss = total / max(1, n)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                model.eval()
                v_total = 0.0
                v_n = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            x, y = batch[0], batch[1]
                        else:
                            raise ValueError("val_loader batch 必须至少包含 (x,y)")
                        x = x.to(device).float()
                        y = y.to(device).float()
                        y_hat = model(x)
                        if y.ndim > y_hat.ndim:
                            y = y[:, -1, :]
                        v_loss = criterion(y_hat, y)
                        v_total += float(v_loss.detach().cpu().item()) * x.size(0)
                        v_n += int(x.size(0))
                history["val_loss"].append(v_total / max(1, v_n))

        return history


def _to_numpy_1d(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.reshape(-1)


@dataclass
class R_Score_Calculator:
    """
    对应论文公式 (3) 的 R-score 计算器：

    公式 (3)：
      R_i = sum_{j=1..m} ( MIC[X_{S^i}^j, Y_{S^i}] - MIC[X_T^j, Y_T] )

    直观解释（中文）：
    - 对每个源域 i，分别计算“输入第 j 个特征 与 输出 Y 的非线性相关性（MIC）”；
    - 再与目标域相同特征-输出的相关性作差并求和；
    - R 越小代表“源域与目标域在 X->Y 关系上越相似”，更适合作为迁移源（对应论文的相似性度量）。
    """

    mine_alpha: float = 0.6
    mine_c: int = 15
    # method:
    # - "mic": 优先使用 minepy 的 MIC（对应论文）
    # - "mi" : 回退到 sklearn 的 mutual_info_regression（非 MIC，但同为非线性依赖度量）
    # - "pearson": 最保底的线性相关系数 |corr|
    method: Literal["mic", "mi", "pearson"] = "mic"

    def _mic(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算“特征-输出”之间的相关性度量。

        - 默认使用 minepy 的 MIC（对应论文公式 (3) 中的 MIC[·,·]）。
        - 但 Windows/新 Python 版本上 minepy 常缺 wheel，安装困难；为保证流程可运行，
          当 minepy 不可用时会自动回退到 MI 或 Pearson（并给出 warning）。
        """
        x1 = _to_numpy_1d(x)
        y1 = _to_numpy_1d(y)

        # 清理 NaN/Inf（否则 minepy 会报错或产生不稳定结果）
        mask = np.isfinite(x1) & np.isfinite(y1)
        x1 = x1[mask]
        y1 = y1[mask]
        if x1.size < 4:
            return 0.0

        # 1) 优先：minepy MIC（严格对应论文）
        if self.method == "mic":
            try:
                from minepy import MINE  # type: ignore

                mine = MINE(alpha=float(self.mine_alpha), c=int(self.mine_c))
                mine.compute_score(x1, y1)
                return float(mine.mic())
            except Exception:
                # 自动回退：不让整条实验流水线因为 minepy 缺失而中断
                warnings.warn(
                    "minepy 不可用，R-score 将回退到 mutual information / pearson；"
                    "如需严格复现论文 MIC，请安装 minepy（pip install minepy）。",
                    RuntimeWarning,
                )
                # 自动选择 mi（若 sklearn 不存在再 pearson）
                self.method = "mi"

        # 2) 回退：sklearn mutual information（连续变量）
        if self.method == "mi":
            try:
                from sklearn.feature_selection import mutual_info_regression  # type: ignore

                mi = mutual_info_regression(x1.reshape(-1, 1), y1, random_state=0)
                return float(mi[0])
            except Exception:
                warnings.warn(
                    "sklearn 不可用，R-score 将进一步回退到 Pearson 相关系数 |corr|。",
                    RuntimeWarning,
                )
                self.method = "pearson"

        # 3) 最保底：Pearson |corr|
        x1c = x1 - np.mean(x1)
        y1c = y1 - np.mean(y1)
        denom = (np.linalg.norm(x1c) * np.linalg.norm(y1c)) + 1e-12
        corr = float(np.dot(x1c, y1c) / denom)
        return float(abs(corr))

    def r_score(
        self,
        x_source: Union[np.ndarray, torch.Tensor],
        y_source: Union[np.ndarray, torch.Tensor],
        x_target: Union[np.ndarray, torch.Tensor],
        y_target: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """
        计算单个源域 i 的 R_i。

        输入形状建议：
        - x_*: (N, m) 或 (N, T, m) 其中 m=特征数
        - y_*: (N, 1) 或 (N, T, 1)

        为了与 MIC 的定义一致，这里将时序维展平，把每个时间点当作独立样本（信号处理上等价于统计相关性）。
        """
        xs = np.asarray(x_source.detach().cpu().numpy() if isinstance(x_source, torch.Tensor) else x_source)
        ys = np.asarray(y_source.detach().cpu().numpy() if isinstance(y_source, torch.Tensor) else y_source)
        xt = np.asarray(x_target.detach().cpu().numpy() if isinstance(x_target, torch.Tensor) else x_target)
        yt = np.asarray(y_target.detach().cpu().numpy() if isinstance(y_target, torch.Tensor) else y_target)

        if xs.ndim == 3:
            xs = xs.reshape(-1, xs.shape[-1])
        if xt.ndim == 3:
            xt = xt.reshape(-1, xt.shape[-1])

        ys = ys.reshape(-1)
        yt = yt.reshape(-1)

        # 鲁棒对齐：如果 X 被展平成 (B*T, m) 但 Y 仍是 (B,)（例如窗口标签只取 last），
        # 则重复 Y 使其长度与 X 一致，避免广播错误。
        if xs.shape[0] != ys.shape[0] and ys.shape[0] > 0 and xs.shape[0] % ys.shape[0] == 0:
            rep = xs.shape[0] // ys.shape[0]
            ys = np.repeat(ys, rep)
        if xt.shape[0] != yt.shape[0] and yt.shape[0] > 0 and xt.shape[0] % yt.shape[0] == 0:
            rep = xt.shape[0] // yt.shape[0]
            yt = np.repeat(yt, rep)

        if xs.shape[1] != xt.shape[1]:
            raise ValueError(f"源域与目标域特征维度不一致：{xs.shape[1]} vs {xt.shape[1]}")

        r = 0.0
        m = xs.shape[1]
        for j in range(m):
            mic_s = self._mic(xs[:, j], ys)
            mic_t = self._mic(xt[:, j], yt)
            r += (mic_s - mic_t)  # 对应论文公式 (3) 的括号项
        return float(r)

    def r_scores_for_sources(
        self,
        sources: Sequence[Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]],
        target: Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]],
    ) -> List[float]:
        xt, yt = target
        scores: List[float] = []
        for xs, ys in sources:
            scores.append(self.r_score(xs, ys, xt, yt))
        return scores


class Ensemble_Predictor(nn.Module):
    """
    多源模型的加权集成预测器。

    论文公式：
    - (4) W_i = R_i^2 / sum(R^2)
    - (5) y_hat = sum_i W_i * f_i(x)

    关键注释（务必阅读）：
    - 论文文字通常会说 “R 越小，源域越相似”，因此更应给 *更小* 的 R 更大的权重；
    - 但公式 (4) 若直接使用 R^2，会导致 R 越大权重越大（与“相似性”直觉相反）。
    - 本实现默认采用 similarity 模式：W_i ∝ 1 / (R_i^2 + eps)，使得 R 越小权重越大；
      若你要严格按公式复现，可将 weight_mode='paper'。
    """

    def __init__(
        self,
        models: Sequence[nn.Module],
        *,
        weights: Optional[Sequence[float]] = None,
        weight_mode: Literal["similarity", "paper"] = "similarity",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("models 不能为空")
        self.models = nn.ModuleList(list(models))
        self.weight_mode = weight_mode
        self.eps = float(eps)

        if weights is None:
            # 默认等权
            w = torch.ones(len(self.models), dtype=torch.float32) / float(len(self.models))
        else:
            w = torch.tensor(list(weights), dtype=torch.float32)
            w = w / (w.sum() + self.eps)
        self.register_buffer("weights", w)

    @staticmethod
    def weights_from_r_scores(
        r_scores: Sequence[float],
        *,
        mode: Literal["similarity", "paper"] = "similarity",
        eps: float = 1e-8,
    ) -> List[float]:
        r = np.asarray(list(r_scores), dtype=np.float64)
        if r.size == 0:
            return []
        if mode == "paper":
            # 严格对应论文公式 (4)
            s = np.square(r)
        else:
            # 更符合“R 越小越相似”的加权直觉：W_i ∝ 1 / (R_i^2 + eps)
            s = 1.0 / (np.square(r) + float(eps))
        s = s / (np.sum(s) + float(eps))
        return [float(x) for x in s]

    def set_weights(self, weights: Sequence[float]) -> None:
        w = torch.tensor(list(weights), dtype=torch.float32, device=self.weights.device)
        w = w / (w.sum() + self.eps)
        self.weights.copy_(w)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,F)
        return: (B,output_dim)
        """
        preds: List[torch.Tensor] = []
        for m in self.models:
            preds.append(m(x))

        # (K,B,D)
        stack = torch.stack(preds, dim=0)
        w = self.weights.view(-1, 1, 1).to(stack.device, dtype=stack.dtype)
        y = (w * stack).sum(dim=0)
        return y


__all__ = [
    "BiLSTM_Predictor",
    "Ensemble_Predictor",
    "R_Score_Calculator",
    "TransferLearningManager",
]
