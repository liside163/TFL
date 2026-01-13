# UAV-DANN: 基于域对抗迁移学习的无人机故障诊断

## 项目概述

本项目实现了基于DANN (Domain-Adversarial Neural Network) 的无人机故障诊断迁移学习方法，用于解决HIL仿真数据到Real真实飞行数据的域偏移问题。

### 研究目标
- **源域**: HIL仿真数据（有标签，~2566例）
- **目标域**: Real真实飞行数据（无标签/少量标签，~497例）
- **任务**: 利用丰富的HIL数据训练模型，使其能够有效诊断Real数据中的故障

### 故障类型
共11种故障类型：
1. No Fault (无故障)
2. Motor (电机故障)
3. Propeller (螺旋桨故障)
4. Low Voltage (低电压)
5. Wind Affect (风干扰)
6. Load Lose (负载丢失)
7. Accelerometer (加速度计故障)
8. Gyroscope (陀螺仪故障)
9. Magnetometer (磁力计故障)
10. Barometer (气压计故障)
11. GPS (GPS故障)

## 项目结构

```
UAV_DANN/
├── config/
│   └── config.yaml          # 所有超参数和配置
├── data/
│   ├── __init__.py
│   ├── preprocess.py        # 数据预处理
│   └── dataloader.py        # 数据加载器
├── models/
│   ├── __init__.py
│   ├── layers.py            # 自定义层（GRL梯度反转层）
│   └── dann.py              # DANN模型
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # 评估指标
│   └── logger.py            # 日志记录
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── main.py                  # 主入口
├── requirements.txt         # 依赖
└── README.md                # 说明文档
```

## 环境配置

### 系统要求
- Python 3.8+
- CUDA 11.x (RTX 5070Ti)
- 16GB+ RAM

### 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 配置数据路径

编辑 `config/config.yaml`，修改数据路径：

```yaml
data:
  data_root: "D:/DL_LEARN/Dataset/Processdata"
  source_domain: "HIL"
  target_domain: "REAL"
```

### 2. 数据预处理

```bash
python main.py preprocess --config ./config/config.yaml
```

### 3. 训练模型

```bash
python main.py train --config ./config/config.yaml
```

### 4. 评估模型

```bash
python main.py evaluate --config ./config/config.yaml --checkpoint ./checkpoints/best.pth
```

### 5. 完整流程（一键执行）

```bash
python main.py all --config ./config/config.yaml
```

## 模型架构

### DANN网络结构

```
输入 (B, 100, 21)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                    特征提取器                       │
│  Conv1d(21→64) → Pool → Conv1d(64→128) → Pool     │
│  → LSTM(128, 2层) → 特征向量(128维)                │
└─────────────────────────────────────────────────────┘
      │
      ├──────────────────┬───────────────────┐
      ▼                  ▼                   │
┌────────────┐    ┌────────────┐            │
│ 故障分类器  │    │   GRL层    │            │
│ FC(128→64) │    │ 梯度反转   │            │
│ FC(64→11)  │    └─────┬──────┘            │
└─────┬──────┘          ▼                   │
      │          ┌────────────┐             │
      │          │ 域判别器   │             │
      │          │ FC(128→64) │             │
      │          │ FC(64→1)   │             │
      │          └─────┬──────┘             │
      ▼                ▼                    │
  故障预测(11类)   域判别(HIL/Real)         │
```

### 关键维度变化

| 阶段 | 维度 | 说明 |
|------|------|------|
| 输入 | (B, 100, 21) | batch×时间步×特征 |
| Conv1 | (B, 64, 100) | 64个卷积核 |
| Pool1 | (B, 64, 50) | 时间维度减半 |
| Conv2 | (B, 128, 50) | 128个卷积核 |
| Pool2 | (B, 128, 25) | 时间维度再减半 |
| LSTM | (B, 128) | 取最后隐藏状态 |
| 分类输出 | (B, 11) | 11种故障类别 |
| 域判别输出 | (B, 1) | 二分类 |

## 训练策略

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda \cdot \mathcal{L}_{domain}$$

- $\mathcal{L}_{cls}$: 交叉熵分类损失（仅源域）
- $\mathcal{L}_{domain}$: 二元交叉熵域判别损失

### GRL系数调度

$$\lambda(p) = \frac{2}{1 + e^{-\gamma p}} - 1$$

其中 $p = epoch / total\_epochs$

| 训练阶段 | Epoch | λ值 | 说明 |
|----------|-------|-----|------|
| 预热 | 1-10 | 0 | 仅分类训练 |
| 适应 | 11-100 | 0→1 | 逐渐增加域适应 |

## 配置说明

所有超参数均可在 `config/config.yaml` 中修改：

### 关键配置项

```yaml
# 训练参数
training:
  batch_size: 32
  num_epochs: 100
  optimizer:
    learning_rate: 0.001

# 模型架构
model:
  feature_extractor:
    lstm:
      hidden_size: 128
      num_layers: 2

# 域适应
  domain_adaptation:
    warmup_epochs: 10
    gamma_grl: 10.0
```

## 预期结果

| 指标 | 基线（无迁移） | DANN |
|------|---------------|------|
| 目标域准确率 | ~50% | **>75%** |
| F1-Score | ~0.45 | **>0.70** |

## 可视化

训练过程会自动生成：
- 训练曲线 (`logs/training_curves.png`)
- 混淆矩阵 (`logs/confusion_matrix.png`)
- 特征t-SNE可视化 (`logs/feature_visualization_tsne.png`)

使用TensorBoard查看详细日志：

```bash
tensorboard --logdir=./runs
```

## 常见问题

### Q: CUDA内存不足

A: 减小 `batch_size` 或 `lstm.hidden_size`

### Q: 训练不收敛

A: 
- 检查数据路径是否正确
- 降低学习率
- 增加预热轮数

### Q: 目标域准确率低

A:
- 增加训练轮数
- 调整GRL系数
- 检查特征选择是否合理

## 参考文献

1. Ganin, Y., et al. "Domain-adversarial training of neural networks." JMLR 2016.
2. RflyMAD Dataset: Multi-rotor UAV Fault Detection Dataset

## 许可证

MIT License
