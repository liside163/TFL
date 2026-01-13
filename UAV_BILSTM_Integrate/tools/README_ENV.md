# 环境一键配置（Conda）

本项目的 PyTorch/CUDA 兼容性会受显卡架构影响：例如新架构 `sm_120` 在旧版 PyTorch wheel 中没有对应 kernel，会在 LSTM/CuDNN 处报 `no kernel image`。因此建议使用本脚本自动扫描硬件并选择 `cpu/cuda` 安装方案。

## 1) 一键安装（推荐）

在项目根目录运行：

```powershell
python tools/setup_env.py --env-name DTL_Learn --device auto
```

脚本会：
- 生成硬件报告：`env/hardware_report.json`
- 创建/更新 conda 环境 `DTL_Learn`
- 安装依赖：`pytorch`、`minepy`、`pandas`、`pyyaml`、`matplotlib`、`tensorboard` 等

## 2) 强制 CPU（避免 CUDA 架构不匹配）

```powershell
python tools/setup_env.py --env-name DTL_Learn --device cpu
```

## 3) 强制 CUDA（已有兼容 PyTorch 的情况下）

```powershell
python tools/setup_env.py --env-name DTL_Learn --device cuda --cuda 12.4
```

## 4) 运行训练

```powershell
conda activate DTL_Learn
python train_evaluate.py --config config.yaml
```

