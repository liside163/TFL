---
description: Optuna调优后完整训练流程
---

# Optuna 调优 → 训练 → 报告 工作流

此工作流定义了从 Optuna 超参数优化到完整训练并生成报告的标准化流程。

## 前置条件

- 已准备好训练数据 (`python main.py preprocess`)
- 已安装 Optuna (`pip install optuna`)

---

## 步骤 1: 运行 Optuna 超参数调优

根据需要选择调优脚本和试验次数：

```bash
# 深度超参数优化 (推荐，搜索网络深度和学习率策略)
python optuna_tune_v2.py --config ./config/config.yaml --n_trials 100

# 或使用基础版本
python optuna_tune.py --config ./config/config.yaml --n_trials 50
```

**输出**：
- `optuna_results/dann_deep_hpo_best_params.yaml` - 最优参数
- `optuna_results/dann_deep_hpo_trials.csv` - 所有试验记录

---

## 步骤 2: 同步最佳参数到 config.yaml

// turbo
```bash
python scripts/sync_optuna_params.py --optuna_result ./optuna_results/dann_deep_hpo_best_params.yaml --config ./config/config.yaml
```

**说明**：自动将 Optuna 找到的最优参数更新到主配置文件中。

---

## 步骤 3: 运行完整训练

// turbo
```bash
python main.py all --config ./config/config.yaml
```

**包含**：
1. 数据预处理
2. 模型训练
3. 模型评估

---

## 步骤 4: 查看结果报告

训练完成后，结果保存在以下位置：

- **模型检查点**: `./checkpoints/`
- **训练日志**: `./logs/`
- **TensorBoard**: `./runs/`

查看 TensorBoard：
```bash
tensorboard --logdir ./runs
```

---

## 可选：单工况迁移实验

如需运行单工况迁移实验（每种故障类型单独迁移）：

```bash
# 运行所有故障类型的单工况实验
python run_all_conditions.py

# 仅运行特定故障类型
python run_all_conditions.py --fault_types 1 2

# 仅生成对比报告
python run_all_conditions.py --compare_only
```

**输出**：
- `./results/comparison_report.md` - 对比报告
- `./results/experiment_log.csv` - 实验记录表格

---

## 快速命令汇总

```bash
# 完整流程 (调优100次 → 同步 → 训练 → 报告)
python optuna_tune_v2.py --n_trials 100 && python scripts/sync_optuna_params.py && python main.py all
```
