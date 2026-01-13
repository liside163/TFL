# UAV-DANN 实验修改记录表

## 项目概述
- **目标**: HIL仿真数据 → Real真实飞行数据的故障诊断迁移学习
- **方法**: DANN (Domain-Adversarial Neural Network)
- **数据集**: RflyMAD

---

## 修改记录汇总

| 序号 | 修改内容           | 修改原因                 | 实际效果              | 备注                        |
|:----:|-------------------|--------------------------|----------------------|----------------------------|
| 1    | HIL故障状态筛选    | HIL数据含正常+故障混合    | ✅ 数据质量提升        | 仅保留fault_state=1         |
| 2    | Optuna超参数优化   | 手动调参效率低            | 最佳34%准确率         | optuna_tune.py              |
| 3    | Real数据变点检测   | Real数据无fault_state列  | ✅ 已实现             | change_point_detection.py   |
| 4    | 11分类→7分类       | Real域缺少4种故障类型     | 待验证               | 移除4种Real缺失类型          |
| 5    | Batch Size调整     | 原16太小,GPU利用率低      | 显存占用↑            | 256→512                     |
| 6    | 多进程DataLoader   | 数据加载成瓶颈            | Windows有问题        | num_workers=4~8             |
| 7    | 深度Optuna调优     | 基础调优效果有限          | ✅ 已完成             | optuna_tune_v2.py           |
| 8    | Optuna参数同步     | 手动同步易出错            | ✅ lstm:64→192       | warmup_cosine调度器          |
| 9    | 单工况迁移学习     | 不同工况特性差异大         | 平均52.77%准确率     | 6种工况独立训练              |
| 10   | 单工况配置优化     | batch=64对小数据集过大    | velocity:14%→58%    | batch=32,epochs=150         |
| 11   | 单工况Optuna调优   | 各工况需独立调优          | 🔄 进行中            | optuna_tune_single_cond.py  |
| 12   | 单工况深度Optuna   | 通用结构不适配各工况       | 待验证               | 搜索CNN/LSTM层数            |
| 13   | 负迁移修复         | GRL启动后准确率暴跌       | 待验证               | 损失函数+超参数调整          |
| 14   | 类别不平衡修复     | Motor类占55%导致模型退化  | 待验证               | FocalLoss+类别权重          |
| 15   | 深度架构增强       | 缺少注意力和残差连接      | 待验证               | TemporalAttention+ResBlock  |
| 16   | 域对齐分析输出     | 无法评估迁移效果          | 已实现               | 训练时打印源/目标域准确率      |
| 17   | 方案A简化模型      | 复杂模型效果差+过拟合     | 待验证               | 关闭注意力/残差+保守训练策略   |
| 18   | 源域崩塌诊断       | 训练集准确率仅4%          | ⚠️ 诊断出双重惩罚     | 全预测Motor(66%)但召回=0    |
| 19   | 方案B: 加权采样    | 解决类别不平衡            | ❌ 源域崩塌           | 加权采样+类别权重=双重惩罚  |
| 20   | 方案C: Focal Loss | 替代类别权重损失          | ❌ 源域崩塌            | 禁用损失权重，使用加权采样  |
---

## 关键实验结果

### 基线 vs Optuna优化

| 阶段         | 源域(HIL) | 目标域(Real) |
|-------------|-----------|--------------|
| 基线         | ~78%      | ~16%         |
| Optuna优化后 | ~78%      | **~34%**     |

### 单工况迁移结果

| 工况     | 准确率     | F1   | 备注        |
|----------|-----------|------|------------|
| hover    | **59.46%** | 0.20 | 最佳        |
| waypoint | 47.61%    | 0.29 | -           |
| velocity | **58.50%** | 0.02 | 优化后+43%  |
| circling | 52.97%    | 0.23 | -           |
| acce     | 45.30%    | 0.10 | -           |
| dece     | -         | -    | 未运行      |



---

## 7分类故障类型对照


| 标签 | 故障类型      | 编码 | HIL | Real |
|:----:|--------------|:----:|:---:|:----:|
| 0    | No_Fault     | 10   | ✅  | ✅   |
| 1    | Motor        | 00   | ✅  | ✅   |
| 2    | Accelerometer| 05   | ✅  | ✅   |
| 3    | Gyroscope    | 06   | ✅  | ✅   |
| 4    | Magnetometer | 07   | ✅  | ✅   |
| 5    | Barometer    | 08   | ✅  | ✅   |
| 6    | GPS          | 09   | ✅  | ✅   |

**已移除**: Propeller(01), Low_Voltage(02), Wind_Affect(03), Load_Lose(04) - Real域缺失

---

## 待尝试方向

| 方向         | 具体方法                    | 优先级 | 状态       |
|-------------|----------------------------|:------:|:----------:|
| 网络结构     | 增加CNN/LSTM层数            | 高     | ✅ 已完成   |
| 学习率策略   | warmup + cosine annealing   | 高     | ✅ 已应用   |
| 域适应权重   | 动态domain_loss_weight      | 高     | ✅ 已优化   |
| 单工况Optuna | 每工况独立调优 (含结构)     | 高     | ✅ 已扩展   |
| 数据增强     | 时间序列增强                | 中     | 未开始      |
| 特征工程     | 增加/减少传感器特征          | 中     | 未开始      |
| 其他DA方法   | MMD/CORAL/ADDA              | 低     | 未开始      |

---

## 文件修改清单

| 文件                               | 类型   | 说明             |
|-----------------------------------|--------|-----------------|
| `config/config.yaml`              | [修改] | 7分类配置        |
| `data/preprocess.py`              | [修改] | 故障筛选         |
| `data/change_point_detection.py`  | [新增] | 变点检测         |
| `optuna_tune.py`                  | [新增] | 基础超参优化     |
| `optuna_tune_v2.py`               | [新增] | 深度超参优化     |
| `config/config_single_condition.yaml` | [新增] | 单工况配置   |
| `train_single_condition.py`       | [新增] | 单工况训练       |
| `run_all_conditions.py`           | [新增] | 批量运行         |
| `optuna_tune_single_condition.py` | [扩展] | 深度结构搜索     |
| `scripts/sync_single_condition_params.py` | [修改] | 结构参数同步     |
| `models/dann_deep.py`             | [新增] | 动态DANN模型     |
| `scripts/sync_optuna_params.py`   | [新增] | 参数同步         |

---

*最后更新: 2026-01-07*
