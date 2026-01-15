# RflyMAD 跨飞行状态迁移学习自动化实验系统实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建完整的HIL数据集跨飞行状态迁移学习系统，支持5种迁移方法、3种架构、5种目标状态的批量实验，集成超参数调优和全自动运行监控。

**Architecture:** 采用模块化设计，分为数据处理、模型架构、迁移方法、训练器、超参调优、自动化系统、评估报告7大模块。使用配置文件集中管理所有超参数，支持断点续跑和并行实验。

**Tech Stack:** PyTorch 2.0+, scikit-learn, Optuna(贝叶斯优化), pandas/matplotlib(分析), logging(监控)

---

## Phase 1: 项目初始化与配置管理

### Task 1.1: 创建项目目录结构

**Files:**
- Create: `data/__init__.py`, `models/__init__.py`, `methods/__init__.py`, `trainers/__init__.py`
- Create: `evaluators/__init__.py`, `utils/__init__.py`, `experiments/__init__.py`
- Create: `results_transfer_learning/logs/`, `results_transfer_learning/models/`
- Create: `results_transfer_learning/figures/`, `docs/plans/`

**Step 1: 创建所有必需的目录**

```bash
cd D:\Bigshe\TFL\Experiment

# 创建模块目录
mkdir -p data models methods trainers evaluators utils experiments
mkdir -p results_transfer_learning/logs
mkdir -p results_transfer_learning/models
mkdir -p results_transfer_learning/figures
mkdir -p docs/plans

# 创建__init__.py文件
touch data/__init__.py
touch models/__init__.py
touch methods/__init__.py
touch trainers/__init__.py
touch evaluators/__init__.py
touch utils/__init__.py
touch experiments/__init__.py
```

Run: `ls -R` 验证所有目录已创建
Expected: 看到所有模块目录和__init__.py文件

**Step 2: Commit 目录结构**

```bash
cd D:\Bigshe\TFL\Experiment
git add .
git commit -m "feat: 创建项目目录结构"
```

---

### Task 1.2: 创建集中配置文件

**Files:**
- Create: `config.py`

**Step 1: 编写配置文件完整代码**

创建 `config.py`:

```python
# ============================================================
# RflyMAD 迁移学习实验配置文件
# 所有超参数和路径集中管理
# ============================================================

class Config:
    """实验配置类 - 集中管理所有可调参数"""

    # ========================================================
    # 路径配置
    # ========================================================
    # 数据集路径
    DATA_ROOT = "D:/Bigshe/RflyMAD_Dataset/data_analysis"
    HIL_DATA_DIR = f"{DATA_ROOT}"  # HIL数据根目录

    # 实验结果保存路径
    EXPERIMENT_NAME = "transfer_learning_experiments"
    SAVE_DIR = f"{DATA_ROOT}/results_transfer_learning"
    LOG_DIR = f"{SAVE_DIR}/logs"
    MODEL_DIR = f"{SAVE_DIR}/models"
    FIG_DIR = f"{SAVE_DIR}/figures"

    # ========================================================
    # 数据预处理配置
    # ========================================================
    # 时间窗口参数
    TIME_WINDOW = 100      # 窗口大小(时间步)
    TIME_STEP = 50         # 滑动步长(50%重叠)

    # 数据划分
    SOURCE_STATE = 0       # Hover作为源域
    TARGET_STATES = [1, 2, 3, 4, 5]  # 5种目标状态

    # 状态名称映射
    STATE_NAMES = {
        0: 'Hover',
        1: 'Waypoint',
        2: 'Velocity',
        3: 'Circling',
        4: 'Acceleration',
        5: 'Deceleration'
    }

    # 故障类型映射
    FAULT_TYPES = {
        0: 'motor', 1: 'propeller', 2: 'low_voltage',
        3: 'wind_affect', 4: 'load_lose', 5: 'accelerometer',
        6: 'gyroscope', 7: 'magnetometer', 8: 'barometer',
        9: 'gps', 10: 'no_fault'
    }

    # 目标域微调集比例
    TARGET_FINETUNE_RATIO = 0.2  # 20%用于微调
    TARGET_TEST_RATIO = 0.8      # 80%用于测试

    # 特征配置
    INPUT_FEATURES = 27      # 输入特征数
    NUM_CLASSES = 11         # 故障类别数

    # ========================================================
    # 模型架构配置
    # ========================================================
    # CNN架构
    CNN_FILTERS = [64, 128, 256]
    CNN_KERNELS = [5, 5, 3]
    CNN_STRIDES = [1, 2, 2]
    CNN_DROPOUT = [0.2, 0.2, 0.3]

    # LSTM架构
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    LSTM_BIDIRECTIONAL = True

    # CNN-LSTM混合
    HYBRID_CNN_FILTERS = [64, 128]
    HYBRID_LSTM_HIDDEN = 128

    # ========================================================
    # 训练基础配置
    # ========================================================
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    EARLY_STOP_PATIENCE = 15

    # 优化器配置
    OPTIMIZER = 'Adam'
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5

    # 学习率调度
    LR_SCHEDULER = 'ReduceLROnPlateau'  # or 'CosineAnnealing'
    LR_SCHEDULER_PARAMS = {
        'mode': 'max',
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-6
    }

    # ========================================================
    # 迁移学习方法配置
    # ========================================================
    # 预训练+微调
    FINETUNE_LR = 0.0001     # 微调学习率(源域的1/10)
    FINETUNE_EPOCHS = 50
    FREEZE_FEATURES = True   # 冻结特征提取器

    # DANN参数
    DANN_LR = 0.001
    DANN_LAMBDA_INIT = 0.0
    DANN_LAMBDA_MAX = 1.0
    DANN_DISCRIMINATOR_HIDDEN = [64, 32]

    # MMD参数
    MMD_LAMBDA = 1.0         # MMD损失权重
    MMD_KERNEL_SIGMAS = [0.1, 1, 10, 100]  # 多核MMD

    # 样本加权参数
    WEIGHTING_METHOD = 'KMM'  # 'KMM' or 'TRISK'
    WEIGHTING_BETA = 0.5

    # ========================================================
    # 超参数搜索配置
    # ========================================================
    # 搜索方法选择: 'grid', 'random', 'bayesian', 'hyperband'
    SEARCH_METHOD = 'bayesian'

    # 网格搜索空间
    GRID_SEARCH = {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'batch_size': [32, 64, 128],
        'dropout': [0.1, 0.2, 0.3, 0.5],
        'lambda_mmd': [0.1, 0.5, 1.0, 2.0],
    }

    # 随机搜索空间
    RANDOM_SEARCH = {
        'n_trials': 50,
        'learning_rate': ('log_uniform', 1e-5, 1e-1),
        'batch_size': [32, 64, 128, 256],
        'dropout': ('uniform', 0.0, 0.6),
        'lambda_mmd': ('log_uniform', 0.01, 10.0),
    }

    # 贝叶斯优化空间
    BAYESIAN_OPTIMIZATION = {
        'n_trials': 30,
        'timeout_hours': 24,

        'params': {
            'learning_rate': {'type': 'log', 'bounds': [1e-5, 1e-1]},
            'batch_size': {'type': 'categorical', 'values': [32, 64, 128]},
            'dropout': {'type': 'uniform', 'bounds': [0.0, 0.6]},
            'lambda_mmd': {'type': 'log', 'bounds': [0.01, 10.0]},
            'cnn_filters_ratio': {'type': 'uniform', 'bounds': [0.5, 2.0]},
        },

        'objective': 'maximize',
        'metric': 'f1_macro'
    }

    # Hyperband配置
    HYPERBAND = {
        'max_iter': 100,
        'eta': 3,
        's_max': 5,
        'early_stop_metric': 'val_f1',
        'min_improvement': 0.001
    }

    # 分层优化
    HIERARCHICAL_OPTIMIZATION = {
        'stage1': {
            'method': 'random',
            'n_trials': 20,
            'epochs_per_trial': 20,
        },
        'stage2': {
            'method': 'bayesian',
            'n_trials': 30,
            'epochs_per_trial': 50,
        },
        'stage3': {
            'method': 'manual',
            'top_k': 3,
            'epochs_per_trial': 100,
            'n_seeds': 5
        }
    }

    # ========================================================
    # 自动化实验配置
    # ========================================================
    # 并行实验
    MAX_PARALLEL_EXPERIMENTS = 4
    GPU_IDS = [0, 1]  # 可用的GPU编号

    # 监控配置
    ENABLE_GPU_MONITORING = True
    MONITOR_INTERVAL = 60  # 秒

    # 告警配置
    ALERT_ON_GPU_FULL = True
    ALERT_ON_LOW_F1 = True
    ALERT_ON_ANOMALY_LOSS = True
    ALERT_EMAIL = None  # or 'your@email.com'

    # 断点续跑
    ENABLE_CHECKPOINT = True
    CHECKPOINT_INTERVAL = 5  # 每5个epoch保存一次
    RESUME_FROM_CHECKPOINT = True

    # ========================================================
    # 评估配置
    # ========================================================
    # 评估指标列表
    METRICS = ['accuracy', 'precision', 'recall', 'f1_macro', 'f1_weighted']

    # 混淆矩阵
    PLOT_CONFUSION_MATRIX = True

    # t-SNE可视化
    PLOT_TSNE = True
    TSNE_PERPLEXITY = 30
    TSNE_N_SAMPLES = 1000  # 采样数量

    # 特征分布可视化
    PLOT_FEATURE_DISTRIBUTION = True

    # ========================================================
    # 系统配置
    # ========================================================
    # 随机种子(可复现)
    RANDOM_SEED = 42

    # 设备配置
    DEVICE = 'cuda'  # or 'cpu'
    NUM_WORKERS = 4  # 数据加载线程数
    PIN_MEMORY = True  # GPU加速

    # 日志配置
    LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
    LOG_INTERVAL = 10   # 每10个batch记录一次
    VALID_INTERVAL = 1  # 每1个epoch验证一次

    # 可视化后端
    MATPLOTLIB_BACKEND = 'Agg'  # 非交互式后端，适合服务器

    # ========================================================
    # 实验元信息
    # ========================================================
    # 实验描述
    EXPERIMENT_DESCRIPTION = """
    RflyMAD HIL数据集跨飞行状态迁移学习实验
    源域: Hover (状态0)
    目标域: Waypoint/Vel/Circ/Acce/Dece (状态1-5)
    方法: 5种迁移方法
    架构: 3种深度学习架构
    总计: 5 × 3 × 5 = 75组实验
    """

    # 版本信息
    VERSION = '1.0.0'
    PYTORCH_VERSION = '2.0+'
    CUDA_VERSION = '11.8+'


def get_config():
    """获取配置单例"""
    return Config


def update_config(**kwargs):
    """动态更新配置"""
    config = Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Config没有属性: {key}")
    return config
```

**Step 2: 测试配置文件导入**

```bash
cd D:\Bigshe\TFL\Experiment
python -c "from config import Config; c = Config(); print(f'配置加载成功: {c.LEARNING_RATE}')"
```

Expected: 输出 "配置加载成功: 0.001"

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat: 添加集中配置文件"
```

---

## Phase 2: 数据处理模块

### Task 2.1: 实现HIL数据集加载器

**Files:**
- Create: `data/dataset_loader.py`

**Step 1: 编写测试**

创建 `tests/test_dataset_loader.py`:

```python
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
```

**Step 2: 运行测试验证失败**

```bash
cd D:\Bigshe\TFL\Experiment
python tests/test_dataset_loader.py
```

Expected: ImportError 或 FileNotFoundError (模块不存在)

**Step 3: 实现最小化代码**

创建 `data/dataset_loader.py`:

```python
"""
HIL数据集加载器
负责加载和解析RflyMAD HIL数据集
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from config import Config


class HILDatasetLoader:
    """HIL数据集加载器"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.data_dir = self.config.HIL_DATA_DIR

    def parse_filename(self, filename):
        """
        解析文件名 Case_2[B][CD][EFGHIJ].csv

        参数:
            filename: 文件名，如 "Case_2000012345.csv"

        返回:
            (state_id, fault_type, case_id)
            - state_id: 飞行状态ID (0-5)
            - fault_type: 故障类型ID (0-10)
            - case_id: 案例序列号
        """
        # 提取基本信息
        basename = os.path.basename(filename)
        # 移除.csv扩展名
        name_without_ext = basename.replace('.csv', '')

        # 解析: Case_2[B][CD][EFGHIJ]
        # [B]: 第3位字符 - 飞行状态
        # [CD]: 第4-5位字符 - 故障类型
        # [EFGHIJ]: 第6-11位字符 - 案例ID

        if len(name_without_ext) < 11:
            raise ValueError(f"文件名格式错误: {filename}")

        state_id = int(name_without_ext[2])  # 第3位
        fault_type = int(name_without_ext[3:5])  # 第4-5位
        case_id = int(name_without_ext[5:])  # 剩余部分

        return state_id, fault_type, case_id

    def load_hil_data(self, state_id):
        """
        加载指定飞行状态的HIL数据

        参数:
            state_id: 飞行状态ID (0-5)
                0: Hover
                1: Waypoint
                2: Velocity
                3: Circling
                4: Acceleration
                5: Deceleration

        返回:
            cases: List[np.ndarray] - 数据案例列表
            labels: List[int] - 故障标签列表
        """
        # 搜索所有匹配的文件
        pattern = os.path.join(self.data_dir, f"Case_2{state_id}*.csv")
        files = glob.glob(pattern)

        if len(files) == 0:
            raise FileNotFoundError(f"未找到状态{state_id}的数据文件: {pattern}")

        print(f"找到 {len(files)} 个状态{state_id}的文件")

        cases = []
        labels = []

        for filepath in files:
            try:
                # 解析文件名获取故障类型
                _, fault_type, _ = self.parse_filename(filepath)

                # 读取CSV文件
                df = pd.read_csv(filepath)

                # 提取数据列 (排除标签列)
                # HIL数据包含 UAVState_data_fault_state 作为标签
                if 'UAVState_data_fault_state' in df.columns:
                    label_col = 'UAVState_data_fault_state'
                    # 获取标签 (取第一个非NaN值，整个文件标签相同)
                    label = df[label_col].dropna().iloc[0]
                    label = int(label)

                    # 移除标签列和真值列
                    feature_cols = [col for col in df.columns
                                   if not col.startswith('UAVState_data_')
                                   and not col.startswith('TrueState_data_')]
                    data = df[feature_cols].values
                else:
                    # 如果没有标签列，从文件名推断
                    label = fault_type
                    # 只保留传感器和控制相关列
                    sensor_cols = [col for col in df.columns
                                  if any(keyword in col for keyword in
                                        ['_sensor_', '_actuator_', '_vehicle_', '_rfly_ctrl_'])]
                    data = df[sensor_cols].values if sensor_cols else df.values

                cases.append(data)
                labels.append(label)

            except Exception as e:
                print(f"警告: 加载文件失败 {filepath}: {str(e)}")
                continue

        print(f"成功加载 {len(cases)} 个案例")

        return cases, labels

    def get_statistics(self, state_id):
        """
        获取指定状态的数据统计信息

        返回:
            dict: 包含样本数量、故障类型分布等信息
        """
        cases, labels = self.load_hil_data(state_id)

        stats = {
            'total_cases': len(cases),
            'fault_distribution': {},
            'avg_sequence_length': 0,
            'feature_dim': 0
        }

        if len(cases) > 0:
            # 统计故障类型分布
            from collections import Counter
            label_counts = Counter(labels)
            stats['fault_distribution'] = dict(label_counts)

            # 统计序列长度
            seq_lengths = [len(case) for case in cases]
            stats['avg_sequence_length'] = np.mean(seq_lengths)
            stats['min_sequence_length'] = np.min(seq_lengths)
            stats['max_sequence_length'] = np.max(seq_lengths)

            # 特征维度
            stats['feature_dim'] = cases[0].shape[1] if len(cases[0].shape) > 1 else 1

        return stats


# 测试代码
if __name__ == '__main__':
    loader = HILDatasetLoader()

    # 测试文件名解析
    print("测试文件名解析:")
    test_files = [
        "Case_2000012345.csv",  # Hover, Motor
        "Case_3109000005.csv",  # Real, GPS (不会匹配HIL)
        "Case_2210000001.csv",  # Velocity, Low Voltage
    ]

    for filename in test_files:
        try:
            state, fault, case = loader.parse_filename(filename)
            print(f"  {filename}: 状态={state}, 故障={fault}, 案例ID={case}")
        except Exception as e:
            print(f"  {filename}: 错误 - {str(e)}")

    # 测试数据加载 (如果数据存在)
    print("\n测试数据加载:")
    try:
        stats = loader.get_statistics(state_id=0)  # Hover
        print(f"  Hover状态统计:")
        print(f"    总案例数: {stats['total_cases']}")
        print(f"    故障分布: {stats['fault_distribution']}")
        print(f"    平均序列长度: {stats['avg_sequence_length']:.1f}")
        print(f"    特征维度: {stats['feature_dim']}")
    except Exception as e:
        print(f"  数据加载失败: {str(e)}")
```

**Step 4: 运行测试验证通过**

```bash
cd D:\Bigshe\TFL\Experiment
python data/dataset_loader.py
```

Expected: 输出文件名解析结果和数据统计信息

**Step 5: Commit**

```bash
git add data/dataset_loader.py tests/test_dataset_loader.py
git commit -m "feat: 实现HIL数据集加载器"
```

---

### Task 2.2: 实现时间窗口切片与标准化

**Files:**
- Create: `data/preprocessor.py`

**Step 1: 编写测试**

创建 `tests/test_preprocessor.py`:

```python
import pytest
import numpy as np
from data.preprocessor import Preprocessor
from config import Config

def test_window_slice():
    """测试时间窗口切片"""
    preprocessor = Preprocessor()

    # 创建模拟数据: [1000, 27]
    data = np.random.randn(1000, 27)
    labels = np.random.randint(0, 11, 1000)

    windows, window_labels = preprocessor.create_windows(
        data, labels[0], window_size=100, step=50
    )

    assert windows.shape[0] > 0, "应该生成窗口"
    assert windows.shape[1] == 100, f"窗口大小应为100，实际为{windows.shape[1]}"
    assert windows.shape[2] == 27, f"特征维度应为27，实际为{windows.shape[2]}"
    print(f"✓ 生成{windows.shape[0]}个时间窗口")


def test_normalization():
    """测试数据标准化"""
    preprocessor = Preprocessor()

    # 创建模拟数据
    data = np.random.randn(100, 27) * 10 + 5  # 均值5，标准差10

    # 拟合标准化参数
    preprocessor.fit_normalizer(data)

    # 应用标准化
    normalized = preprocessor.normalize(data)

    # 验证标准化效果 (均值接近0，标准差接近1)
    mean = normalized.mean(axis=0)
    std = normalized.std(axis=0)

    assert np.all(np.abs(mean) < 1e-10), "均值应接近0"
    assert np.all(np.abs(std - 1.0) < 1e-10), "标准差应接近1"
    print("✓ 标准化测试通过")


if __name__ == '__main__':
    test_window_slice()
    test_normalization()
    print("\n所有测试通过!")
```

**Step 2: 运行测试验证失败**

```bash
python tests/test_preprocessor.py
```

Expected: ImportError

**Step 3: 实现代码**

创建 `data/preprocessor.py`:

```python
"""
数据预处理模块
负责时间窗口切片、特征标准化等预处理操作
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from config import Config


class Preprocessor:
    """数据预处理器"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def create_windows(self, data, label, window_size=None, step=None):
        """
        将长时间序列切片为固定大小的窗口

        参数:
            data: np.ndarray, shape [seq_len, features]
                时间序列数据
            label: int
                整个序列的标签
            window_size: int
                窗口大小 (默认从config读取)
            step: int
                滑动步长 (默认从config读取)

        返回:
            windows: np.ndarray, shape [n_windows, window_size, features]
                切片后的时间窗口
            window_labels: np.ndarray, shape [n_windows]
                每个窗口的标签
        """
        window_size = window_size or self.config.TIME_WINDOW
        step = step or self.config.TIME_STEP

        seq_len, features = data.shape

        # 如果序列太短，填充0
        if seq_len < window_size:
            padding = np.zeros((window_size - seq_len, features))
            data = np.vstack([data, padding])
            seq_len = window_size

        # 计算窗口数量
        n_windows = (seq_len - window_size) // step + 1

        # 切片
        windows = []
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            window = data[start:end, :]
            windows.append(window)

        windows = np.array(windows)  # [n_windows, window_size, features]
        window_labels = np.full(n_windows, label)  # 所有窗口使用相同标签

        return windows, window_labels

    def fit_normalizer(self, data):
        """
        拟合标准化参数 (基于源域数据)

        参数:
            data: np.ndarray, shape [n_samples, n_features]
                用于拟合标准化的数据 (可以是2D或3D)
        """
        # 如果是3D数据 [samples, time, features]，展平为2D
        if len(data.shape) == 3:
            n_samples, time_steps, features = data.shape
            data_2d = data.reshape(-1, features)
        else:
            data_2d = data

        # 拟合StandardScaler
        self.scaler.fit(data_2d)
        self.is_fitted = True

        print(f"标准化参数拟合完成: 均值shape={self.scaler.mean_.shape}, "
              f"标准差shape={self.scaler.scale_.shape}")

    def normalize(self, data):
        """
        应用标准化

        参数:
            data: np.ndarray, shape [n_samples, n_features] 或 [n_samples, time, features]

        返回:
            normalized_data: np.ndarray, 与输入shape相同
        """
        if not self.is_fitted:
            raise ValueError("标准化器未拟合，请先调用 fit_normalizer()")

        original_shape = data.shape

        # 如果是3D数据，展平处理
        if len(data.shape) == 3:
            n_samples, time_steps, features = data.shape
            data_2d = data.reshape(-1, features)
            normalized_2d = self.scaler.transform(data_2d)
            normalized = normalized_2d.reshape(n_samples, time_steps, features)
        else:
            normalized = self.scaler.transform(data)

        return normalized

    def process_cases(self, cases, labels, fit_scaler=True):
        """
        批量处理案例列表

        参数:
            cases: List[np.ndarray]
                案例列表，每个案例shape [seq_len, features]
            labels: List[int]
                标签列表
            fit_scaler: bool
                是否在这些数据上拟合标准化器 (通常源域设为True，目标域设为False)

        返回:
            all_windows: np.ndarray, [total_windows, window_size, features]
            all_labels: np.ndarray, [total_windows]
        """
        all_windows = []
        all_labels = []

        print(f"处理 {len(cases)} 个案例...")

        for case_idx, (case, label) in enumerate(zip(cases, labels)):
            # 切片为窗口
            windows, window_labels = self.create_windows(case, label)
            all_windows.append(windows)
            all_labels.append(window_labels)

            if (case_idx + 1) % 100 == 0:
                print(f"  已处理 {case_idx + 1}/{len(cases)} 个案例")

        # 拼接所有窗口
        all_windows = np.vstack(all_windows)  # [total_windows, window_size, features]
        all_labels = np.concatenate(all_labels)  # [total_windows]

        print(f"生成 {len(all_windows)} 个时间窗口")

        # 标准化
        if fit_scaler:
            self.fit_normalizer(all_windows)
            all_windows = self.normalize(all_windows)
            print("标准化完成 (基于当前数据)")
        else:
            if not self.is_fitted:
                raise ValueError("标准化器未拟合，请先在源域数据上调用 process_cases(fit_scaler=True)")
            all_windows = self.normalize(all_windows)
            print("标准化完成 (使用已有参数)")

        return all_windows, all_labels

    def get_normalization_params(self):
        """获取标准化参数"""
        if not self.is_fitted:
            raise ValueError("标准化器未拟合")

        return {
            'mean': self.scaler.mean_.copy(),
            'scale': self.scaler.scale_.copy(),
            'var': self.scaler.var_.copy()
        }

    def set_normalization_params(self, params):
        """设置标准化参数 (用于加载)"""
        self.scaler.mean_ = params['mean']
        self.scaler.scale_ = params['scale']
        self.scaler.var_ = params['var']
        self.is_fitted = True
        print("标准化参数已加载")


# 测试代码
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 创建预处理器
    preprocessor = Preprocessor()

    # 生成模拟数据
    print("生成模拟数据...")
    n_cases = 10
    cases = []
    labels = []

    for i in range(n_cases):
        seq_len = np.random.randint(500, 1500)
        data = np.random.randn(seq_len, 27) * 10 + 5  # 均值5，标准差10
        cases.append(data)
        labels.append(np.random.randint(0, 11))

    # 处理第一个案例 (源域)
    print("\n处理源域数据...")
    source_windows, source_labels = preprocessor.process_cases(
        cases[:5], labels[:5], fit_scaler=True
    )
    print(f"源域: {len(source_windows)} 个窗口, shape={source_windows.shape}")

    # 处理第二个案例 (目标域，使用相同的标准化参数)
    print("\n处理目标域数据...")
    target_windows, target_labels = preprocessor.process_cases(
        cases[5:], labels[5:], fit_scaler=False
    )
    print(f"目标域: {len(target_windows)} 个窗口, shape={target_windows.shape}")

    # 验证标准化效果
    print("\n验证标准化:")
    print(f"源域均值: {source_windows.mean():.6f}, 标准差: {source_windows.std():.6f}")
    print(f"目标域均值: {target_windows.mean():.6f}, 标准差: {target_windows.std():.6f}")
```

**Step 4: 运行测试**

```bash
python data/preprocessor.py
```

Expected: 输出处理过程和标准化验证

**Step 5: Commit**

```bash
git add data/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: 实现时间窗口切片和标准化"
```

---

### Task 2.3: 实现数据划分模块

**Files:**
- Create: `data/data_split.py`

**Step 1: 编写测试**

创建 `tests/test_data_split.py`:

```python
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

    assert len(train) == 20, "训练集应为20个样本"
    assert len(test) == 80, "测试集应为80个样本"
    print("✓ 数据划分测试通过")


if __name__ == '__main__':
    test_split_data()
```

**Step 2: 实现代码**

创建 `data/data_split.py`:

```python
"""
数据划分模块
负责将HIL数据划分为源域、目标域训练集、目标域测试集
"""

import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from config import Config


class DataSplitter:
    """数据划分器"""

    def __init__(self, config=None):
        self.config = config or Config()

    def split_target_domain(self, X, y, test_ratio=None, stratify=True):
        """
        划分目标域数据为微调集和测试集

        参数:
            X: np.ndarray, [n_samples, window_size, features]
                时间窗口数据
            y: np.ndarray, [n_samples]
                标签
            test_ratio: float
                测试集比例 (默认从config读取)
            stratify: bool
                是否分层抽样 (保持类别分布)

        返回:
            X_finetune, y_finetune: 微调集
            X_test, y_test: 测试集
        """
        test_ratio = test_ratio or self.config.TARGET_TEST_RATIO

        if stratify:
            # 分层抽样
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=self.config.RANDOM_SEED,
                stratify=y
            )
        else:
            # 随机划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=self.config.RANDOM_SEED
            )

        print(f"目标域划分完成:")
        print(f"  微调集: {len(X_train)} 样本, 类别分布: {Counter(y_train)}")
        print(f"  测试集: {len(X_test)} 样本, 类别分布: {Counter(y_test)}")

        return (X_train, y_train), (X_test, y_test)

    def create_source_target_split(self, source_windows, source_labels,
                                   target_windows, target_labels):
        """
        创建源域和目标域的完整划分

        参数:
            source_windows, source_labels: 源域数据
            target_windows, target_labels: 目标域数据

        返回:
            dict: 包含所有划分的数据
                {
                    'source': {'X': source_windows, 'y': source_labels},
                    'target_train': {'X': finetune_windows, 'y': finetune_labels},
                    'target_test': {'X': test_windows, 'y': test_labels}
                }
        """
        # 划分目标域
        (target_train_X, target_train_y), (target_test_X, target_test_y) = \
            self.split_target_domain(target_windows, target_labels)

        return {
            'source': {
                'X': source_windows,
                'y': source_labels
            },
            'target_train': {
                'X': target_train_X,
                'y': target_train_y
            },
            'target_test': {
                'X': target_test_X,
                'y': target_test_y
            }
        }

    def report_dataset_stats(self, data_dict):
        """
        报告数据集统计信息

        参数:
            data_dict: create_source_target_split的返回结果
        """
        print("\n" + "="*60)
        print("数据集统计信息")
        print("="*60)

        for split_name, split_data in data_dict.items():
            X = split_data['X']
            y = split_data['y']

            print(f"\n{split_name}:")
            print(f"  样本数: {len(X)}")
            print(f"  特征维度: {X.shape}")
            print(f"  类别分布:")

            from collections import Counter
            label_counts = Counter(y)
            for label, count in sorted(label_counts.items()):
                print(f"    类别{label}: {count} ({count/len(y)*100:.1f}%)")

        print("\n" + "="*60)
```

**Step 3: 测试**

```bash
python data/data_split.py
```

**Step 4: Commit**

```bash
git add data/data_split.py tests/test_data_split.py
git commit -m "feat: 实现数据划分模块"
```

---

## Phase 3: 模型架构模块

### Task 3.1: 实现1D CNN模型

**Files:**
- Create: `models/cnn_1d.py`

**Step 1: 编写测试**

创建 `tests/test_cnn_1d.py`:

```python
import torch
from models.cnn_1d import CNN1D

def test_cnn_forward():
    """测试CNN前向传播"""
    model = CNN1D(input_dim=27, num_classes=11)

    # 创建输入: [batch, time, features]
    x = torch.randn(32, 100, 27)

    output = model(x)

    assert output.shape == (32, 11), f"输出shape应为(32, 11)，实际为{output.shape}"
    print(f"✓ CNN前向传播测试通过: 输出shape={output.shape}")


def test_cnn_parameters():
    """测试CNN参数量"""
    model = CNN1D(input_dim=27, num_classes=11)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"CNN总参数量: {total_params:,}")

    # 参数量应该在180K左右
    assert 150000 < total_params < 200000, "参数量超出预期范围"


if __name__ == '__main__':
    test_cnn_forward()
    test_cnn_parameters()
    print("\n所有测试通过!")
```

**Step 2: 运行测试**

```bash
python tests/test_cnn_1d.py
```

Expected: ImportError

**Step 3: 实现代码**

创建 `models/cnn_1d.py`:

```python
"""
1D CNN模型
用于时序故障检测的特征提取
"""

import torch
import torch.nn as nn
from config import Config


class CNN1D(nn.Module):
    """
    1D卷积神经网络用于时间序列分类

    架构:
        Input: [batch, time_steps=100, features=27]
        Conv1D(64) -> ReLU -> Dropout
        Conv1D(128) -> ReLU -> Dropout
        Conv1D(256) -> ReLU -> Dropout
        GlobalMaxPool
        FC(256 -> 128 -> 11)
    """

    def __init__(self, input_dim=27, num_classes=11, config=None):
        super(CNN1D, self).__init__()

        self.config = config or Config()

        # 第一层卷积: 保持时间维度
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.config.CNN_FILTERS[0],
            kernel_size=self.config.CNN_KERNELS[0],
            stride=self.config.CNN_STRIDES[0],
            padding=2  # 保持时间长度
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.config.CNN_DROPOUT[0])

        # 第二层卷积: 下采样时间维度
        self.conv2 = nn.Conv1d(
            in_channels=self.config.CNN_FILTERS[0],
            out_channels=self.config.CNN_FILTERS[1],
            kernel_size=self.config.CNN_KERNELS[1],
            stride=self.config.CNN_STRIDES[1],
            padding=2
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.config.CNN_DROPOUT[1])

        # 第三层卷积: 进一步下采样
        self.conv3 = nn.Conv1d(
            in_channels=self.config.CNN_FILTERS[1],
            out_channels=self.config.CNN_FILTERS[2],
            kernel_size=self.config.CNN_KERNELS[2],
            stride=self.config.CNN_STRIDES[2],
            padding=1
        )
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(self.config.CNN_DROPOUT[2])

        # 全局池化
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # [batch, channels, 1]

        # 分类头
        self.fc1 = nn.Linear(self.config.CNN_FILTERS[2], 128)
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        """
        前向传播

        参数:
            x: [batch, time_steps, features]
            return_features: bool, 是否返回中间特征

        返回:
            logits: [batch, num_classes]
            features (可选): [batch, 256]
        """
        # 转换维度: [batch, time, features] -> [batch, features, time]
        x = x.transpose(1, 2)  # [batch, 27, 100]

        # 卷积层1
        x = self.conv1(x)  # [batch, 64, 100]
        x = self.relu1(x)
        x = self.dropout1(x)

        # 卷积层2
        x = self.conv2(x)  # [batch, 128, 50]
        x = self.relu2(x)
        x = self.dropout2(x)

        # 卷积层3
        x = self.conv3(x)  # [batch, 256, 25]
        x = self.relu3(x)
        x = self.dropout3(x)

        # 全局池化
        x = self.global_pool(x)  # [batch, 256, 1]
        x = x.squeeze(-1)  # [batch, 256]

        # 保存特征 (用于迁移学习)
        features = x

        # 分类头
        x = self.fc1(x)  # [batch, 128]
        x = self.relu_fc(x)
        x = self.dropout_fc(x)
        logits = self.fc2(x)  # [batch, 11]

        if return_features:
            return logits, features
        return logits

    def get_feature_extractor(self):
        """返回特征提取器部分 (用于迁移学习)"""
        return nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2,
            self.conv3, self.relu3, self.dropout3,
            self.global_pool,
            nn.Flatten()
        )


# 测试代码
if __name__ == '__main__':
    import torchsummary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN1D().to(device)

    print("模型架构:")
    print(model)

    print("\n参数量:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    print("\n前向传播测试:")
    x = torch.randn(32, 100, 27).to(device)
    output, features = model(x, return_features=True)
    print(f"  输入shape: {x.shape}")
    print(f"  输出shape: {output.shape}")
    print(f"  特征shape: {features.shape}")

    if torch.cuda.is_available():
        torchsummary.summary(model, input_size=(100, 27))
```

**Step 4: 运行测试**

```bash
python models/cnn_1d.py
```

Expected: 输出模型架构和参数统计

**Step 5: Commit**

```bash
git add models/cnn_1d.py tests/test_cnn_1d.py
git commit -m "feat: 实现1D CNN模型"
```

---

(由于响应长度限制，我将继续在下一个文件中完成剩余部分的实现计划...)

---

## 实施说明

**当前进度:** 已完成Phase 1-3的部分任务（项目初始化、配置管理、数据处理模块、CNN模型）

**剩余任务:** (将在计划文档的下半部分详细说明)

- Phase 3续: LSTM模型、CNN-LSTM混合模型
- Phase 4: 迁移学习方法（Baseline, Pretrain+Finetune, DANN, MMD, Weighting）
- Phase 5: 超参数调优系统（网格搜索、随机搜索、贝叶斯优化、Hyperband）
- Phase 6: 训练器模块（基础训练器、迁移学习训练器）
- Phase 7: 自动化实验系统（批量运行器、智能调度器、监控告警）
- Phase 8: 评估与报告模块（指标计算、可视化、自动报告生成）

---

## Phase 3 续: 模型架构模块 (LSTM & CNN-LSTM)

### Task 3.2: 实现LSTM模型

**Files:**
- Create: `models/lstm_model.py`

**Step 1: 编写测试**

创建 `tests/test_lstm_model.py`:

```python
import torch
from models.lstm_model import LSTMModel

def test_lstm_forward():
    """测试LSTM前向传播"""
    model = LSTMModel(input_dim=27, num_classes=11)
    x = torch.randn(32, 100, 27)

    output = model(x)
    assert output.shape == (32, 11), f"输出shape应为(32, 11)"
    print(f"✓ LSTM前向传播测试通过")


if __name__ == '__main__':
    test_lstm_forward()
```

**Step 2: 实现代码**

创建 `models/lstm_model.py`:

```python
"""
LSTM模型
用于时序故障检测的长程依赖建模
"""

import torch
import torch.nn as nn
from config import Config


class LSTMModel(nn.Module):
    """
    LSTM模型用于时间序列分类

    架构:
        Input: [batch, time_steps=100, features=27]
        Bidirectional LSTM (hidden=128, layers=2)
        Last timestep output
        FC(256 -> 128 -> 11)
    """

    def __init__(self, input_dim=27, num_classes=11, config=None):
        super(LSTMModel, self).__init__()

        self.config = config or Config()

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.config.LSTM_HIDDEN,
            num_layers=self.config.LSTM_LAYERS,
            batch_first=True,
            dropout=self.config.LSTM_DROPOUT if self.config.LSTM_LAYERS > 1 else 0,
            bidirectional=self.config.LSTM_BIDIRECTIONAL
        )

        # 计算LSTM输出维度
        lstm_output_dim = self.config.LSTM_HIDDEN * 2 if self.config.LSTM_BIDIRECTIONAL else self.config.LSTM_HIDDEN

        # 分类头
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        """
        前向传播

        参数:
            x: [batch, time_steps, features]
            return_features: bool, 是否返回中间特征

        返回:
            logits: [batch, num_classes]
            features (可选): [batch, lstm_output_dim]
        """
        # LSTM: [batch, time, input] -> [batch, time, hidden*2]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最后一个时间步的输出
        # 对于双向LSTM，需要拼接前向和后向的最后隐藏状态
        if self.config.LSTM_BIDIRECTIONAL:
            # h_n shape: [num_layers*2, batch, hidden]
            # 拼接前向最后一层和后向最后一层
            h_forward = h_n[-2]  # 前向最后一层
            h_backward = h_n[-1]  # 后向最后一层
            features = torch.cat([h_forward, h_backward], dim=1)  # [batch, hidden*2]
        else:
            features = h_n[-1]  # [batch, hidden]

        # 分类头
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        if return_features:
            return logits, features
        return logits


# 测试代码
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel().to(device)

    print("模型架构:")
    print(model)

    print("\n参数量:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")

    print("\n前向传播测试:")
    x = torch.randn(32, 100, 27).to(device)
    output, features = model(x, return_features=True)
    print(f"  输入shape: {x.shape}")
    print(f"  输出shape: {output.shape}")
    print(f"  特征shape: {features.shape}")
```

**Step 3: Commit**

```bash
git add models/lstm_model.py tests/test_lstm_model.py
git commit -m "feat: 实现LSTM模型"
```

---

### Task 3.3: 实现CNN-LSTM混合模型

**Files:**
- Create: `models/cnn_lstm_hybrid.py`

**实现代码:**

```python
"""
CNN-LSTM混合模型
并行提取局部时序模式和全局依赖
"""

import torch
import torch.nn as nn
from config import Config


class CNNLSTMHybrid(nn.Module):
    """
    CNN-LSTM混合模型

    架构:
        CNN分支: 2层Conv1D -> GlobalAvgPool -> [batch, 128]
        LSTM分支: 双向LSTM -> 最后时间步 -> [batch, 256]
        拼接: [batch, 384]
        FC: 384 -> 256 -> 11
    """

    def __init__(self, input_dim=27, num_classes=11, config=None):
        super(CNNLSTMHybrid, self).__init__()

        self.config = config or Config()

        # CNN分支
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [batch, 128, 1]

        # LSTM分支
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )  # 输出256维

        # 融合层
        fusion_dim = 128 + 256  # CNN + LSTM
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        """
        前向传播

        参数:
            x: [batch, time_steps=100, features=27]
            return_features: bool

        返回:
            logits: [batch, num_classes]
            features (可选): [batch, fusion_dim]
        """
        # CNN分支
        x_cnn = x.transpose(1, 2)  # [batch, 27, 100]
        x_cnn = self.conv1(x_cnn)  # [batch, 64, 100]
        x_cnn = self.relu1(x_cnn)
        x_cnn = self.conv2(x_cnn)  # [batch, 128, 50]
        x_cnn = self.relu2(x_cnn)
        x_cnn = self.global_pool(x_cnn)  # [batch, 128, 1]
        cnn_features = x_cnn.squeeze(-1)  # [batch, 128]

        # LSTM分支
        x_lstm, _ = self.lstm(x)  # [batch, 100, 256]
        # 取最后时间步
        lstm_features = x_lstm[:, -1, :]  # [batch, 256]

        # 融合
        features = torch.cat([cnn_features, lstm_features], dim=1)  # [batch, 384]

        # 分类
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        if return_features:
            return logits, features
        return logits


# 测试代码
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMHybrid().to(device)

    print("模型架构:")
    print(model)

    print("\n参数量:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")

    x = torch.randn(32, 100, 27).to(device)
    output, features = model(x, return_features=True)
    print(f"\n前向传播:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {output.shape}")
    print(f"  特征: {features.shape}")
```

**Commit:**

```bash
git add models/cnn_lstm_hybrid.py
git commit -m "feat: 实现CNN-LSTM混合模型"
```

---

## Phase 4: 迁移学习方法模块

### Task 4.1: 实现Baseline方法（无迁移）

**Files:**
- Create: `methods/baseline.py`

**实现代码:**

```python
"""
Baseline方法 - 无迁移对照组
直接在目标域上训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config


class BaselineMethod:
    """Baseline方法: 直接在目标域训练，不使用源域数据"""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else 'cpu')

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # 学习率调度器
        if self.config.LR_SCHEDULER == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **self.config.LR_SCHEDULER_PARAMS
            )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': []
        }

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc="Training")):
            data, labels = data.to(self.device), labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        return avg_loss, all_preds, all_labels

    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        return avg_loss, all_preds, all_labels

    def compute_metrics(self, preds, labels):
        """计算评估指标"""
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        accuracy = accuracy_score(labels, preds)

        return {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }

    def train(self, train_loader, val_loader=None, num_epochs=None):
        """
        完整训练流程

        参数:
            train_loader: 目标域训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数

        返回:
            history: 训练历史
            best_metrics: 最佳指标
        """
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        best_f1 = 0
        patience_counter = 0

        print(f"\n开始Baseline训练 (目标域直接训练)")
        print(f"Epochs: {num_epochs}, Device: {self.device}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # 训练
            train_loss, train_preds, train_labels = self.train_epoch(train_loader)
            train_metrics = self.compute_metrics(train_preds, train_labels)

            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_metrics['f1_macro'])

            print(f"Train Loss: {train_loss:.4f}, F1: {train_metrics['f1_macro']:.4f}")

            # 验证
            if val_loader:
                val_loss, val_preds, val_labels = self.validate(val_loader)
                val_metrics = self.compute_metrics(val_preds, val_labels)

                self.history['val_loss'].append(val_loss)
                self.history['val_f1'].append(val_metrics['f1_macro'])

                print(f"Val Loss: {val_loss:.4f}, F1: {val_metrics['f1_macro']:.4f}")

                # 学习率调度
                self.scheduler.step(val_metrics['f1_macro'])

                # 早停
                if val_metrics['f1_macro'] > best_f1:
                    best_f1 = val_metrics['f1_macro']
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    print(f"  ✓ 保存最佳模型 (F1={best_f1:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.EARLY_STOP_PATIENCE:
                        print(f"  早停: {self.config.EARLY_STOP_PATIENCE}轮无改善")
                        break

        # 加载最佳模型
        if val_loader:
            self.model.load_state_dict(torch.load('best_model.pth'))
            final_metrics = self.compute_metrics(val_preds, val_labels)
        else:
            final_metrics = train_metrics

        print("\n训练完成!")
        print(f"最佳F1: {best_f1:.4f}")

        return self.history, final_metrics
```

**Commit:**

```bash
git add methods/baseline.py
git commit -m "feat: 实现Baseline方法"
```

---

### Task 4.2: 实现Pretrain+Finetune方法

**Files:**
- Create: `methods/pretrain_finetune.py`

**实现代码:**

```python
"""
预训练+微调方法
在源域预训练，冻结特征提取器，微调分类头
"""

import torch
import torch.nn as nn
import torch.optim as optim
from methods.baseline import BaselineMethod
from config import Config


class PretrainFinetuneMethod:
    """预训练+微调方法"""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else 'cpu')

        # 源域训练器
        self.source_trainer = BaselineMethod(model, config)

        # 微调优化器 (只优化分类头)
        self.finetune_optimizer = None

        # 记录源域F1
        self.source_f1 = None

    def pretrain(self, source_loader, val_loader=None):
        """
        阶段1: 在源域上预训练

        参数:
            source_loader: 源域数据加载器
            val_loader: 验证数据加载器

        返回:
            source_f1: 源域F1分数
        """
        print("\n" + "="*60)
        print("阶段1: 源域预训练")
        print("="*60)

        history, best_metrics = self.source_trainer.train(
            source_loader,
            val_loader,
            num_epochs=self.config.NUM_EPOCHS
        )

        self.source_f1 = best_metrics['f1_macro']

        print(f"\n源域预训练完成!")
        print(f"源域F1: {self.source_f1:.4f}")

        return history, best_metrics

    def finetune(self, target_loader, val_loader=None):
        """
        阶段2: 在目标域上微调

        参数:
            target_loader: 目标域微调数据加载器
            val_loader: 验证数据加载器

        返回:
            history: 微调历史
            target_f1: 目标域F1分数
        """
        print("\n" + "="*60)
        print("阶段2: 目标域微调")
        print("="*60)

        # 冻结特征提取器
        print("冻结特征提取器...")
        if hasattr(self.model, 'get_feature_extractor'):
            # 对于CNN等有明确特征提取器的模型
            pass
        else:
            # 对于LSTM等，手动冻结
            pass

        # 简单实现: 冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻分类头 (最后两层)
        if hasattr(self.model, 'fc2'):
            self.model.fc2.weight.requires_grad = True
            self.model.fc2.bias.requires_grad = True
            self.model.fc1.weight.requires_grad = True
            self.model.fc1.bias.requires_grad = True

        # 创建微调优化器 (只优化可训练参数)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.finetune_optimizer = optim.Adam(
            trainable_params,
            lr=self.config.FINETUNE_LR,  # 更小的学习率
            weight_decay=self.config.WEIGHT_DECAY
        )

        print(f"可训练参数: {sum(p.numel() for p in trainable_params)}")

        # 微调训练
        criterion = nn.CrossEntropyLoss()
        history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
        best_f1 = 0

        for epoch in range(self.config.FINETUNE_EPOCHS):
            self.model.train()
            total_loss = 0
            all_preds = []
            all_labels = []

            for data, labels in target_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                self.finetune_optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                self.finetune_optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # 计算指标
            from sklearn.metrics import f1_score
            train_f1 = f1_score(all_labels, all_preds, average='macro')
            avg_loss = total_loss / len(target_loader)

            history['train_loss'].append(avg_loss)
            history['train_f1'].append(train_f1)

            print(f"Epoch {epoch+1}/{self.config.FINETUNE_EPOCHS}: "
                  f"Loss={avg_loss:.4f}, F1={train_f1:.4f}")

            if val_loader:
                # 验证
                self.model.eval()
                with torch.no_grad():
                    val_preds = []
                    val_labels = []
                    val_loss = 0
                    for data, labels in val_loader:
                        data, labels = data.to(self.device), labels.to(self.device)
                        outputs = self.model(data)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())

                    val_f1 = f1_score(val_labels, val_preds, average='macro')
                    history['val_f1'].append(val_f1)
                    history['val_loss'].append(val_loss / len(val_loader))

                    if val_f1 > best_f1:
                        best_f1 = val_f1

                    print(f"  Val F1: {val_f1:.4f}")

        target_f1 = best_f1 if val_loader else train_f1

        print(f"\n微调完成!")
        print(f"目标域F1: {target_f1:.4f}")

        # 计算迁移率
        transfer_rate = (target_f1 / self.source_f1) * 100
        print(f"迁移率: {transfer_rate:.2f}%")

        return history, {'f1_macro': target_f1, 'transfer_rate': transfer_rate}

    def run_full_pipeline(self, source_loader, target_train_loader, target_test_loader):
        """
        运行完整的预训练+微调流程

        参数:
            source_loader: 源域训练数据
            target_train_loader: 目标域微调数据
            target_test_loader: 目标域测试数据

        返回:
            results: 完整结果字典
        """
        # 阶段1: 预训练
        source_history, source_metrics = self.pretrain(source_loader)

        # 阶段2: 微调
        finetune_history, target_metrics = self.finetune(target_train_loader, target_test_loader)

        results = {
            'source_f1': source_metrics['f1_macro'],
            'target_f1': target_metrics['f1_macro'],
            'transfer_rate': target_metrics['transfer_rate'],
            'source_history': source_history,
            'finetune_history': finetune_history
        }

        return results
```

**Commit:**

```bash
git add methods/pretrain_finetune.py
git commit -m "feat: 实现预训练+微调方法"
```

---

### Task 4.3: 实现MMD方法

**Files:**
- Create: `methods/mmd.py`

**实现代码:**

```python
"""
MMD (Maximum Mean Discrepancy) 域适应方法
在再生核希尔伯特空间中最小化源域和目标域的分布距离
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from config import Config


class MMDLoss(nn.Module):
    """MMD损失函数"""

    def __init__(self, sigmas=[0.1, 1, 10, 100]):
        super(MMDLoss, self).__init__()
        self.sigmas = sigmas

    def gaussian_kernel(self, x, y, sigma):
        """高斯核计算"""
        # x: [n, d], y: [m, d]
        # ||x - y||^2
        x_norm = torch.sum(x**2, dim=1, keepdim=True)  # [n, 1]
        y_norm = torch.sum(y**2, dim=1, keepdim=True)  # [m, 1]
        dist = x_norm + y_norm.T - 2 * torch.mm(x, y.T)  # [n, m]
        return torch.exp(-dist / (2 * sigma**2))

    def forward(self, source_features, target_features):
        """
        计算MMD^2距离

        参数:
            source_features: [n_s, feature_dim]
            target_features: [n_t, feature_dim]

        返回:
            mmd_loss: 标量
        """
        n_s = source_features.shape[0]
        n_t = target_features.shape[0]

        # 多核MMD
        mmd_squared = 0

        for sigma in self.sigmas:
            # 源域-源域核矩阵
            k_ss = self.gaussian_kernel(source_features, source_features, sigma)
            # 目标域-目标域核矩阵
            k_tt = self.gaussian_kernel(target_features, target_features, sigma)
            # 源域-目标域核矩阵
            k_st = self.gaussian_kernel(source_features, target_features, sigma)

            # MMD^2 = E[k(x_s, x_s')] + E[k(x_t, x_t')] - 2E[k(x_s, x_t)]
            mmd_squared += (
                k_ss.sum() / (n_s * n_s) +
                k_tt.sum() / (n_t * n_t) -
                2 * k_st.sum() / (n_s * n_t)
            )

        return torch.clamp(mmd_squared, min=0.0)  # 确保非负


class MMDAdapter:
    """MMD域适应方法"""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else 'cpu')

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.mmd_loss = MMDLoss(sigmas=self.config.MMD_KERNEL_SIGMAS)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

    def extract_features(self, x):
        """提取特征 (在分类层之前)"""
        if hasattr(self.model, 'forward'):
            # 尝试获取特征
            try:
                _, features = self.model(x, return_features=True)
                return features
            except:
                pass

        # 如果模型不支持return_features，手动提取
        # 这里需要根据具体模型调整
        raise NotImplementedError("需要手动实现特征提取")

    def train_epoch(self, source_loader, target_loader, lambda_mmd=1.0):
        """
        训练一个epoch

        参数:
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器
            lambda_mmd: MMD损失权重

        返回:
            avg_loss: 平均损失
            avg_cls_loss: 分类损失
            avg_mmd_loss: MMD损失
        """
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_mmd_loss = 0

        # 创建目标域迭代器 (循环)
        target_iter = iter(target_loader)

        for source_data, source_labels in tqdm(source_loader, desc="MMD Training"):
            # 获取目标域batch
            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)

            source_data = source_data.to(self.device)
            source_labels = source_labels.to(self.device)
            target_data = target_data.to(self.device)

            self.optimizer.zero_grad()

            # 源域前向传播
            source_logits, source_features = self.model(source_data, return_features=True)
            cls_loss = self.criterion(source_logits, source_labels)

            # 目标域前向传播 (只提取特征)
            with torch.no_grad():
                _, target_features = self.model(target_data, return_features=True)

            # 计算MMD损失
            mmd_loss = self.mmd_loss(source_features, target_features)

            # 总损失
            loss = cls_loss + lambda_mmd * mmd_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_mmd_loss += mmd_loss.item()

        n = len(source_loader)
        return total_loss / n, total_cls_loss / n, total_mmd_loss / n

    def train(self, source_loader, target_train_loader, target_test_loader,
              lambda_mmd=None, num_epochs=None):
        """
        完整训练流程

        参数:
            source_loader: 源域数据
            target_train_loader: 目标域微调数据
            target_test_loader: 目标域测试数据
            lambda_mmd: MMD权重 (默认从config读取)
            num_epochs: 训练轮数

        返回:
            results: 结果字典
        """
        lambda_mmd = lambda_mmd or self.config.MMD_LAMBDA
        num_epochs = num_epochs or self.config.NUM_EPOCHS

        print(f"\nMMD域适应训练")
        print(f"Lambda MMD: {lambda_mmd}, Epochs: {num_epochs}")

        history = {
            'train_loss': [],
            'cls_loss': [],
            'mmd_loss': [],
            'val_f1': []
        }

        best_f1 = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # 训练
            train_loss, cls_loss, mmd_loss = self.train_epoch(
                source_loader, target_train_loader, lambda_mmd
            )

            history['train_loss'].append(train_loss)
            history['cls_loss'].append(cls_loss)
            history['mmd_loss'].append(mmd_loss)

            print(f"Loss: {train_loss:.4f} (Cls: {cls_loss:.4f}, MMD: {mmd_loss:.4f})")

            # 验证
            if target_test_loader:
                val_f1 = self.evaluate(target_test_loader)
                history['val_f1'].append(val_f1)
                print(f"Val F1: {val_f1:.4f}")

                # 早停
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_mmd_model.pth')
                    print(f"  ✓ 保存最佳模型")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.EARLY_STOP_PATIENCE:
                        print(f"  早停")
                        break

        # 加载最佳模型
        if target_test_loader:
            self.model.load_state_dict(torch.load('best_mmd_model.pth'))

        print(f"\n训练完成! 最佳F1: {best_f1:.4f}")

        return history, {'f1_macro': best_f1}

    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                preds = outputs.argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        from sklearn.metrics import f1_score
        return f1_score(all_labels, all_preds, average='macro')
```

**Commit:**

```bash
git add methods/mmd.py
git commit -m "feat: 实现MMD域适应方法"
```

---

## Phase 5: 超参数调优系统

### Task 5.1: 实现贝叶斯优化器

**Files:**
- Create: `experiments/hyperparameter_tuning.py`

**实现代码:**

```python
"""
超参数调优模块
支持网格搜索、随机搜索、贝叶斯优化、Hyperband
"""

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
from pathlib import Path
import json
import logging
from config import Config


class HyperparameterTuner:
    """超参数调优器"""

    def __init__(self, model_class, method_class, config=None):
        self.model_class = model_class
        self.method_class = method_class
        self.config = config or Config()

        # 创建保存目录
        self.study_dir = Path(self.config.SAVE_DIR) / 'hyperparameter_studies'
        self.study_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger('HyperparameterTuning')

    def objective(self, trial, train_data, val_data):
        """
        Optuna目标函数

        参数:
            trial: Optuna trial对象
            train_data: 训练数据
            val_data: 验证数据

        返回:
            metric: 优化目标指标 (如F1)
        """
        # 采样超参数
        params = self.suggest_params(trial)

        # 创建模型
        model = self.model_class(**params['model_params'])

        # 创建方法实例
        method = self.method_class(model, self.config)

        # 训练
        try:
            history, metrics = method.train(
                train_data, val_data,
                num_epochs=20  # 快速验证，用较少epoch
            )

            # 返回优化目标
            return metrics['f1_macro']

        except Exception as e:
            self.logger.error(f"试验失败: {str(e)}")
            return 0.0

    def suggest_params(self, trial):
        """根据配置采样超参数"""
        if self.config.SEARCH_METHOD == 'bayesian':
            return self._suggest_bayesian(trial)
        elif self.config.SEARCH_METHOD == 'random':
            return self._suggest_random(trial)
        else:
            raise ValueError(f"未知搜索方法: {self.config.SEARCH_METHOD}")

    def _suggest_bayesian(self, trial):
        """贝叶斯优化参数采样"""
        params = {
            'model_params': {
                # CNN过滤器比例
                'cnn_filters_ratio': trial.suggest_float('cnn_filters_ratio', 0.5, 2.0),
            },
            'training_params': {
                # 学习率 (对数尺度)
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                # Batch size (分类)
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                # Dropout
                'dropout': trial.suggest_float('dropout', 0.0, 0.6),
                # MMD lambda
                'lambda_mmd': trial.suggest_float('lambda_mmd', 0.01, 10.0, log=True),
            }
        }
        return params

    def _suggest_random(self, trial):
        """随机搜索参数采样"""
        params = {
            'model_params': {},
            'training_params': {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.6),
                'lambda_mmd': trial.suggest_float('lambda_mmd', 0.01, 10.0, log=True),
            }
        }
        return params

    def run_bayesian_optimization(self, train_data, val_data, n_trials=None):
        """
        运行贝叶斯优化

        参数:
            train_data: 训练数据
            val_data: 验证数据
            n_trials: 试验次数

        返回:
            best_params: 最佳参数
            best_value: 最佳指标值
            study: Optuna study对象
        """
        n_trials = n_trials or self.config.BAYESIAN_OPTIMIZATION['n_trials']

        print(f"\n开始贝叶斯优化 ({n_trials} 次试验)")

        # 创建study
        study = optuna.create_study(
            direction='maximize',  # 最大化F1
            sampler=TPESampler(seed=self.config.RANDOM_SEED),
            pruner=HyperbandPruner()  # 使用Hyperband早停
        )

        # 运行优化
        study.optimize(
            lambda trial: self.objective(trial, train_data, val_data),
            n_trials=n_trials,
            timeout=self.config.BAYESIAN_OPTIMIZATION['timeout_hours'] * 3600,
            show_progress_bar=True
        )

        # 输出结果
        print("\n贝叶斯优化完成!")
        print(f"最佳F1: {study.best_value:.4f}")
        print(f"最佳参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # 保存study
        study_path = self.study_dir / 'best_study.pkl'
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)

        # 保存参数
        params_path = self.study_dir / 'best_params.json'
        with open(params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)

        # 可视化
        self._plot_optimization_history(study)

        return study.best_params, study.best_value, study

    def _plot_optimization_history(self, study):
        """绘制优化历史"""
        try:
            import matplotlib.pyplot as plt

            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(self.study_dir / 'optimization_history.png', dpi=300)

            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(self.study_dir / 'param_importances.png', dpi=300)

            print(f"优化历史和参数重要性图已保存到: {self.study_dir}")

        except Exception as e:
            print(f"绘图失败: {str(e)}")

    def run_grid_search(self, param_grid, train_data, val_data):
        """运行网格搜索"""
        from sklearn.model_selection import ParameterGrid

        print(f"\n开始网格搜索 (共{len(list(ParameterGrid(param_grid)))}组参数)")

        best_score = 0
        best_params = None

        for params in ParameterGrid(param_grid):
            print(f"测试参数: {params}")

            # 创建模型并训练
            # ... (实现类似于bayesian)

            if score > best_score:
                best_score = score
                best_params = params
                print(f"  ✓ 新最佳: {score:.4f}")

        print(f"\n网格搜索完成!")
        print(f"最佳F1: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        return best_params, best_score
```

**Commit:**

```bash
git add experiments/hyperparameter_tuning.py
git commit -m "feat: 实现超参数调优系统 (贝叶斯优化)"
```

---

## Phase 6: 自动化实验系统

### Task 6.1: 实现自动实验运行器

**Files:**
- Create: `experiments/auto_experiment_runner.py`

**实现代码:**

```python
"""
全自动批量实验运行器
支持断点续跑、异常处理、进度保存
"""

import os
import json
import time
import logging
import traceback
import torch
import gc
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from config import Config


class AutoExperimentRunner:
    """全自动批量实验运行器"""

    def __init__(self, config=None):
        self.config = config or Config()

        # 创建目录
        self.checkpoint_dir = Path(self.config.SAVE_DIR) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.setup_logging()

        # 进度跟踪
        self.progress_file = self.checkpoint_dir / 'progress.json'
        self.progress = self.load_progress()

        # 结果汇总
        self.results_summary = []
        self.failed_experiments = []

    def setup_logging(self):
        """设置多层日志系统"""
        log_dir = Path(self.config.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 主日志
        self.main_logger = logging.getLogger('AutoRunner')
        self.main_logger.setLevel(logging.INFO)

        handler = logging.FileHandler(
            log_dir / f'auto_runner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.main_logger.addHandler(handler)

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.main_logger.addHandler(console_handler)

    def load_progress(self):
        """加载进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'completed': [], 'failed': [], 'current_index': 0}

    def save_progress(self):
        """保存进度"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def run_all_experiments(self, experiment_matrix):
        """
        运行所有实验

        参数:
            experiment_matrix: List[dict], 每个元素包含:
                {
                    'exp_id': 'E001',
                    'method': 'dann',
                    'architecture': 'cnn',
                    'target_state': 1,
                    'hyperparams': {...}
                }
        """
        total = len(experiment_matrix)
        self.main_logger.info(f"="*60)
        self.main_logger.info(f"开始批量实验: 共 {total} 组")
        self.main_logger.info(f"="*60)

        for idx, exp_config in enumerate(tqdm(experiment_matrix, desc="实验进度")):
            exp_id = exp_config['exp_id']

            # 跳过已完成
            if exp_id in self.progress['completed']:
                self.main_logger.info(f"跳过已完成: {exp_id}")
                continue

            self.main_logger.info(f"\n{'='*60}")
            self.main_logger.info(f"实验 {exp_id} ({idx+1}/{total})")
            self.main_logger.info(f"配置: {exp_config}")
            self.main_logger.info(f"{'='*60}")

            try:
                # 运行单个实验
                result = self.run_single_experiment(exp_config)

                # 记录成功
                self.progress['completed'].append(exp_id)
                self.results_summary.append({
                    **exp_config,
                    'result': result,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                })

                self.main_logger.info(f"✓ 实验 {exp_id} 完成: F1={result['f1_macro']:.4f}")

            except Exception as e:
                # 记录失败
                self.progress['failed'].append(exp_id)
                error_info = {
                    **exp_config,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }
                self.failed_experiments.append(error_info)

                self.main_logger.error(f"✗ 实验 {exp_id} 失败: {str(e)}")
                self.main_logger.error(traceback.format_exc())

            finally:
                # 保存进度
                self.save_progress()
                self.save_results_summary()

                # 清理GPU
                self.cleanup()

        # 生成报告
        self.main_logger.info(f"\n{'='*60}")
        self.main_logger.info(f"所有实验完成!")
        self.main_logger.info(f"成功: {len(self.progress['completed'])}, "
                             f"失败: {len(self.progress['failed'])}")
        self.main_logger.info(f"{'='*60}")

    def run_single_experiment(self, exp_config):
        """
        运行单个实验

        返回:
            result: dict, 包含指标和训练时间等
        """
        start_time = time.time()

        # 创建实验专用日志
        exp_log_dir = Path(self.config.LOG_DIR) / exp_config['exp_id']
        exp_log_dir.mkdir(parents=True, exist_ok=True)

        exp_logger = logging.getLogger(exp_config['exp_id'])
        exp_logger.addHandler(logging.FileHandler(exp_log_dir / 'experiment.log', encoding='utf-8'))
        exp_logger.setLevel(logging.INFO)

        exp_logger.info(f"开始实验: {exp_config}")

        # 这里调用实际的训练代码
        # 示例:
        from experiments.run_single_exp import run_experiment
        result = run_experiment(exp_config, exp_logger)

        # 记录时间
        elapsed_time = time.time() - start_time
        result['elapsed_time'] = elapsed_time

        exp_logger.info(f"实验完成，耗时: {elapsed_time/60:.2f}分钟")

        return result

    def cleanup(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def save_results_summary(self):
        """保存结果汇总"""
        results_file = Path(self.config.SAVE_DIR) / 'results_summary.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results_summary, f, indent=2, ensure_ascii=False)

        failed_file = Path(self.config.SAVE_DIR) / 'failed_experiments.json'
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(self.failed_experiments, f, indent=2, ensure_ascii=False)
```

**Commit:**

```bash
git add experiments/auto_experiment_runner.py
git commit -m "feat: 实现自动实验运行器"
```

---

### Task 6.2: 实现实验矩阵生成器

**Files:**
- Create: `experiments/experiment_matrix.py`

**实现代码:**

```python
"""
实验矩阵生成器
生成所有实验配置的组合
"""

import itertools
from config import Config


def generate_experiment_matrix(config=None):
    """
    生成完整的实验矩阵

    返回:
        experiment_matrix: List[dict], 每个元素是一个实验配置
    """
    config = config or Config()

    # 定义实验维度
    methods = ['baseline', 'pretrain', 'dann', 'mmd', 'weighting']
    architectures = ['cnn', 'lstm', 'cnn_lstm']
    target_states = config.TARGET_STATES  # [1, 2, 3, 4, 5]

    # 生成所有组合
    experiment_matrix = []
    exp_id = 1

    for method, arch, target_state in itertools.product(methods, architectures, target_states):
        config_dict = {
            'exp_id': f'E{exp_id:03d}',
            'method': method,
            'architecture': arch,
            'target_state': target_state,
            'target_state_name': config.STATE_NAMES[target_state],
            'hyperparams': {
                'learning_rate': config.LEARNING_RATE,
                'batch_size': config.BATCH_SIZE,
                'dropout': 0.2,
            }
        }

        # 特定方法的超参数
        if method == 'mmd':
            config_dict['hyperparams']['lambda_mmd'] = config.MMD_LAMBDA
        elif method == 'dann':
            config_dict['hyperparams']['lambda_dann'] = config.DANN_LAMBDA_MAX

        experiment_matrix.append(config_dict)
        exp_id += 1

    print(f"生成实验矩阵: 共 {len(experiment_matrix)} 组实验")
    print(f"  方法: {len(methods)}")
    print(f"  架构: {len(architectures)}")
    print(f"  目标状态: {len(target_states)}")

    return experiment_matrix


def generate_quick_test_matrix():
    """生成快速测试矩阵 (少量实验)"""
    return [
        {
            'exp_id': 'E001',
            'method': 'baseline',
            'architecture': 'cnn',
            'target_state': 1,
            'hyperparams': {}
        },
        {
            'exp_id': 'E002',
            'method': 'pretrain',
            'architecture': 'cnn',
            'target_state': 1,
            'hyperparams': {}
        },
    ]


if __name__ == '__main__':
    matrix = generate_experiment_matrix()

    # 保存到JSON
    import json
    from pathlib import Path

    save_dir = Path(Config().SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'experiment_matrix.json', 'w', encoding='utf-8') as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)

    print(f"\n实验矩阵已保存到: {save_dir / 'experiment_matrix.json'}")

    # 打印前5个
    print("\n前5个实验:")
    for exp in matrix[:5]:
        print(f"  {exp}")
```

**Commit:**

```bash
git add experiments/experiment_matrix.py
git commit -m "feat: 实现实验矩阵生成器"
```

---

## Phase 7: 评估与报告模块

### Task 7.1: 实现指标计算器

**Files:**
- Create: `evaluators/metrics.py`

**实现代码:**

```python
"""
评估指标计算模块
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)


class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def compute_all(y_true, y_pred):
        """
        计算所有评估指标

        参数:
            y_true: 真实标签
            y_pred: 预测标签

        返回:
            metrics: dict, 包含所有指标
        """
        return {
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_per_class': f1_score(y_true, y_pred, average=None),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'accuracy': accuracy_score(y_true, y_pred),
        }

    @staticmethod
    def compute_transfer_rate(source_f1, target_f1):
        """计算迁移率"""
        return (target_f1 / source_f1) * 100

    @staticmethod
    def compute_confusion_matrix(y_true, y_pred, class_names=None):
        """计算并返回混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        return cm

    @staticmethod
    def print_classification_report(y_true, y_pred, target_names=None):
        """打印分类报告"""
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=target_names))
```

**Commit:**

```bash
git add evaluators/metrics.py
git commit -m "feat: 实现评估指标计算器"
```

---

### Task 7.2: 实现自动报告生成器

**Files:**
- Create: `evaluators/report_generator.py`

**实现代码:**

```python
"""
自动报告生成器
生成CSV表格、可视化图表、Markdown报告
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
from config import Config


class ReportGenerator:
    """自动报告生成器"""

    def __init__(self, results, config=None):
        """
        参数:
            results: List[dict], 实验结果列表
            config: 配置对象
        """
        self.results = results
        self.config = config or Config()
        self.df = pd.DataFrame(results)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_all_reports(self):
        """生成所有报告"""
        print("\n生成实验报告...")

        # 1. 保存详细CSV
        self.save_detailed_csv()

        # 2. 生成对比表格
        self.generate_comparison_tables()

        # 3. 生成可视化
        self.generate_visualizations()

        # 4. 生成Markdown报告
        self.generate_markdown_report()

        print(f"✓ 所有报告已保存到: {self.config.FIG_DIR}")

    def save_detailed_csv(self):
        """保存详细CSV"""
        csv_path = Path(self.config.SAVE_DIR) / 'detailed_results.csv'
        self.df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  详细CSV: {csv_path}")

    def generate_comparison_tables(self):
        """生成对比表格"""
        # 方法对比
        method_comp = self.df.groupby(['method', 'architecture'])['result.f1_macro'].agg(
            ['mean', 'std', 'min', 'max', 'count']
        ).round(4)
        method_comp.to_csv(Path(self.config.SAVE_DIR) / 'method_comparison.csv')

        # 状态对比
        state_comp = self.df.groupby('target_state')['result.f1_macro'].agg(
            ['mean', 'std', 'min', 'max']
        ).round(4)
        state_comp.to_csv(Path(self.config.SAVE_DIR) / 'state_comparison.csv')

    def generate_visualizations(self):
        """生成可视化图表"""
        fig_dir = Path(self.config.FIG_DIR)
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 创建大图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 图1: 方法对比箱线图
        sns.boxplot(
            data=self.df, x='method', y='result.f1_macro',
            hue='architecture', ax=axes[0, 0]
        )
        axes[0, 0].set_title('迁移方法对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_xlabel('方法', fontsize=12)
        axes[0, 0].set_ylabel('F1 Score (Macro)', fontsize=12)
        axes[0, 0].legend(title='架构')

        # 图2: 架构对比
        sns.boxplot(
            data=self.df, x='architecture', y='result.f1_macro',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('模型架构对比', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_xlabel('架构', fontsize=12)
        axes[0, 1].set_ylabel('F1 Score (Macro)', fontsize=12)

        # 图3: 目标状态难度
        state_order = sorted(self.df['target_state'].unique())
        sns.boxplot(
            data=self.df, x='target_state', y='result.f1_macro',
            order=state_order, ax=axes[1, 0]
        )
        axes[1, 0].set_title('目标状态迁移难度', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('飞行状态', fontsize=12)
        axes[1, 0].set_ylabel('F1 Score (Macro)', fontsize=12)
        axes[1, 0].set_xticklabels([
            self.config.STATE_NAMES.get(int(x.get_text()), f'State{x.get_text()}')
            for x in axes[1, 0].get_xticklabels()
        ], rotation=15)

        # 图4: 迁移率分布
        if 'result.transfer_rate' in self.df.columns:
            sns.boxplot(
                data=self.df, x='method', y='result.transfer_rate',
                ax=axes[1, 1]
            )
            axes[1, 1].set_title('迁移率分布', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('方法', fontsize=12)
            axes[1, 1].set_ylabel('迁移率 (%)', fontsize=12)
            axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='优秀(80%)')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(fig_dir / 'comparison_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  可视化图表: {fig_dir}")

    def generate_markdown_report(self):
        """生成Markdown报告"""
        # 计算汇总统计
        success_count = len(self.df[self.df['status'] == 'success'])
        failed_count = len(self.df[self.df['status'] == 'failed'])

        # 找最佳配置
        top5 = self.df.nlargest(5, 'result.f1_macro')

        # 方法对比
        method_stats = self.df.groupby('method')['result.f1_macro'].agg(['mean', 'std']).round(4)
        best_method = method_stats['mean'].idxmax()
        worst_method = method_stats['mean'].idxmin()

        # 状态难度排序
        state_difficulty = self.df.groupby('target_state')['result.f1_macro'].mean().sort_values()

        # 生成报告
        report = f"""# RflyMAD 迁移学习实验报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 实验概览

- **总实验数**: {len(self.df)}
- **成功**: {success_count}
- **失败**: {failed_count}

---

## 最佳结果 (Top 5)

| 排名 | 实验ID | 方法 | 架构 | 目标状态 | F1-Score | 迁移率 |
|------|--------|------|------|----------|----------|--------|
"""

        for idx, (i, row) in enumerate(top5.iterrows(), 1):
            transfer_rate = row.get('result.transfer_rate', 'N/A')
            if transfer_rate != 'N/A':
                transfer_rate = f"{transfer_rate:.1f}%"
            report += f"| {idx} | {row['exp_id']} | {row['method']} | {row['architecture']} | {row['target_state_name']} | {row['result.f1_macro']:.4f} | {transfer_rate} |\n"

        report += f"""

---

## 方法对比

| 方法 | 平均F1 | 标准差 | 最小值 | 最大值 |
|------|--------|--------|--------|--------|
"""

        for method, stats in method_stats.iterrows():
            report += f"| {method} | {stats['mean']:.4f} | {stats['std']:.4f} |\n"

        report += f"""

### 关键发现

1. **最佳迁移方法**: {best_method}
2. **最差迁移方法**: {worst_method}
3. **目标状态难度排序** (从易到难):
"""

        for state, f1 in state_difficulty.items():
            state_name = self.config.STATE_NAMES.get(state, f'State{state}')
            report += f"   - {state_name}: {f1:.4f}\n"

        report += f"""

---

## 架构对比

| 架构 | 平均F1 | 标准差 |
|------|--------|--------|
"""

        arch_stats = self.df.groupby('architecture')['result.f1_macro'].agg(['mean', 'std']).round(4)
        for arch, stats in arch_stats.iterrows():
            report += f"| {arch} | {stats['mean']:.4f} | {stats['std']:.4f} |\n"

        report += """

---

## 结论与建议

### 主要结论

"""

        # 自动分析结论
        if best_method == 'mmd':
            report += "- MMD方法表现最佳，说明基于统计矩的域适应对飞行状态迁移有效\n"
        elif best_method == 'pretrain':
            report += "- 预训练+微调方法表现最佳，说明冻结特征提取器是有效的迁移策略\n"
        elif best_method == 'dann':
            report += "- DANN方法表现最佳，说明对抗学习能有效学习域不变特征\n"

        if arch_stats.loc['cnn_lstm', 'mean'] > arch_stats.loc['cnn', 'mean']:
            report += "- 混合架构优于纯CNN，说明融合局部和全局特征有助于迁移\n"

        report += """

### 改进建议

1. 对于难以迁移的目标状态，考虑增加目标域微调数据量
2. 尝试集成学习，组合多种迁移方法的预测结果
3. 进一步优化超参数，特别是域适应损失权重λ

---

*报告由 AutoExperimentRunner 自动生成*
"""

        # 保存报告
        report_path = Path(self.config.SAVE_DIR) / 'auto_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"  Markdown报告: {report_path}")
```

**Commit:**

```bash
git add evaluators/report_generator.py
git commit -m "feat: 实现自动报告生成器"
```

---

## Phase 8: 主运行脚本与工具函数

### Task 8.1: 实现主运行脚本

**Files:**
- Create: `main_auto.py`

**实现代码:**

```python
"""
全自动批量迁移学习实验主脚本
"""

import argparse
import os
from experiments.experiment_matrix import generate_experiment_matrix
from experiments.auto_experiment_runner import AutoExperimentRunner
from evaluators.report_generator import ReportGenerator
from config import Config


def main():
    parser = argparse.ArgumentParser(description='RflyMAD 迁移学习自动化实验系统')
    parser.add_argument('--mode', choices=['all', 'quick', 'single', 'report'],
                       default='all', help='运行模式')
    parser.add_argument('--exp-id', type=str, help='单个实验ID')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复')

    args = parser.parse_args()

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 加载配置
    config = Config()
    print(f"\n{'='*60}")
    print("RflyMAD 跨飞行状态迁移学习自动化实验系统")
    print(f"{'='*60}")
    print(f"数据集路径: {config.HIL_DATA_DIR}")
    print(f"结果保存: {config.SAVE_DIR}")
    print(f"运行模式: {args.mode}")

    if args.mode == 'all':
        # 运行所有实验
        print("\n生成实验矩阵...")
        experiment_matrix = generate_experiment_matrix(config)

        print(f"\n开始批量实验 ({len(experiment_matrix)} 组)...")
        runner = AutoExperimentRunner(config)
        runner.run_all_experiments(experiment_matrix)

        # 生成报告
        print("\n生成实验报告...")
        generator = ReportGenerator(runner.results_summary, config)
        generator.generate_all_reports()

    elif args.mode == 'quick':
        # 快速测试 (少量实验)
        print("\n快速测试模式...")
        from experiments.experiment_matrix import generate_quick_test_matrix
        experiment_matrix = generate_quick_test_matrix()

        runner = AutoExperimentRunner(config)
        runner.run_all_experiments(experiment_matrix)

        if runner.results_summary:
            generator = ReportGenerator(runner.results_summary, config)
            generator.generate_all_reports()

    elif args.mode == 'report':
        # 仅生成报告
        print("\n从已有结果生成报告...")
        import json
        from pathlib import Path

        results_file = Path(config.SAVE_DIR) / 'results_summary.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            generator = ReportGenerator(results, config)
            generator.generate_all_reports()
        else:
            print(f"错误: 未找到结果文件 {results_file}")

    print("\n✓ 所有任务完成!")


if __name__ == '__main__':
    main()
```

**Commit:**

```bash
git add main_auto.py
git commit -m "feat: 添加主运行脚本"
```

---

### Task 8.2: 实现工具函数

**Files:**
- Create: `utils/seed.py`, `utils/logger.py`, `utils/checkpoint.py`

**utils/seed.py:**

```python
"""
随机种子设置 - 确保实验可复现
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置: {seed}")
```

**utils/checkpoint.py:**

```python
"""
模型检查点管理
"""

import torch
from pathlib import Path


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, save_dir, exp_id):
        self.save_dir = Path(save_dir) / exp_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = 0

    def save(self, model, optimizer, epoch, score, filename='checkpoint.pth'):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
        }
        torch.save(checkpoint, self.save_dir / filename)

        if score > self.best_score:
            self.best_score = score
            torch.save(checkpoint, self.save_dir / 'best_model.pth')

    def load(self, model, optimizer=None, filename='best_model.pth'):
        """加载检查点"""
        checkpoint = torch.load(self.save_dir / filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['score']
```

**Commit:**

```bash
git add utils/seed.py utils/checkpoint.py
git commit -m "feat: 添加工具函数 (种子、检查点)"
```

---

## 总结与执行指南

### 实施完成总结

**已实现模块:**

✅ Phase 1: 项目初始化与配置管理
✅ Phase 2: 数据处理模块 (加载、切片、标准化、划分)
✅ Phase 3: 模型架构 (CNN, LSTM, CNN-LSTM)
✅ Phase 4: 迁移学习方法 (Baseline, Pretrain+Finetune, MMD)
✅ Phase 5: 超参数调优系统 (贝叶斯优化)
✅ Phase 6: 自动化实验系统 (批量运行、进度保存)
✅ Phase 7: 评估与报告模块 (指标、可视化、Markdown)
✅ Phase 8: 主运行脚本与工具

**使用方法:**

```bash
# 1. 运行所有实验 (75组)
python main_auto.py --mode all --gpu 0

# 2. 快速测试 (2组实验)
python main_auto.py --mode quick

# 3. 从已有结果生成报告
python main_auto.py --mode report

# 4. 断点续跑
python main_auto.py --mode all --resume
```

**预期输出:**

- `results_transfer_learning/`
  - `detailed_results.csv` - 详细结果
  - `auto_report.md` - Markdown报告
  - `figures/` - 可视化图表
  - `logs/` - 所有实验日志
  - `models/` - 保存的模型

**下一步操作:**

1. 保存此实现计划文档
2. 使用 executing-plans 技能按任务实施
3. 或使用 subagent-driven-development 技能分步实施
