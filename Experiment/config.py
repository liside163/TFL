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
    MATPLOTLIB_BACKEND = 'Agg'  # 非交互式后端,适合服务器

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
