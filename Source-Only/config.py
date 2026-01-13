# =====================================================================
# HIL→REAL 单一工况迁移学习 - Source-Only实验
# 配置文件 - 集中管理所有超参数和配置
# =====================================================================

import os
from pathlib import Path

# =====================================================================
# 路径配置
# =====================================================================
# 数据集根目录 (使用环境变量，便于 Docker 容器和本地环境切换)
# Docker 容器中：设置 DATA_ROOT=/data
# Windows/WSL 中：不设置环境变量时使用默认路径
DATA_ROOT = Path(os.getenv('DATA_ROOT', '/mnt/d/DL_LEARN/Dataset/Processdata_HIL&REAL'))

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
RESULT_DIR = OUTPUT_DIR / "results"

# 创建必要的目录
for dir_path in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, RESULT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 数据集配置
# =====================================================================
# 域编码映射 (文件名中的[A]位)
DOMAIN_MAPPING = {
    "1": "SIL",   # 软件在环仿真
    "2": "HIL",   # 硬件在环仿真
    "3": "Real",  # 真实飞行
    "4": "SIL_ROS",
    "5": "HIL_ROS"
}

# 飞行状态编码映射 (文件名中的[B]位)
FLIGHT_STATE_MAPPING = {
    "0": "hover",      # 悬停
    "1": "waypoint",   # 航点
    "2": "velocity",   # 速度控制
    "3": "circling",   # 绕圈
    "4": "acce",       # 加速
    "5": "dece"        # 减速
}

# REAL数据中可用的飞行状态 (dece在REAL中没有数据)
AVAILABLE_FLIGHT_STATES = ["hover", "waypoint", "velocity", "circling", "acce"]

# 故障类型编码映射 (文件名中的[CD]位) - 完整11分类
FAULT_TYPE_MAPPING_FULL = {
    "00": "Motor",
    "01": "Propeller",
    "02": "Low Voltage",
    "03": "Wind Affect",
    "04": "Load Lose",
    "05": "Accelerometer",
    "06": "Gyroscope",
    "07": "Magnetometer",
    "08": "Barometer",
    "09": "GPS",
    "10": "No Fault"
}

# 7分类故障类型 (REAL数据中可用的故障类型)
# REAL数据缺少: Propeller, Low Voltage, Wind Affect, Load Lose
FAULT_TYPE_MAPPING_7CLASS = {
    "00": "Motor",
    "05": "Accelerometer",
    "06": "Gyroscope",
    "07": "Magnetometer",
    "08": "Barometer",
    "09": "GPS",
    "10": "No Fault"
}

# 7分类标签到索引的映射
FAULT_LABEL_TO_IDX = {
    "Motor": 0,
    "Accelerometer": 1,
    "Gyroscope": 2,
    "Magnetometer": 3,
    "Barometer": 4,
    "GPS": 5,
    "No Fault": 6
}

# 索引到标签的映射
FAULT_IDX_TO_LABEL = {v: k for k, v in FAULT_LABEL_TO_IDX.items()}

# 类别数量
NUM_CLASSES = 7

# =====================================================================
# 特征配置
# =====================================================================
# HIL与REAL共有的特征 (21个特征)
SHARED_FEATURES = [
    # 控制指令 (4维) - 姿态控制输出
    "_actuator_controls_0_0_control[0]",   # roll控制
    "_actuator_controls_0_0_control[1]",   # pitch控制
    "_actuator_controls_0_0_control[2]",   # yaw控制
    "_actuator_controls_0_0_control[3]",   # thrust控制
    
    # PWM输出 (4维) - 电机驱动信号
    "_actuator_outputs_0_output[0]",       # 电机1 PWM
    "_actuator_outputs_0_output[1]",       # 电机2 PWM
    "_actuator_outputs_0_output[2]",       # 电机3 PWM
    "_actuator_outputs_0_output[3]",       # 电机4 PWM
    
    # 陀螺仪 (3维) - 角速度
    "_sensor_combined_0_gyro_rad[0]",      # X轴角速度
    "_sensor_combined_0_gyro_rad[1]",      # Y轴角速度
    "_sensor_combined_0_gyro_rad[2]",      # Z轴角速度
    
    # 加速度计 (3维) - 线加速度
    "_sensor_combined_0_accelerometer_m_s2[0]",  # X轴加速度
    "_sensor_combined_0_accelerometer_m_s2[1]",  # Y轴加速度
    "_sensor_combined_0_accelerometer_m_s2[2]",  # Z轴加速度
    
    # 气压计 (3维)
    "_vehicle_air_data_0_baro_alt_meter",        # 气压高度
    "_vehicle_air_data_0_baro_temp_celcius",     # 温度
    "_vehicle_air_data_0_baro_pressure_pa",      # 气压
    
    # 姿态四元数 (4维) - 姿态估计
    "_vehicle_attitude_0_q[0]",            # q0 (w)
    "_vehicle_attitude_0_q[1]",            # q1 (x)
    "_vehicle_attitude_0_q[2]",            # q2 (y)
    "_vehicle_attitude_0_q[3]",            # q3 (z)
]

# 特征数量
NUM_FEATURES = len(SHARED_FEATURES)  # 21

# =====================================================================
# 序列配置
# =====================================================================
# 固定序列长度 (根据数据集统计，中位数约1400，选择1000作为折中)
SEQUENCE_LENGTH = 1000

# 序列处理方式: "truncate" 截断, "pad" 填充, "both" 先截断后填充
SEQUENCE_MODE = "both"

# 填充值
PAD_VALUE = 0.0

# =====================================================================
# 模型超参数
# =====================================================================
# CNN特征提取器配置
CNN_CONFIG = {
    "in_channels": NUM_FEATURES,      # 输入通道数 = 特征数 (30)
    "layer1_channels": 64,            # 第一层输出通道
    "layer2_channels": 128,           # 第二层输出通道
    "layer3_channels": 256,           # 第三层输出通道
    "kernel_size": 3,                 # 卷积核大小
    "pool_size": 2,                   # 池化核大小
    "dropout": 0.3                    # Dropout比率
}

# LSTM配置
LSTM_CONFIG = {
    "input_size": CNN_CONFIG["layer3_channels"],  # LSTM输入维度 = CNN输出通道 (256)
    "hidden_size": 128,               # LSTM隐藏层大小
    "num_layers": 2,                  # LSTM层数
    "bidirectional": True,            # 是否双向
    "dropout": 0.3                    # Dropout比率 (仅在num_layers > 1时生效)
}

# 分类器配置
CLASSIFIER_CONFIG = {
    "input_size": LSTM_CONFIG["hidden_size"] * (2 if LSTM_CONFIG["bidirectional"] else 1),  # 256
    "hidden_size": 128,               # 隐藏层大小
    "num_classes": NUM_CLASSES,       # 输出类别数 (7)
    "dropout": 0.5                    # Dropout比率
}

# =====================================================================
# 训练超参数
# =====================================================================
TRAIN_CONFIG = {
    "batch_size": 32,                 # 批次大小
    "epochs": 100,                    # 训练轮数
    "learning_rate": 1e-3,            # 初始学习率
    "weight_decay": 1e-4,             # L2正则化系数
    "early_stopping_patience": 15,    # 早停耐心值
    "scheduler_patience": 5,          # 学习率调度器耐心值
    "scheduler_factor": 0.5,          # 学习率衰减因子
    "min_lr": 1e-6,                   # 最小学习率
    "val_split": 0.2,                 # 验证集比例
    "random_seed": 42,                # 随机种子
    "num_workers": 4,                 # DataLoader工作进程数
    "pin_memory": True,               # 是否使用锁页内存
}

# =====================================================================
# 设备配置
# =====================================================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 日志配置
# =====================================================================
LOG_CONFIG = {
    "log_interval": 10,               # 每N个batch打印一次日志
    "save_interval": 5,               # 每N个epoch保存一次模型
    "tensorboard": True,              # 是否使用TensorBoard
}

# =====================================================================
# 打印配置信息
# =====================================================================
def print_config():
    """打印当前配置信息"""
    print("=" * 60)
    print("Source-Only 实验配置")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"数据集根目录: {DATA_ROOT}")
    print(f"类别数: {NUM_CLASSES}")
    print(f"特征数: {NUM_FEATURES}")
    print(f"序列长度: {SEQUENCE_LENGTH}")
    print(f"批次大小: {TRAIN_CONFIG['batch_size']}")
    print(f"训练轮数: {TRAIN_CONFIG['epochs']}")
    print(f"学习率: {TRAIN_CONFIG['learning_rate']}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
