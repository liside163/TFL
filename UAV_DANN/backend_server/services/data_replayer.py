# 作用: 数据回放服务,从CSV读取数据并模拟实时流
import asyncio
from typing import AsyncGenerator, List
import numpy as np
import pandas as pd

from backend_server.config import settings


# RflyMAD 数据集真实的21维特征列名
# 维度变换: CSV文件 -> ndarray (N, 21) -> 单样本 (21,)
FEATURE_COLUMNS: List[str] = [
    # 控制指令 (4维) - 索引 0-3
    "_actuator_controls_0_0_control[0]",   # roll控制
    "_actuator_controls_0_0_control[1]",   # pitch控制
    "_actuator_controls_0_0_control[2]",   # yaw控制
    "_actuator_controls_0_0_control[3]",   # thrust控制
    
    # PWM输出 (4维) - 索引 4-7
    "_actuator_outputs_0_output[0]",       # 电机1 PWM
    "_actuator_outputs_0_output[1]",       # 电机2 PWM
    "_actuator_outputs_0_output[2]",       # 电机3 PWM
    "_actuator_outputs_0_output[3]",       # 电机4 PWM
    
    # 陀螺仪 (3维) - 索引 8-10
    "_sensor_combined_0_gyro_rad[0]",      # X轴角速度 (rad/s)
    "_sensor_combined_0_gyro_rad[1]",      # Y轴角速度 (rad/s)
    "_sensor_combined_0_gyro_rad[2]",      # Z轴角速度 (rad/s)
    
    # 加速度计 (3维) - 索引 11-13
    "_sensor_combined_0_accelerometer_m_s2[0]",  # X轴加速度 (m/s²)
    "_sensor_combined_0_accelerometer_m_s2[1]",  # Y轴加速度 (m/s²)
    "_sensor_combined_0_accelerometer_m_s2[2]",  # Z轴加速度 (m/s²)
    
    # 气压计 (3维) - 索引 14-16
    "_vehicle_air_data_0_baro_alt_meter",       # 气压高度 (m)
    "_vehicle_air_data_0_baro_temp_celcius",    # 温度 (°C)
    "_vehicle_air_data_0_baro_pressure_pa",     # 气压 (Pa)
    
    # 姿态四元数 (4维) - 索引 17-20
    "_vehicle_attitude_0_q[0]",            # q0 (w)
    "_vehicle_attitude_0_q[1]",            # q1 (x)
    "_vehicle_attitude_0_q[2]",            # q2 (y)
    "_vehicle_attitude_0_q[3]",            # q3 (z)
]


class DataReplayer:
    def __init__(self, csv_path: str, speed_factor: float = 1.0) -> None:
        self.csv_path = csv_path
        self.speed_factor = speed_factor
        self.data: np.ndarray | None = None

    def load_data(self) -> None:
        df = pd.read_csv(self.csv_path)
        if all(col in df.columns for col in FEATURE_COLUMNS):
            features = df[FEATURE_COLUMNS]
        else:
            features = df.iloc[:, : settings.feature_dim]
        self.data = features.values.astype(np.float32)
        self.current_index = 0  # 添加索引追踪

    def reset(self) -> None:
        """重置播放位置到数据开头"""
        self.current_index = 0

    async def replay_stream(self) -> AsyncGenerator[np.ndarray, None]:
        if self.data is None:
            self.load_data()
        assert self.data is not None

        interval = 1.0 / settings.sample_rate
        interval = interval / max(self.speed_factor, 1e-6)

        for row in self.data:
            yield row
            await asyncio.sleep(interval)
