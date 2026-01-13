
import pickle
from pathlib import Path
from sre_parse import expand_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 本脚本用于从指定训练数据集中统计特征分布，并生成标准化所需的Scaler文件
# 需要参与标准化的特征列名，顺序用于与数据对齐，保证各通道含义一致
# 维度变换: CSV (N行 x M列) -> 选取21列 -> ndarray (N, 21)
FEATURE_COLUMNS = [
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
    "_vehicle_air_data_0_baro_alt_meter",       # 气压高度
    "_vehicle_air_data_0_baro_temp_celcius",    # 温度
    "_vehicle_air_data_0_baro_pressure_pa",     # 气压
    
    # 姿态四元数 (4维) - 姿态估计
    "_vehicle_attitude_0_q[0]",            # q0 (w)
    "_vehicle_attitude_0_q[1]",            # q1 (x)
    "_vehicle_attitude_0_q[2]",            # q2 (y)
    "_vehicle_attitude_0_q[3]",            # q3 (z)
]


def main() -> None:
    # 数据根目录：包含多种工况的CSV，脚本仅选择符合条件的训练文件
    data_root = Path("D:/Bigshe/RflyMAD_Dataset/Processdata_HIL&REAL")
    # 只采集Case_2000000000.csv 的hover样本作为标准化统计基础，保证与训练工况一致
    example_root = Path(data_root / "HIL")
    train_files = list[Path](example_root.glob("Case_2000000000.csv"))
    # 若未找到文件，直接终止并提示，避免生成空的或无效的Scaler
    if not train_files:
        raise FileNotFoundError("No training files found for scaler generation")

    # 收集所有训练样本的特征矩阵，最终做纵向拼接用于全局统计
    all_data = []
    for file in train_files:
        # 逐个读取CSV，避免一次性加载过多文件导致内存峰值
        df = pd.read_csv(file)
        # 若包含指定特征列，按列名选择，避免列顺序变动导致特征错位
        if all(col in df.columns for col in FEATURE_COLUMNS):
            all_data.append(df[FEATURE_COLUMNS].values)
        else:
            # 若缺少列名，则退化为取前21列，保持特征维度一致
            all_data.append(df.iloc[:, :21].values)

    # 将各文件的样本纵向拼接，形成统一训练矩阵用于统计均值与方差
    X_train = np.vstack(all_data)
    # 使用StandardScaler进行标准化统计，后续训练/推理可复用相同尺度
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 输出目录：集中存放不同工况的Scaler文件，便于管理与复现
    output_dir = Path("D:/Bigshe/TFL/UAV_DANN/scalers")
    # 若目录不存在则创建，确保写入路径有效
    output_dir.mkdir(parents=True, exist_ok=True)
    # 输出文件命名与工况对应，便于多工况训练时选择正确Scaler
    output_path = output_dir / "condition_0_scaler.pkl"

    # 以二进制形式序列化保存，确保sklearn对象完整可恢复
    with open(output_path, "wb") as f:
        pickle.dump(scaler, f)

    # 打印输出路径，便于脚本调用者确认生成结果
    print(f"Scaler saved: {output_path}")


if __name__ == "__main__":
    # 作为脚本执行入口，避免被导入时自动运行
    main()
