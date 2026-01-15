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
        # 移除.csv扩展名和Case_前缀
        name_without_ext = basename.replace('.csv', '').replace('Case_', '')

        # 解析: X[S][FT][CASEID]
        # X: 第1位 - 数据类型 (1=sil, 2=hil, 3=real)
        # S: 第2位 - 飞行状态 (0-5)
        # FT: 第3-4位 - 故障类型 (00-10)
        # CASEID: 第5位及之后 - 案例ID

        if len(name_without_ext) < 5:
            raise ValueError(f"文件名格式错误: {filename}")

        data_type = int(name_without_ext[0])  # 第1位: 1=sil, 2=hil, 3=real
        state_id = int(name_without_ext[1])  # 第2位: 飞行状态
        fault_type = int(name_without_ext[2:4])  # 第3-4位: 故障类型
        case_id = int(name_without_ext[4:])  # 第5位及之后: 案例ID

        # 只返回state_id和fault_type，data_type用于验证是否为HIL数据
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
        # HIL数据的格式：Case_2[S][FT]*.csv (第1位为2表示HIL，第2位为state_id)
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
