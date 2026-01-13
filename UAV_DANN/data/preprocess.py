# -*- coding: utf-8 -*-
"""
==============================================================================
数据预处理模块
==============================================================================
功能：读取并处理RflyMAD数据集的HIL和Real数据
- 特征选择与对齐
- 滑动窗口分割
- 数据标准化
- 数据集划分

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import yaml
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    数据预处理器类
    
    负责读取原始CSV数据、进行特征选择、滑动窗口分割和标准化处理
    
    Attributes:
        config (dict): 配置字典，包含所有预处理参数
        scaler: 用于数据标准化的Scaler对象
    
    维度变化说明：
    ----------------
    原始CSV数据: (T_total, all_features) → 每个CSV文件是一次完整飞行记录
    
    预处理后: (N_samples, seq_len, n_features) = (N, 100, 21)
        - N_samples: 滑动窗口分割后的样本数
        - seq_len: 窗口大小，默认100个时间步
        - n_features: 选择的特征数，默认21维
    """
    
    def __init__(self, config_path: str = None, config: dict = None):
        """
        初始化数据预处理器
        
        Args:
            config_path: 配置文件路径（YAML格式）
            config: 配置字典（直接传入，优先级高于config_path）
        """
        # 加载配置
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("必须提供config_path或config参数")
        
        # 提取关键配置
        self.data_root = self.config['data']['data_root']
        self.window_size = self.config['preprocessing']['window_size']
        self.stride = self.config['preprocessing']['stride']
        self.normalization = self.config['preprocessing']['normalization']
        self.selected_features = self.config['preprocessing']['selected_features']
        self.n_features = len(self.selected_features)
        
        # 故障类型映射
        self.fault_labels = self.config['fault_types']['labels']
        self.num_classes = self.config['fault_types']['num_classes']
        
        # 变点检测配置 (用于Real数据故障定位)
        cpd_config = self.config.get('change_point_detection', {})
        self.cpd_enabled = cpd_config.get('enabled', True)
        self.cpd_method = cpd_config.get('method', 'combined')
        self.cpd_threshold = cpd_config.get('threshold', 2.5)
        self.cpd_window_size = cpd_config.get('window_size', 30)
        self.cpd_min_segment_length = cpd_config.get('min_segment_length', 100)
        self.cpd_detection_features_count = cpd_config.get('detection_features_count', 8)
        
        # 初始化Scaler（稍后根据源域数据拟合）
        self.scaler = None
        
        # 随机种子
        self.random_seed = self.config['data']['random_seed']
        np.random.seed(self.random_seed)
    
    def _get_fault_label(self, file_path: str) -> int:
        """
        从文件名中解析故障类型标签
        
        RflyMAD数据集文件命名规则: Case_[A][B][CD][EFGHIJ].csv
        
        [A]: 子数据集
            1=SIL, 2=HIL, 3=Real, 4=SIL with ROS, 5=HIL with ROS
        
        [B]: 飞行状态
            0=hover, 1=waypoint, 2=velocity, 3=circling, 4=acce, 5=dece
        
        [CD]: 故障类型 (我们需要提取的部分)
            00=motor, 01=propeller, 02=low voltage, 03=wind affect,
            04=load lose, 05=accelerometer, 06=gyroscope, 07=magnetometer,
            08=barometer, 09=GPS, 10=No fault
        
        [EFGHIJ]: 序列号
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            fault_label: 故障类型的数值标签（0-10）
            
        示例:
            Case_3109000005.csv → 3=Real, 1=waypoint, 09=GPS故障, 000005=序列号
            返回标签: 9 (对应GPS)
        """
        import re
        
        # 获取文件名（不含路径和扩展名）
        filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        # 7分类故障类型编码到标签的映射
        # 移除Real域缺失的类型: 01(propeller), 02(low_voltage), 03(wind_affect), 04(load_lose)
        # 新映射: 0=No_Fault, 1=Motor, 2=Accelerometer, 3=Gyroscope, 
        #        4=Magnetometer, 5=Barometer, 6=GPS
        fault_code_to_label = {
            '00': 1,   # motor → Motor (标签1)
            '05': 2,   # accelerometer → Accelerometer (标签2)
            '06': 3,   # gyroscope → Gyroscope (标签3)
            '07': 4,   # magnetometer → Magnetometer (标签4)
            '08': 5,   # barometer → Barometer (标签5)
            '09': 6,   # GPS → GPS (标签6)
            '10': 0,   # No fault → No_Fault (标签0)
        }
        
        # 需要跳过的故障代码（Real域中缺失）
        skip_codes = {'01', '02', '03', '04'}
        
        # 尝试匹配 Case_XXXXXXXXXX 格式
        match = re.match(r'Case_(\d)(\d)(\d{2})(\d{6})', filename_no_ext)
        
        if match:
            dataset_code = match.group(1)
            flight_state = match.group(2)
            fault_code = match.group(3)
            sequence_num = match.group(4)
            
            # 跳过缺失的故障类型
            if fault_code in skip_codes:
                return -1  # 返回-1表示跳过此文件
            
            if fault_code in fault_code_to_label:
                return fault_code_to_label[fault_code]
            else:
                print(f"[警告] 未知故障代码 '{fault_code}'，文件: {filename}")
                return -1
        
        # 如果文件名格式不匹配，尝试从文件夹名解析
        folder_name = os.path.basename(os.path.dirname(file_path))
        
        # 7分类文件夹名映射
        fault_name_to_label = {
            'No_Fault': 0, 'NoFault': 0, 'Normal': 0, 'normal': 0, 'nofault': 0,
            'Motor': 1, 'motor': 1,
            'Accelerometer': 2, 'accelerometer': 2, 'acc': 2,
            'Gyroscope': 3, 'gyroscope': 3, 'gyro': 3,
            'Magnetometer': 4, 'magnetometer': 4, 'mag': 4,
            'Barometer': 5, 'barometer': 5, 'baro': 5,
            'GPS': 6, 'gps': 6
        }
        
        for key, label in fault_name_to_label.items():
            if key.lower() in folder_name.lower():
                return label
        
        # 如果都无法解析，发出警告
        print(f"[警告] 无法解析故障类型，文件: {filename}，文件夹: {folder_name}，默认设为No_Fault(0)")
        return 0
    
    def _load_csv_file(
        self, 
        file_path: str, 
        fault_label: int, 
        is_hil: bool = False,
        is_real: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        加载单个CSV文件并选择特征
        
        数据筛选策略：
        - HIL数据：根据 UAVState_data_fault_state 列筛选
        - Real数据：使用变点检测定位故障发生时刻
        
        Args:
            file_path: CSV文件路径
            fault_label: 故障类型标签（0=无故障，1-10=各类故障）
            is_hil: 是否为HIL数据
            is_real: 是否为Real数据
            
        Returns:
            df: 选择特征后的DataFrame；如果失败返回None
        
        维度变化：
            原始: (T_total, all_cols)
            筛选后: (T_fault, all_cols) → 仅故障阶段
            特征选择后: (T_fault, n_features=21)
        """
        try:
            # 读取CSV
            df = pd.read_csv(file_path)
            
            # ========== HIL数据处理：根据故障状态列筛选 ==========
            if is_hil:
                fault_state_col = 'UAVState_data_fault_state'
                
                if fault_state_col in df.columns:
                    if fault_label == 0:
                        # 无故障样本：保留 fault_state=0 的数据
                        mask = df[fault_state_col] == 0
                    else:
                        # 故障样本：保留 fault_state=1 的数据（故障发生阶段）
                        mask = df[fault_state_col] == 1
                    
                    df_filtered = df[mask].copy()
                    
                    # 检查筛选后是否有足够数据
                    if len(df_filtered) < self.window_size:
                        return None
                    
                    df = df_filtered
                else:
                    pass  # 没有fault_state列，使用全部数据
            
            # ========== Real数据处理：使用变点检测 ==========
            if is_real and fault_label != 0 and self.cpd_enabled:
                # 对于Real故障样本，使用变点检测定位故障起始点
                from data.change_point_detection import ChangePointDetector
                
                # 先提取用于变点检测的特征 (使用配置的特征数量)
                detection_features = []
                for feat in self.selected_features[:self.cpd_detection_features_count]:
                    if feat in df.columns:
                        detection_features.append(feat)
                    else:
                        base_name = feat.split('[')[0]
                        matched = [col for col in df.columns if base_name in col]
                        if matched:
                            detection_features.append(matched[0])
                
                if len(detection_features) >= 4:
                    detection_data = df[detection_features].values
                    
                    # 变点检测 (使用配置文件中的参数)
                    detector = ChangePointDetector(
                        method=self.cpd_method,
                        threshold=self.cpd_threshold,
                        window_size=self.cpd_window_size,
                        min_segment_length=self.cpd_min_segment_length
                    )
                    
                    fault_data, change_point = detector.get_fault_segment(
                        detection_data, 
                        fault_label
                    )
                    
                    # 如果检测到变点，截取故障阶段数据
                    if change_point > 0:
                        df = df.iloc[change_point:].copy()
                        if len(df) < self.window_size:
                            return None
            
            # ========== 特征选择 ==========
            available_features = []
            feature_mapping = {}
            
            for feat in self.selected_features:
                if feat in df.columns:
                    available_features.append(feat)
                    feature_mapping[feat] = feat
                else:
                    base_name = feat.split('[')[0]
                    matched = [col for col in df.columns if base_name in col]
                    if matched:
                        exact_match = [col for col in matched if feat in col]
                        if exact_match:
                            available_features.append(exact_match[0])
                            feature_mapping[feat] = exact_match[0]
                        else:
                            available_features.append(matched[0])
                            feature_mapping[feat] = matched[0]
            
            # 检查特征数量是否足够
            if len(available_features) < len(self.selected_features) * 0.6:
                return None
            
            # 选择特征
            df_selected = df[available_features].copy()
            
            # 处理缺失值
            df_selected = df_selected.interpolate(method='linear', limit_direction='both')
            df_selected = df_selected.fillna(0)
            
            return df_selected
            
        except Exception as e:
            print(f"[错误] 加载文件失败 {file_path}: {e}")
            return None
    
    def _sliding_window(self, data: np.ndarray, label: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单个时间序列应用滑动窗口分割
        
        Args:
            data: 输入数据，shape = (T, n_features)
            label: 故障类型标签
            
        Returns:
            X: 分割后的样本，shape = (N_windows, window_size, n_features)
            y: 标签数组，shape = (N_windows,)
        
        维度变化示例：
            输入: (1000, 21) 时间序列
            窗口大小: 100, 步长: 50
            滑动窗口数量: (1000 - 100) / 50 + 1 = 19
            输出: (19, 100, 21)
        """
        T, F = data.shape
        
        # 计算窗口数量
        n_windows = (T - self.window_size) // self.stride + 1
        
        if n_windows <= 0:
            return np.array([]), np.array([])
        
        # 滑动窗口分割
        X = []
        for i in range(n_windows):
            start = i * self.stride
            end = start + self.window_size
            window = data[start:end, :]  # (window_size, n_features)
            X.append(window)
        
        X = np.array(X)  # (N_windows, window_size, n_features)
        y = np.full(n_windows, label)  # (N_windows,)
        
        return X, y
    
    def _load_domain_data(self, domain: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载指定域的所有数据
        
        Args:
            domain: 域名称，"HIL" 或 "REAL"
            
        Returns:
            X: 所有样本，shape = (N_total, window_size, n_features)
            y: 所有标签，shape = (N_total,)
        
        处理流程：
            1. 遍历域目录下所有CSV文件
            2. 解析故障类型标签
            3. 如果是HIL数据，根据fault_state筛选故障阶段
            4. 加载并选择特征
            5. 滑动窗口分割
            6. 合并所有样本
        """
        domain_path = os.path.join(self.data_root, domain)
        
        if not os.path.exists(domain_path):
            raise FileNotFoundError(f"域目录不存在: {domain_path}")
        
        # 判断数据类型
        is_hil = 'hil' in domain.lower()
        is_real = 'real' in domain.lower()
        
        # 查找所有CSV文件
        csv_files = glob.glob(os.path.join(domain_path, '**', '*.csv'), recursive=True)
        print(f"[信息] {domain}域找到 {len(csv_files)} 个CSV文件")
        
        if is_hil:
            print(f"[信息] HIL数据: 根据fault_state列筛选故障阶段")
        if is_real:
            print(f"[信息] Real数据: 使用变点检测定位故障起始点")
        
        all_X, all_y = [], []
        skipped_count = 0
        
        from tqdm import tqdm
        for file_path in tqdm(csv_files, desc=f'加载{domain}数据'):
            # 解析故障标签
            label = self._get_fault_label(file_path)
            
            # 跳过缺失的故障类型（label=-1）
            if label == -1:
                skipped_count += 1
                continue
            
            # 加载CSV（传递故障标签、是否HIL/Real数据的标志）
            df = self._load_csv_file(
                file_path, 
                fault_label=label, 
                is_hil=is_hil, 
                is_real=is_real
            )
            if df is None:
                skipped_count += 1
                continue
            
            # 转换为numpy数组
            data = df.values  # (T, n_features)
            
            # 滑动窗口分割
            X, y = self._sliding_window(data, label)
            
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        
        # 合并所有样本
        if len(all_X) == 0:
            raise ValueError(f"[错误] {domain}域没有有效数据！")
        
        X_concat = np.concatenate(all_X, axis=0)  # (N_total, window_size, n_features)
        y_concat = np.concatenate(all_y, axis=0)  # (N_total,)
        
        print(f"\n[信息] {domain}域数据加载完成:")
        print(f"  - 成功加载: {len(csv_files) - skipped_count} 个文件")
        print(f"  - 跳过: {skipped_count} 个文件")
        print(f"  - 数据形状: X.shape={X_concat.shape}, y.shape={y_concat.shape}")
        print(f"  - 类别分布: {np.bincount(y_concat, minlength=11)}")
        
        return X_concat, y_concat
    
    def _fit_scaler(self, X_source: np.ndarray) -> None:
        """
        基于源域数据拟合标准化器
        
        Args:
            X_source: 源域数据，shape = (N, T, F)
        
        注意：
            - 标准化应基于源域训练数据拟合，以避免数据泄露
            - 同一个Scaler用于源域和目标域，保证分布对齐的公平性
        """
        # 将数据reshape为2D进行拟合: (N*T, F)
        N, T, F = X_source.shape
        X_flat = X_source.reshape(-1, F)
        
        if self.normalization == 'zscore':
            self.scaler = StandardScaler()
        elif self.normalization == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            return
        
        self.scaler.fit(X_flat)
        print(f"[信息] 标准化器({self.normalization})已拟合，基于源域数据")
    
    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """
        应用标准化器
        
        Args:
            X: 输入数据，shape = (N, T, F)
            
        Returns:
            X_scaled: 标准化后的数据，shape = (N, T, F)
        """
        if self.scaler is None:
            return X
        
        N, T, F = X.shape
        X_flat = X.reshape(-1, F)
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(N, T, F)
        
        return X_scaled
    
    def process(self, save_processed: bool = True) -> Dict[str, np.ndarray]:
        """
        执行完整的数据预处理流程
        
        Args:
            save_processed: 是否保存处理后的数据
            
        Returns:
            data_dict: 包含以下键的字典
                - 'X_source_train': 源域训练数据
                - 'y_source_train': 源域训练标签
                - 'X_source_val': 源域验证数据
                - 'y_source_val': 源域验证标签
                - 'X_target_train': 目标域训练数据（无标签用于域适应）
                - 'y_target_train': 目标域训练标签（仅评估时使用）
                - 'X_target_test': 目标域测试数据
                - 'y_target_test': 目标域测试标签
        
        维度变化总结：
        ---------------
        源域 (HIL):
            原始: ~2566个CSV文件
            处理后: (N_source, 100, 21) ≈ (50000+, 100, 21)
        
        目标域 (Real):
            原始: ~497个CSV文件  
            处理后: (N_target, 100, 21) ≈ (10000+, 100, 21)
        """
        print("=" * 60)
        print("开始数据预处理...")
        print("=" * 60)
        
        # -------------------- 1. 加载源域数据 (HIL) --------------------
        print("\n>>> 加载源域(HIL)数据...")
        source_domain = self.config['data']['source_domain']
        X_source, y_source = self._load_domain_data(source_domain)
        
        # -------------------- 2. 加载目标域数据 (Real) --------------------
        print("\n>>> 加载目标域(Real)数据...")
        target_domain = self.config['data']['target_domain']
        X_target, y_target = self._load_domain_data(target_domain)
        
        # -------------------- 3. 拟合标准化器（基于源域） --------------------
        print("\n>>> 拟合标准化器...")
        self._fit_scaler(X_source)
        
        # -------------------- 4. 应用标准化 --------------------
        print("\n>>> 应用标准化...")
        X_source = self._apply_scaler(X_source)
        X_target = self._apply_scaler(X_target)
        
        # -------------------- 5. 数据集划分 --------------------
        print("\n>>> 划分数据集...")
        
        # 源域划分：训练集 + 验证集
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
            X_source, y_source,
            test_size=(1 - train_ratio),
            random_state=self.random_seed,
            stratify=y_source  # 分层采样保证类别平衡
        )
        
        # 目标域划分：训练集（用于域适应）+ 测试集（最终评估）
        X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
            X_target, y_target,
            test_size=self.config['data']['test_ratio'],
            random_state=self.random_seed,
            stratify=y_target
        )
        
        # -------------------- 6. 输出统计信息 --------------------
        print("\n" + "=" * 60)
        print("数据预处理完成！")
        print("=" * 60)
        print(f"\n源域 (HIL) 数据统计:")
        print(f"  训练集: X_shape={X_source_train.shape}, 类别分布={np.bincount(y_source_train)}")
        print(f"  验证集: X_shape={X_source_val.shape}, 类别分布={np.bincount(y_source_val)}")
        print(f"\n目标域 (Real) 数据统计:")
        print(f"  训练集: X_shape={X_target_train.shape}, 类别分布={np.bincount(y_target_train)}")
        print(f"  测试集: X_shape={X_target_test.shape}, 类别分布={np.bincount(y_target_test)}")
        
        # -------------------- 7. 组装返回数据 --------------------
        data_dict = {
            'X_source_train': X_source_train.astype(np.float32),
            'y_source_train': y_source_train.astype(np.int64),
            'X_source_val': X_source_val.astype(np.float32),
            'y_source_val': y_source_val.astype(np.int64),
            'X_target_train': X_target_train.astype(np.float32),
            'y_target_train': y_target_train.astype(np.int64),
            'X_target_test': X_target_test.astype(np.float32),
            'y_target_test': y_target_test.astype(np.int64),
        }
        
        # -------------------- 8. 保存处理后的数据 --------------------
        if save_processed:
            processed_dir = self.config['data']['processed_dir']
            os.makedirs(processed_dir, exist_ok=True)
            
            save_path = os.path.join(processed_dir, 'processed_data.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(data_dict, f)
            
            # 保存Scaler
            scaler_path = os.path.join(processed_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"\n[信息] 处理后数据已保存至: {save_path}")
            print(f"[信息] 标准化器已保存至: {scaler_path}")
        
        return data_dict


def load_processed_data(processed_dir: str) -> Dict[str, np.ndarray]:
    """
    加载已处理的数据
    
    Args:
        processed_dir: 处理后数据的保存目录
        
    Returns:
        data_dict: 预处理后的数据字典
    """
    data_path = os.path.join(processed_dir, 'processed_data.pkl')
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    print(f"[信息] 已加载处理后的数据: {data_path}")
    return data_dict


if __name__ == "__main__":
    """
    独立运行此脚本进行数据预处理测试
    """
    import sys
    
    # 获取配置文件路径
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    
    print("=" * 60)
    print("UAV-DANN 数据预处理模块测试")
    print("=" * 60)
    
    # 创建预处理器并执行
    preprocessor = DataPreprocessor(config_path=config_path)
    data_dict = preprocessor.process(save_processed=True)
    
    print("\n测试完成！")
