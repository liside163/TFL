# -*- coding: utf-8 -*-
"""
==============================================================================
单工况数据预处理模块
==============================================================================
功能：为单工况（单一飞行状态）迁移学习提供数据预处理
- 按飞行状态筛选数据 (hover/waypoint/velocity/circling/acce/dece)
- 保留所有故障类型，仅筛选特定工况

【关键概念】
- 工况 (Condition): 飞行状态，共6种
  0=hover, 1=waypoint, 2=velocity, 3=circling, 4=acce, 5=dece
- 故障类型 (Fault Type): 共11种 (HIL+Real共有7种)

维度变化说明：
--------------
混合模式 (所有工况):
  X_source: (N_total, 100, 21) ≈ (50000+, 100, 21)
  
单工况模式 (仅一种飞行状态):
  X_source: (N_single, 100, 21) ≈ (8000, 100, 21)
  数据量约为混合模式的 1/6

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import sys
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

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from path_utils import process_config_paths


class SingleConditionPreprocessor:
    """
    单工况数据预处理器
    
    按飞行状态（工况）筛选数据，进行 HIL→REAL 域适应的数据预处理
    
    Attributes:
        config (dict): 主配置字典
        sc_config (dict): 单工况配置字典
        condition (int): 当前处理的飞行状态 (0-5)
    
    文件名格式: Case_[A][B][CD][EFGHIJ].csv
    - [B] 位置是飞行状态代码，用于筛选工况
    """
    
    def __init__(
        self, 
        config_path: str = None, 
        sc_config_path: str = None,
        condition: int = None
    ):
        """
        初始化单工况预处理器
        
        Args:
            config_path: 主配置文件路径
            sc_config_path: 单工况配置文件路径
            condition: 飞行状态代码 (0-5)
                0=hover, 1=waypoint, 2=velocity, 
                3=circling, 4=acce, 5=dece
        """
        # 加载主配置
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.config = process_config_paths(self.config)
        
        # 加载单工况配置
        if sc_config_path is None:
            sc_config_path = os.path.join(project_root, 'config', 'config_single_condition.yaml')
        
        with open(sc_config_path, 'r', encoding='utf-8') as f:
            self.sc_config = yaml.safe_load(f)
        self.sc_config = process_config_paths(self.sc_config)
        
        # 设置飞行状态
        if condition is not None:
            self.condition = condition
        else:
            self.condition = self.sc_config['single_condition']['current_condition']
        
        # 验证工况有效性
        if self.condition not in range(6):
            raise ValueError(f"飞行状态代码必须在0-5之间，当前值: {self.condition}")
        
        # 数据路径
        self.data_root = self.config['data']['data_root']
        if os.name == 'nt' and isinstance(self.data_root, str) and self.data_root.startswith('/mnt/'):
            # WSL-style path -> Windows path (e.g. /mnt/d/... -> D:\...)
            drive_letter = self.data_root[5:6]
            remainder = self.data_root[6:]
            if drive_letter.isalpha():
                self.data_root = f"{drive_letter.upper()}:{remainder}"
                self.data_root = os.path.normpath(self.data_root)
        self.random_seed = self.config['data']['random_seed']
        
        # 预处理参数
        self.window_size = self.config['preprocessing']['window_size']
        self.stride = self.config['preprocessing']['stride']
        self.selected_features = self.config['preprocessing']['selected_features']
        
        # 标准化器
        self.scaler = None
        
        # 工况名称
        conditions_cfg = self.sc_config.get('conditions', {})
        condition_names = conditions_cfg.get('names', {})
        condition_descs = conditions_cfg.get('descriptions', {})
        self.condition_name = condition_names.get(
            self.condition, f"Condition_{self.condition}"
        )
        self.condition_desc = condition_descs.get(self.condition, "")
        
        print(f"[单工况预处理] 初始化完成")
        print(f"  - 飞行状态: {self.condition} ({self.condition_name})")
        print(f"  - 描述: {self.condition_desc}")
    
    def _parse_filename(self, file_path: str) -> Dict:
        """
        解析文件名获取子数据集、飞行状态和故障类型
        
        文件名格式: Case_[A][B][CD][EFGHIJ].csv
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析结果字典，包含 subdataset, condition, fault_code
            如果解析失败返回 None
        """
        filename = os.path.basename(file_path)
        
        if not filename.startswith('Case_') or not filename.endswith('.csv'):
            return None
        
        try:
            case_code = filename.replace('Case_', '').replace('.csv', '')
            
            if len(case_code) < 4:
                return None
            
            return {
                'subdataset': int(case_code[0]),    # [A] 子数据集
                'condition': int(case_code[1]),      # [B] 飞行状态！
                'fault_code': case_code[2:4],        # [CD] 故障类型
                'sequence': case_code[4:]            # [EFGHIJ] 序列号
            }
            
        except Exception:
            return None
    
    def _get_fault_label(self, fault_code: str) -> int:
        """
        将故障代码转换为标签
        
        Args:
            fault_code: 两位故障代码字符串 (00-10)
            
        Returns:
            故障标签 (0-6)，如果应跳过则返回 -1
        """
        # 跳过不存在于Real域的故障类型
        skip_codes = self.config['fault_types'].get('skip_codes', [])
        if fault_code in skip_codes:
            return -1
        
        # 使用映射转换标签
        code_to_label = self.config['fault_types']['code_to_label']
        if fault_code in code_to_label:
            return code_to_label[fault_code]
        
        return -1
    
    def _load_csv_file(
        self,
        file_path: str,
        fault_label: int,
        is_hil: bool = False,
        is_real: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        加载单个CSV文件并选择特征

        Args:
            file_path: CSV文件路径
            fault_label: 故障标签
            is_hil: 是否为HIL数据
            is_real: 是否为Real数据

        Returns:
            处理后的DataFrame，或None（如果加载失败）
        """
        try:
            df_full = pd.read_csv(file_path)

            # 检查必要特征是否存在
            missing_features = [f for f in self.selected_features if f not in df_full.columns]
            if len(missing_features) > 0:
                return None

            # ==================== 修复：HIL数据处理逻辑 ====================
            # 关键理解：
            # - HIL数据的故障类型由文件名决定（Motor_xxx.csv, GPS_xxx.csv等）
            # - fault_state列只是二值标志: 0=正常, >0=故障
            # - 不同文件已经按故障类型分类，无需再用fault_state区分类型
            if is_hil:
                if 'UAVState_data_fault_state' in df_full.columns:
                    if fault_label == 0:  # No_Fault - 需要正常阶段数据
                        mask = df_full['UAVState_data_fault_state'] == 0
                    else:  # 任何故障类型 - 需要故障阶段数据
                        # 所有fault_state>0的数据都是故障数据
                        # 具体是什么故障类型由文件名决定
                        mask = df_full['UAVState_data_fault_state'] > 0

                    if mask.sum() == 0:
                        return None

                    df_full = df_full.loc[mask].reset_index(drop=True)

            # 选择特征
            df = df_full[self.selected_features].copy()

            # Real数据处理：使用变点检测（仅对故障类型）
            if is_real and fault_label != 0:
                try:
                    from data.change_point_detection import ChangePointDetector

                    cpd_config = self.config.get('change_point_detection', {})
                    if cpd_config.get('enabled', True):
                        detector = ChangePointDetector(
                            threshold=cpd_config.get('threshold', 2.5),
                            window_size=cpd_config.get('window_size', 30)
                        )

                        detection_features_count = cpd_config.get('detection_features_count', 8)
                        detection_data = df.iloc[:, :detection_features_count].values

                        change_points = detector.detect(detection_data)

                        if len(change_points) > 0:
                            first_cp = change_points[0]
                            df = df.iloc[first_cp:].reset_index(drop=True)
                except Exception:
                    pass

            # 检查数据长度
            if len(df) < self.window_size:
                return None

            return df

        except Exception as e:
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
        """
        n_samples = (len(data) - self.window_size) // self.stride + 1
        
        if n_samples <= 0:
            return np.array([]), np.array([])
        
        X = np.zeros((n_samples, self.window_size, data.shape[1]))
        
        for i in range(n_samples):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            X[i] = data[start_idx:end_idx]
        
        y = np.full(n_samples, label)
        
        return X, y
    
    def _load_domain_data_single_condition(
        self, 
        domain: str, 
        condition: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载指定域的单工况数据
        
        Args:
            domain: 域名称，"HIL" 或 "REAL"
            condition: 飞行状态代码 (0-5)
            
        Returns:
            X: 样本数据，shape = (N, window_size, n_features)
            y: 标签数组，shape = (N,)
        """
        domain_path = os.path.join(self.data_root, domain)
        
        if not os.path.exists(domain_path):
            raise FileNotFoundError(f"域目录不存在: {domain_path}")
        
        is_hil = 'hil' in domain.lower()
        is_real = 'real' in domain.lower()
        
        # 查找所有CSV文件
        csv_files = glob.glob(os.path.join(domain_path, '**', '*.csv'), recursive=True)
        print(f"[信息] {domain}域找到 {len(csv_files)} 个CSV文件")
        print(f"[信息] 筛选飞行状态: {condition} ({self.condition_name})")
        
        all_X, all_y = [], []
        loaded_count = 0
        skipped_wrong_condition = 0
        skipped_wrong_fault = 0
        skipped_process_error = 0
        
        from tqdm import tqdm
        for file_path in tqdm(csv_files, desc=f'加载{domain}数据 ({self.condition_name})'):
            # 解析文件名
            parsed = self._parse_filename(file_path)
            
            if parsed is None:
                skipped_process_error += 1
                continue
            
            # 筛选飞行状态 (工况)
            if parsed['condition'] != condition:
                skipped_wrong_condition += 1
                continue
            
            # 获取故障标签
            label = self._get_fault_label(parsed['fault_code'])
            
            if label == -1:
                skipped_wrong_fault += 1
                continue
            
            # 加载CSV
            df = self._load_csv_file(
                file_path, 
                fault_label=label, 
                is_hil=is_hil, 
                is_real=is_real
            )
            
            if df is None:
                skipped_process_error += 1
                continue
            
            # 转换并分割
            data = df.values
            X, y = self._sliding_window(data, label)
            
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                loaded_count += 1
        
        # 合并所有样本
        if len(all_X) == 0:
            raise ValueError(f"[错误] {domain}域在工况 {condition} ({self.condition_name}) 下没有有效数据！")
        
        X_concat = np.concatenate(all_X, axis=0)
        y_concat = np.concatenate(all_y, axis=0)
        
        print(f"\n[信息] {domain}域单工况数据加载完成:")
        print(f"  - 飞行状态: {condition} ({self.condition_name})")
        print(f"  - 成功加载: {loaded_count} 个文件")
        print(f"  - 跳过 (工况不匹配): {skipped_wrong_condition} 个")
        print(f"  - 跳过 (故障类型): {skipped_wrong_fault} 个")
        print(f"  - 跳过 (处理错误): {skipped_process_error} 个")
        print(f"  - 数据形状: X.shape={X_concat.shape}, y.shape={y_concat.shape}")
        print(f"  - 类别分布: {np.bincount(y_concat, minlength=7)}")
        
        return X_concat, y_concat
    
    def _fit_scaler(self, X_source: np.ndarray) -> None:
        """基于源域数据拟合标准化器"""
        norm_type = self.config['preprocessing'].get('normalization', 'zscore')
        
        if norm_type == 'zscore':
            self.scaler = StandardScaler()
        elif norm_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            return
        
        N, T, F = X_source.shape
        X_flat = X_source.reshape(-1, F)
        self.scaler.fit(X_flat)
        
        print(f"[信息] 标准化器拟合完成 ({norm_type})")
    
    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """应用标准化器"""
        if self.scaler is None:
            return X
        
        N, T, F = X.shape
        X_flat = X.reshape(-1, F)
        X_scaled = self.scaler.transform(X_flat)
        
        return X_scaled.reshape(N, T, F)
    
    def process(self, save_processed: bool = True) -> Dict[str, np.ndarray]:
        """
        执行单工况数据预处理流程
        
        Args:
            save_processed: 是否保存处理后的数据
            
        Returns:
            data_dict: 包含训练/验证/测试数据的字典
        """
        print("=" * 60)
        print(f"开始单工况数据预处理...")
        print(f"飞行状态: {self.condition} ({self.condition_name})")
        print("=" * 60)
        
        # 1. 加载源域数据 (HIL)
        print("\n>>> 加载源域(HIL)数据...")
        source_domain = self.config['data']['source_domain']
        X_source, y_source = self._load_domain_data_single_condition(
            source_domain, self.condition
        )
        
        # 2. 加载目标域数据 (Real)
        print("\n>>> 加载目标域(Real)数据...")
        target_domain = self.config['data']['target_domain']
        X_target, y_target = self._load_domain_data_single_condition(
            target_domain, self.condition
        )
        
        # 3. 拟合标准化器
        print("\n>>> 拟合标准化器...")
        self._fit_scaler(X_source)
        
        # 4. 应用标准化
        print("\n>>> 应用标准化...")
        X_source = self._apply_scaler(X_source)
        X_target = self._apply_scaler(X_target)
        
        # 5. 数据集划分
        print("\n>>> 划分数据集...")
        
        train_ratio = self.config['data']['train_ratio']
        
        X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
            X_source, y_source,
            test_size=(1 - train_ratio),
            random_state=self.random_seed,
            stratify=y_source
        )
        
        X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
            X_target, y_target,
            test_size=self.config['data']['test_ratio'],
            random_state=self.random_seed,
            stratify=y_target
        )
        
        # 6. 输出统计信息
        print("\n" + "=" * 60)
        print(f"单工况数据预处理完成！")
        print(f"飞行状态: {self.condition} ({self.condition_name})")
        print("=" * 60)
        print(f"\n源域 (HIL) 数据统计:")
        print(f"  训练集: X_shape={X_source_train.shape}, 类别分布={np.bincount(y_source_train, minlength=7)}")
        print(f"  验证集: X_shape={X_source_val.shape}, 类别分布={np.bincount(y_source_val, minlength=7)}")
        print(f"\n目标域 (Real) 数据统计:")
        print(f"  训练集: X_shape={X_target_train.shape}, 类别分布={np.bincount(y_target_train, minlength=7)}")
        print(f"  测试集: X_shape={X_target_test.shape}, 类别分布={np.bincount(y_target_test, minlength=7)}")
        
        # 7. 组装返回数据
        data_dict = {
            'X_source_train': X_source_train.astype(np.float32),
            'y_source_train': y_source_train.astype(np.int64),
            'X_source_val': X_source_val.astype(np.float32),
            'y_source_val': y_source_val.astype(np.int64),
            'X_target_train': X_target_train.astype(np.float32),
            'y_target_train': y_target_train.astype(np.int64),
            'X_target_test': X_target_test.astype(np.float32),
            'y_target_test': y_target_test.astype(np.int64),
            'condition': self.condition,
            'condition_name': self.condition_name,
            'num_classes': self.config['fault_types']['num_classes']
        }
        
        # 8. 保存处理后的数据
        if save_processed:
            output_dir = self.sc_config['output']['results_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            save_path = os.path.join(
                output_dir, 
                f'processed_data_condition_{self.condition}.pkl'
            )
            with open(save_path, 'wb') as f:
                pickle.dump(data_dict, f)
            
            scaler_path = os.path.join(
                output_dir, 
                f'scaler_condition_{self.condition}.pkl'
            )
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"\n[信息] 处理后数据已保存至: {save_path}")
        
        return data_dict


def load_single_condition_data(condition: int, results_dir: str = None) -> Dict[str, np.ndarray]:
    """
    加载已处理的单工况数据
    
    Args:
        condition: 飞行状态代码 (0-5)
        results_dir: 结果目录
    """
    if results_dir is None:
        sc_config_path = os.path.join(project_root, 'config', 'config_single_condition.yaml')
        with open(sc_config_path, 'r', encoding='utf-8') as f:
            sc_config = yaml.safe_load(f)
        results_dir = sc_config['output']['results_dir']
    
    load_path = os.path.join(results_dir, f'processed_data_condition_{condition}.pkl')
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"未找到工况 {condition} 的处理数据: {load_path}")
    
    with open(load_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    print(f"[信息] 已加载工况 {condition} ({data_dict.get('condition_name', '')}) 的处理数据")
    print(f"  - 源域训练集: {data_dict['X_source_train'].shape}")
    print(f"  - 目标域测试集: {data_dict['X_target_test'].shape}")
    
    return data_dict


if __name__ == "__main__":
    """
    独立运行测试
    
    用法：
        python preprocess_single_condition.py --condition 0
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='单工况数据预处理')
    parser.add_argument('--condition', type=int, default=0, 
                        help='飞行状态代码 (0-5): 0=hover, 1=waypoint, 2=velocity, '
                             '3=circling, 4=acce, 5=dece')
    parser.add_argument('--config', type=str, default='./config/config.yaml')
    parser.add_argument('--sc_config', type=str, default='./config/config_single_condition.yaml')
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    if not os.path.isabs(args.sc_config):
        args.sc_config = os.path.join(project_root, args.sc_config)
    
    print("=" * 60)
    print("单工况数据预处理测试")
    print("=" * 60)
    
    preprocessor = SingleConditionPreprocessor(
        config_path=args.config,
        sc_config_path=args.sc_config,
        condition=args.condition
    )
    
    data_dict = preprocessor.process(save_processed=True)
    
    print("\n测试完成！")
