# =====================================================================
# 数据预处理工具模块
# 功能: 从文件名解析元数据、特征标准化、数据筛选
# =====================================================================

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DOMAIN_MAPPING, 
    FLIGHT_STATE_MAPPING, 
    FAULT_TYPE_MAPPING_7CLASS,
    FAULT_LABEL_TO_IDX,
    DATA_ROOT
)


def parse_filename(filename: str) -> Dict[str, str]:
    """
    从文件名解析domain、飞行状态、故障类型信息
    
    文件名格式: Case_[A][B][CD][EFGHIJ].csv
    - [A]: 子数据集 (1=SIL, 2=HIL, 3=Real)
    - [B]: 飞行状态 (0-5)
    - [CD]: 故障类型 (00-10)
    - [EFGHIJ]: 序列号
    
    Args:
        filename: 文件名，如 "Case_2109000005.csv"
    
    Returns:
        包含解析信息的字典:
        {
            "domain": "HIL",
            "flight_state": "waypoint",
            "fault_code": "09",
            "fault_type": "GPS",
            "sequence_id": "000005"
        }
    """
    # 提取文件名主体 (不含扩展名)
    stem = Path(filename).stem
    
    # 匹配Case_后面的数字部分
    match = re.match(r"Case_(\d)(\d)(\d{2})(\d{6})", stem)
    if not match:
        raise ValueError(f"无法解析文件名: {filename}")
    
    domain_code, flight_code, fault_code, seq_id = match.groups()
    
    # 解析各字段
    domain = DOMAIN_MAPPING.get(domain_code, f"Unknown_{domain_code}")
    flight_state = FLIGHT_STATE_MAPPING.get(flight_code, f"Unknown_{flight_code}")
    fault_type = FAULT_TYPE_MAPPING_7CLASS.get(fault_code, None)  # 7分类中不存在的返回None
    
    return {
        "domain": domain,
        "domain_code": domain_code,
        "flight_state": flight_state,
        "flight_code": flight_code,
        "fault_code": fault_code,
        "fault_type": fault_type,
        "sequence_id": seq_id
    }


def filter_files_by_condition(
    data_dir: Path,
    domain: Optional[str] = None,
    flight_state: Optional[str] = None,
    fault_types: Optional[List[str]] = None
) -> List[Path]:
    """
    根据条件筛选数据文件
    
    支持两种目录结构:
    1. 扁平结构: data_dir/Case_*.csv
    2. 子目录结构: data_dir/HIL/Case_*.csv, data_dir/REAL/Case_*.csv
    
    Args:
        data_dir: 数据目录路径
        domain: 目标域 ("HIL", "Real", "SIL", None=全部)
        flight_state: 飞行状态 ("hover", "waypoint", etc., None=全部)
        fault_types: 故障类型列表 (只筛选这些类型, None=全部7分类)
    
    Returns:
        符合条件的文件路径列表
    """
    # 默认使用7分类的故障类型
    if fault_types is None:
        fault_types = list(FAULT_TYPE_MAPPING_7CLASS.values())
    
    filtered_files = []
    
    # 确定搜索目录列表
    # 检查是否为子目录结构
    domain_subdirs = {
        "HIL": ["HIL", "hil"],
        "Real": ["REAL", "Real", "real"],
        "SIL": ["SIL", "sil"],
    }
    
    search_dirs = []
    
    if domain is not None:
        # 指定了domain，查找对应子目录
        possible_subdirs = domain_subdirs.get(domain, [domain])
        for subdir_name in possible_subdirs:
            subdir = data_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                search_dirs.append(subdir)
                break
        
        # 如果没找到子目录，尝试在data_dir直接搜索
        if not search_dirs:
            search_dirs.append(data_dir)
    else:
        # 未指定domain，搜索所有子目录和根目录
        for domain_name, subdirs in domain_subdirs.items():
            for subdir_name in subdirs:
                subdir = data_dir / subdir_name
                if subdir.exists() and subdir.is_dir():
                    search_dirs.append(subdir)
                    break
        # 也搜索根目录
        search_dirs.append(data_dir)
    
    # 在所有搜索目录中查找文件
    for search_dir in search_dirs:
        for csv_file in search_dir.glob("Case_*.csv"):
            try:
                info = parse_filename(csv_file.name)
                
                # 检查domain条件
                if domain is not None and info["domain"] != domain:
                    continue
                
                # 检查飞行状态条件
                if flight_state is not None and info["flight_state"] != flight_state:
                    continue
                
                # 检查故障类型条件 (必须在7分类中)
                if info["fault_type"] is None:  # 不在7分类中的故障类型
                    continue
                if info["fault_type"] not in fault_types:
                    continue
                
                filtered_files.append(csv_file)
                
            except ValueError as e:
                print(f"警告: 跳过文件 {csv_file.name} - {e}")
                continue
    
    return filtered_files


def get_feature_columns(df: pd.DataFrame, domain: str) -> List[str]:
    """
    根据域获取特征列名
    
    直接使用config中定义的SHARED_FEATURES列表，在DataFrame中查找匹配的列名
    
    Args:
        df: 数据DataFrame
        domain: 数据域 ("HIL" 或 "Real")
    
    Returns:
        可用的特征列名列表
    """
    from config import SHARED_FEATURES
    
    available_cols = df.columns.tolist()
    selected_cols = []
    
    for feature_name in SHARED_FEATURES:
        if feature_name in available_cols:
            selected_cols.append(feature_name)
        else:
            # 尝试不同的命名变体
            # 有些列名可能使用下划线分隔而不是方括号
            alt_name = feature_name.replace("[", "_").replace("]", "")
            if alt_name in available_cols:
                selected_cols.append(alt_name)
            else:
                print(f"警告: {domain}数据中找不到特征 {feature_name} 或 {alt_name}")
    
    return selected_cols


class FeatureScaler:
    """
    特征标准化器
    
    使用源域(HIL)数据拟合标准化参数，然后应用于源域和目标域
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, data: np.ndarray):
        """
        使用源域数据拟合标准化器
        
        Args:
            data: 形状为 [N, seq_len, features] 或 [N*seq_len, features] 的数据
        """
        # 如果是3D数据，reshape为2D
        if len(data.shape) == 3:
            n, seq_len, features = data.shape
            data = data.reshape(-1, features)
        
        self.scaler.fit(data)
        self.is_fitted = True
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        标准化数据
        
        Args:
            data: 形状为 [N, seq_len, features] 的数据
        
        Returns:
            标准化后的数据，形状不变
        """
        if not self.is_fitted:
            raise RuntimeError("标准化器尚未拟合，请先调用fit方法")
        
        original_shape = data.shape
        
        if len(data.shape) == 3:
            n, seq_len, features = data.shape
            data = data.reshape(-1, features)
            transformed = self.scaler.transform(data)
            return transformed.reshape(original_shape)
        else:
            return self.scaler.transform(data)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并标准化数据"""
        self.fit(data)
        return self.transform(data)
    
    def save(self, path: Path):
        """保存标准化器参数"""
        np.savez(
            path,
            mean=self.scaler.mean_,
            scale=self.scaler.scale_
        )
    
    def load(self, path: Path):
        """加载标准化器参数"""
        params = np.load(path)
        self.scaler.mean_ = params["mean"]
        self.scaler.scale_ = params["scale"]
        self.is_fitted = True


def print_dataset_statistics(files: List[Path], name: str = "Dataset"):
    """
    打印数据集统计信息
    
    Args:
        files: 文件路径列表
        name: 数据集名称
    """
    fault_counts = {}
    flight_counts = {}
    
    for f in files:
        info = parse_filename(f.name)
        fault_type = info["fault_type"]
        flight_state = info["flight_state"]
        
        fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
        flight_counts[flight_state] = flight_counts.get(flight_state, 0) + 1
    
    print(f"\n{'=' * 50}")
    print(f"{name} 统计信息")
    print(f"{'=' * 50}")
    print(f"总样本数: {len(files)}")
    
    print(f"\n故障类型分布:")
    for fault, count in sorted(fault_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fault}: {count} ({100*count/len(files):.1f}%)")
    
    print(f"\n飞行状态分布:")
    for state, count in sorted(flight_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {state}: {count} ({100*count/len(files):.1f}%)")


if __name__ == "__main__":
    # 测试代码
    print("测试文件名解析...")
    test_names = [
        "Case_2109000005.csv",  # HIL, waypoint, GPS
        "Case_3000000001.csv",  # Real, hover, Motor
        "Case_2510000003.csv",  # HIL, dece, No Fault
    ]
    
    for name in test_names:
        try:
            info = parse_filename(name)
            print(f"{name}:")
            print(f"  Domain: {info['domain']}, Flight: {info['flight_state']}, "
                  f"Fault: {info['fault_type']}")
        except ValueError as e:
            print(f"  错误: {e}")
    
    # 测试文件筛选
    print(f"\n测试文件筛选 (数据目录: {DATA_ROOT})...")
    if DATA_ROOT.exists():
        hil_hover_files = filter_files_by_condition(
            DATA_ROOT, domain="HIL", flight_state="hover"
        )
        print(f"HIL + hover 文件数: {len(hil_hover_files)}")
        
        if hil_hover_files:
            print_dataset_statistics(hil_hover_files, "HIL-Hover")
    else:
        print(f"数据目录不存在: {DATA_ROOT}")
