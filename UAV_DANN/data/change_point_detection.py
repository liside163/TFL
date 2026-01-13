# -*- coding: utf-8 -*-
"""
==============================================================================
变点检测模块 (Change Point Detection)
==============================================================================
功能：检测时间序列数据中的突变点，用于定位故障发生时刻

算法实现：
1. CUSUM (累积和) - 快速检测均值突变
2. PELT (Pruned Exact Linear Time) - 基于惩罚的精确方法
3. 滑动窗口统计量变化检测

对于Real数据：没有fault_state标签，需要通过变点检测来识别故障发生时刻

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ChangePointDetector:
    """
    变点检测器
    
    用于检测时间序列中的突变点，定位故障发生时刻
    
    Attributes:
        method: 检测方法 ('cusum', 'zscore', 'combined')
        threshold: 检测阈值
        window_size: 滑动窗口大小
    """
    
    def __init__(
        self,
        method: str = 'combined',
        threshold: float = 3.0,
        window_size: int = 50,
        min_segment_length: int = 100
    ):
        """
        初始化变点检测器
        
        Args:
            method: 检测方法
                - 'cusum': 累积和方法，检测均值突变
                - 'zscore': Z-score方法，检测异常点
                - 'combined': 综合多种方法
            threshold: 检测阈值（Z-score的倍数）
            window_size: 滑动窗口大小
            min_segment_length: 最小分段长度（防止过度分割）
        """
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self.min_segment_length = min_segment_length
    
    def detect(self, data: np.ndarray) -> List[int]:
        """
        检测变点
        
        Args:
            data: 输入数据，shape = (T, F) 或 (T,)
                  T: 时间步数
                  F: 特征数
        
        Returns:
            change_points: 变点位置列表（时间索引）
        """
        # 如果是多维数据，计算综合统计量
        if data.ndim == 2:
            # 方法1：计算每个时间步的L2范数变化
            data_norm = np.linalg.norm(data, axis=1)
            # 方法2：计算主要特征的变化
            data_mean = np.mean(data, axis=1)
            # 综合考虑
            signal = data_norm
        else:
            signal = data
        
        if self.method == 'cusum':
            return self._detect_cusum(signal)
        elif self.method == 'zscore':
            return self._detect_zscore(signal)
        elif self.method == 'combined':
            return self._detect_combined(signal, data if data.ndim == 2 else None)
        else:
            raise ValueError(f"未知的检测方法: {self.method}")
    
    def _detect_cusum(self, signal: np.ndarray) -> List[int]:
        """
        CUSUM累积和变点检测
        
        原理：
        - 计算信号与均值的累积偏差
        - 当累积和超过阈值时，检测到变点
        
        Args:
            signal: 1D信号，shape = (T,)
        
        Returns:
            change_points: 变点位置列表
        """
        n = len(signal)
        if n < self.window_size * 2:
            return []
        
        # 使用前window_size个点估计基准均值和标准差
        baseline_mean = np.mean(signal[:self.window_size])
        baseline_std = np.std(signal[:self.window_size]) + 1e-8
        
        # 计算标准化残差
        residuals = (signal - baseline_mean) / baseline_std
        
        # 累积和
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        
        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i-1] + residuals[i] - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i-1] + residuals[i] + 0.5)
        
        # 检测超过阈值的点
        change_points = []
        threshold_value = self.threshold * np.sqrt(self.window_size)
        
        i = self.window_size
        while i < n:
            if cusum_pos[i] > threshold_value or abs(cusum_neg[i]) > threshold_value:
                change_points.append(i)
                # 跳过最小分段长度
                i += self.min_segment_length
                # 重置CUSUM
                if i < n:
                    cusum_pos[i] = 0
                    cusum_neg[i] = 0
            else:
                i += 1
        
        return change_points
    
    def _detect_zscore(self, signal: np.ndarray) -> List[int]:
        """
        Z-score滑动窗口变点检测
        
        原理：
        - 计算每个点相对于前一段数据的Z-score
        - 当Z-score超过阈值时，检测到变点
        
        Args:
            signal: 1D信号，shape = (T,)
        
        Returns:
            change_points: 变点位置列表
        """
        n = len(signal)
        if n < self.window_size * 2:
            return []
        
        change_points = []
        i = self.window_size
        
        while i < n - self.window_size:
            # 前窗口和后窗口
            window_before = signal[i-self.window_size:i]
            window_after = signal[i:i+self.window_size]
            
            # 计算统计量
            mean_before = np.mean(window_before)
            std_before = np.std(window_before) + 1e-8
            mean_after = np.mean(window_after)
            
            # Z-score
            z_score = abs(mean_after - mean_before) / std_before
            
            if z_score > self.threshold:
                change_points.append(i)
                i += self.min_segment_length
            else:
                i += 1
        
        return change_points
    
    def _detect_combined(
        self, 
        signal: np.ndarray, 
        full_data: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        综合多种方法的变点检测
        
        结合多个特征的变化来提高检测准确性
        
        Args:
            signal: 主信号（如L2范数）
            full_data: 完整多维数据（可选）
        
        Returns:
            change_points: 变点位置列表
        """
        n = len(signal)
        if n < self.window_size * 2:
            return []
        
        # 计算多种变化指标
        scores = np.zeros(n)
        
        for i in range(self.window_size, n - self.window_size):
            # 1. 均值变化
            mean_before = np.mean(signal[i-self.window_size:i])
            mean_after = np.mean(signal[i:i+self.window_size])
            std_before = np.std(signal[i-self.window_size:i]) + 1e-8
            mean_change = abs(mean_after - mean_before) / std_before
            
            # 2. 方差变化
            var_before = np.var(signal[i-self.window_size:i]) + 1e-8
            var_after = np.var(signal[i:i+self.window_size])
            var_ratio = max(var_after / var_before, var_before / var_after)
            
            # 3. 如果有多维数据，计算更多特征
            if full_data is not None and full_data.ndim == 2:
                # 协方差变化（简化：使用特征间相关性变化）
                corr_before = np.corrcoef(full_data[i-self.window_size:i].T)
                corr_after = np.corrcoef(full_data[i:i+self.window_size].T)
                corr_change = np.nanmean(np.abs(corr_after - corr_before))
            else:
                corr_change = 0
            
            # 综合得分
            scores[i] = mean_change + 0.5 * np.log(var_ratio) + corr_change
        
        # 找到得分超过阈值的点
        change_points = []
        i = self.window_size
        
        while i < n - self.window_size:
            if scores[i] > self.threshold:
                # 在局部区域找最大得分点
                local_end = min(i + self.window_size, n - self.window_size)
                local_max_idx = i + np.argmax(scores[i:local_end])
                change_points.append(local_max_idx)
                i = local_max_idx + self.min_segment_length
            else:
                i += 1
        
        return change_points
    
    def get_fault_segment(
        self, 
        data: np.ndarray, 
        fault_label: int
    ) -> Tuple[np.ndarray, int]:
        """
        获取故障发生后的数据段
        
        对于Real数据，检测变点后返回故障阶段的数据
        
        Args:
            data: 输入数据，shape = (T, F)
            fault_label: 故障类型标签
        
        Returns:
            fault_data: 故障阶段数据
            change_point: 检测到的变点位置
        
        处理逻辑：
        - 如果是无故障样本(label=0)：返回全部数据
        - 如果是故障样本：检测变点，返回变点之后的数据
        """
        if fault_label == 0:
            # 无故障样本，返回全部数据
            return data, 0
        
        # 检测变点
        change_points = self.detect(data)
        
        if len(change_points) == 0:
            # 未检测到变点，假设整个序列都是故障状态
            # 保守策略：返回后半段数据
            mid_point = len(data) // 2
            return data[mid_point:], mid_point
        
        # 取第一个变点（故障开始时刻）
        first_change_point = change_points[0]
        
        # 返回变点之后的数据
        return data[first_change_point:], first_change_point


def detect_fault_onset_multivariate(
    data: np.ndarray,
    key_features: List[int] = None,
    threshold: float = 2.5,
    window_size: int = 30
) -> int:
    """
    多变量故障起始点检测
    
    使用多个关键特征的联合变化来检测故障起始点
    
    Args:
        data: 输入数据，shape = (T, F)
        key_features: 用于检测的关键特征索引列表
                     如果为None，使用所有特征
        threshold: 检测阈值
        window_size: 滑动窗口大小
    
    Returns:
        onset_point: 故障起始点位置
    """
    T, F = data.shape
    
    if key_features is None:
        key_features = list(range(F))
    
    # 提取关键特征
    key_data = data[:, key_features]
    
    # 计算每个时间点的异常得分
    anomaly_scores = np.zeros(T)
    
    for t in range(window_size, T):
        # 使用前window_size个点作为参考
        ref_data = key_data[:window_size]
        ref_mean = np.mean(ref_data, axis=0)
        ref_std = np.std(ref_data, axis=0) + 1e-8
        
        # 当前点的Z-score
        current = key_data[t]
        z_scores = np.abs((current - ref_mean) / ref_std)
        
        # 综合异常得分
        anomaly_scores[t] = np.mean(z_scores)
    
    # 找到第一个持续超过阈值的点
    consecutive_count = 0
    required_consecutive = 10  # 需要连续10个点超过阈值
    
    for t in range(window_size, T):
        if anomaly_scores[t] > threshold:
            consecutive_count += 1
            if consecutive_count >= required_consecutive:
                return t - required_consecutive + 1
        else:
            consecutive_count = 0
    
    # 如果没找到，返回中点
    return T // 2


if __name__ == "__main__":
    """
    测试变点检测模块
    """
    print("=" * 60)
    print("变点检测模块测试")
    print("=" * 60)
    
    # 生成模拟数据：正常状态 + 突变 + 故障状态
    np.random.seed(42)
    
    # 正常阶段
    normal_phase = np.random.randn(200, 5) * 0.5 + 1.0
    
    # 故障阶段（均值和方差都变化）
    fault_phase = np.random.randn(300, 5) * 1.5 + 3.0
    
    # 合并
    data = np.vstack([normal_phase, fault_phase])
    
    print(f"模拟数据形状: {data.shape}")
    print(f"真实变点位置: 200")
    
    # 测试变点检测
    detector = ChangePointDetector(method='combined', threshold=2.5, window_size=30)
    change_points = detector.detect(data)
    
    print(f"\n检测到的变点: {change_points}")
    
    # 测试获取故障段
    fault_data, cp = detector.get_fault_segment(data, fault_label=1)
    print(f"故障段数据形状: {fault_data.shape}")
    print(f"检测到的变点: {cp}")
    
    # 计算检测误差
    if len(change_points) > 0:
        error = abs(change_points[0] - 200)
        print(f"检测误差: {error} 个时间步")
    
    print("\n测试完成！")
