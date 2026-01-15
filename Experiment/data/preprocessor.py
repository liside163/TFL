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
