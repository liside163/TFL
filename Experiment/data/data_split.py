"""
数据划分模块
负责将HIL数据划分为源域、目标域训练集、目标域测试集
"""

import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from config import Config


class DataSplitter:
    """数据划分器"""

    def __init__(self, config=None):
        self.config = config or Config()

    def split_target_domain(self, X, y, test_ratio=None, stratify=True):
        """
        划分目标域数据为微调集和测试集

        参数:
            X: np.ndarray, [n_samples, window_size, features]
                时间窗口数据
            y: np.ndarray, [n_samples]
                标签
            test_ratio: float
                测试集比例 (默认从config读取)
            stratify: bool
                是否分层抽样 (保持类别分布)

        返回:
            X_finetune, y_finetune: 微调集
            X_test, y_test: 测试集
        """
        test_ratio = test_ratio or self.config.TARGET_TEST_RATIO

        if stratify:
            # 分层抽样
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=self.config.RANDOM_SEED,
                stratify=y
            )
        else:
            # 随机划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=self.config.RANDOM_SEED
            )

        print(f"目标域划分完成:")
        print(f"  微调集: {len(X_train)} 样本, 类别分布: {Counter(y_train)}")
        print(f"  测试集: {len(X_test)} 样本, 类别分布: {Counter(y_test)}")

        return (X_train, y_train), (X_test, y_test)

    def create_source_target_split(self, source_windows, source_labels,
                                   target_windows, target_labels):
        """
        创建源域和目标域的完整划分

        参数:
            source_windows, source_labels: 源域数据
            target_windows, target_labels: 目标域数据

        返回:
            dict: 包含所有划分的数据
                {
                    'source': {'X': source_windows, 'y': source_labels},
                    'target_train': {'X': finetune_windows, 'y': finetune_labels},
                    'target_test': {'X': test_windows, 'y': test_labels}
                }
        """
        # 划分目标域
        (target_train_X, target_train_y), (target_test_X, target_test_y) = \
            self.split_target_domain(target_windows, target_labels)

        return {
            'source': {
                'X': source_windows,
                'y': source_labels
            },
            'target_train': {
                'X': target_train_X,
                'y': target_train_y
            },
            'target_test': {
                'X': target_test_X,
                'y': target_test_y
            }
        }

    def report_dataset_stats(self, data_dict):
        """
        报告数据集统计信息

        参数:
            data_dict: create_source_target_split的返回结果
        """
        print("\n" + "="*60)
        print("数据集统计信息")
        print("="*60)

        for split_name, split_data in data_dict.items():
            X = split_data['X']
            y = split_data['y']

            print(f"\n{split_name}:")
            print(f"  样本数: {len(X)}")
            print(f"  特征维度: {X.shape}")
            print(f"  类别分布:")

            from collections import Counter
            label_counts = Counter(y)
            for label, count in sorted(label_counts.items()):
                print(f"    类别{label}: {count} ({count/len(y)*100:.1f}%)")

        print("\n" + "="*60)
