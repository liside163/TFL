"""
评估指标计算模块
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)


class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def compute_all(y_true, y_pred):
        """
        计算所有评估指标

        参数:
            y_true: 真实标签，array-like
            y_pred: 预测标签，array-like

        返回:
            metrics: dict, 包含所有指标
                - f1_macro: 宏平均F1分数
                - f1_weighted: 加权平均F1分数
                - f1_per_class: 每个类别的F1分数
                - precision_macro: 宏平均精确率
                - recall_macro: 宏平均召回率
                - accuracy: 准确率
        """
        return {
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_per_class': f1_score(y_true, y_pred, average=None),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'accuracy': accuracy_score(y_true, y_pred),
        }

    @staticmethod
    def compute_confusion_matrix(y_true, y_pred, labels=None):
        """
        计算混淆矩阵

        参数:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 类别标签列表

        返回:
            cm: 混淆矩阵，shape (n_classes, n_classes)
        """
        return confusion_matrix(y_true, y_pred, labels=labels)

    @staticmethod
    def compute_transfer_rate(source_f1, target_f1):
        """
        计算迁移率

        参数:
            source_f1: 源域F1分数
            target_f1: 目标域F1分数

        返回:
            transfer_rate: 迁移率百分比
        """
        if source_f1 == 0:
            return 0.0
        return (target_f1 / source_f1) * 100

    @staticmethod
    def compute_classification_report(y_true, y_pred, target_names=None):
        """
        生成分类报告

        参数:
            y_true: 真实标签
            y_pred: 预测标签
            target_names: 类别名称列表

        返回:
            report: 分类报告字符串
        """
        return classification_report(y_true, y_pred, target_names=target_names)

    @staticmethod
    def compute_per_class_metrics(y_true, y_pred, class_names=None):
        """
        计算每个类别的详细指标

        参数:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表

        返回:
            per_class_metrics: dict, 每个类别的指标
        """
        cm = confusion_matrix(y_true, y_pred)

        # 计算每个类别的precision, recall, F1
        n_classes = cm.shape[0]
        per_class_metrics = {}

        for i in range(n_classes):
            tp = cm[i, i]  # True Positive
            fp = cm[:, i].sum() - tp  # False Positive
            fn = cm[i, :].sum() - tp  # False Negative
            tn = cm.sum() - tp - fp - fn  # True Negative

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            class_name = class_names[i] if class_names else f"Class_{i}"

            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'support': int(cm[i, :].sum())
            }

        return per_class_metrics
