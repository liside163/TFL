"""
超参数调优模块
支持网格搜索、随机搜索、贝叶斯优化、Hyperband
"""

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
from pathlib import Path
import json
import logging
from config import Config


class HyperparameterTuner:
    """超参数调优器"""

    def __init__(self, model_class, method_class, config=None):
        """
        参数:
            model_class: 模型类 (如CNN1D, LSTMModel等)
            method_class: 方法类 (如BaselineMethod, MMDAdapter等)
            config: 配置对象
        """
        self.model_class = model_class
        self.method_class = method_class
        self.config = config or Config()

        # 创建保存目录
        self.study_dir = Path(self.config.SAVE_DIR) / 'hyperparameter_studies'
        self.study_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger('HyperparameterTuning')

    def objective(self, trial, train_data, val_data):
        """
        Optuna目标函数

        参数:
            trial: Optuna trial对象
            train_data: 训练数据
            val_data: 验证数据

        返回:
            metric: 优化目标指标 (如F1)
        """
        # 采样超参数
        params = self.suggest_params(trial)

        # 创建模型
        model = self.model_class(**params['model_params'])

        # 创建方法实例
        method = self.method_class(model, self.config)

        # 训练 (快速验证，用较少epoch)
        try:
            history, metrics = method.train(
                train_data, val_data,
                num_epochs=20  # 快速验证
            )

            # 返回优化目标
            return metrics['f1_macro']

        except Exception as e:
            self.logger.error(f"试验失败: {str(e)}")
            return 0.0

    def suggest_params(self, trial):
        """根据配置采样超参数"""
        if self.config.SEARCH_METHOD == 'bayesian':
            return self._suggest_bayesian(trial)
        elif self.config.SEARCH_METHOD == 'random':
            return self._suggest_random(trial)
        else:
            raise ValueError(f"未知搜索方法: {self.config.SEARCH_METHOD}")

    def _suggest_bayesian(self, trial):
        """贝叶斯优化参数采样"""
        params = {
            'model_params': {
                # CNN过滤器比例 (如果适用)
                'cnn_filters_ratio': trial.suggest_float('cnn_filters_ratio', 0.5, 2.0),
            },
            'training_params': {
                # 学习率 (对数尺度)
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                # Batch size (分类)
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                # Dropout
                'dropout': trial.suggest_float('dropout', 0.0, 0.6),
                # MMD lambda
                'lambda_mmd': trial.suggest_float('lambda_mmd', 0.01, 10.0, log=True),
            }
        }
        return params

    def _suggest_random(self, trial):
        """随机搜索参数采样"""
        params = {
            'model_params': {},
            'training_params': {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.6),
                'lambda_mmd': trial.suggest_float('lambda_mmd', 0.01, 10.0, log=True),
            }
        }
        return params

    def run_bayesian_optimization(self, train_data, val_data, n_trials=None):
        """
        运行贝叶斯优化

        参数:
            train_data: 训练数据
            val_data: 验证数据
            n_trials: 试验次数

        返回:
            best_params: 最佳参数
            best_value: 最佳指标值
            study: Optuna study对象
        """
        n_trials = n_trials or self.config.BAYESIAN_OPTIMIZATION['n_trials']

        print(f"\n开始贝叶斯优化 ({n_trials} 次试验)")

        # 创建study
        study = optuna.create_study(
            direction='maximize',  # 最大化F1
            sampler=TPESampler(seed=self.config.RANDOM_SEED),
            pruner=HyperbandPruner()  # 使用Hyperband早停
        )

        # 运行优化
        study.optimize(
            lambda trial: self.objective(trial, train_data, val_data),
            n_trials=n_trials,
            timeout=self.config.BAYESIAN_OPTIMIZATION['timeout_hours'] * 3600,
            show_progress_bar=True
        )

        # 输出结果
        print("\n贝叶斯优化完成!")
        print(f"最佳F1: {study.best_value:.4f}")
        print(f"最佳参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # 保存study
        study_path = self.study_dir / 'best_study.pkl'
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)

        # 保存参数
        params_path = self.study_dir / 'best_params.json'
        with open(params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)

        # 可视化
        self._plot_optimization_history(study)

        return study.best_params, study.best_value, study

    def _plot_optimization_history(self, study):
        """绘制优化历史"""
        try:
            import matplotlib.pyplot as plt

            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(self.study_dir / 'optimization_history.png', dpi=300)

            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(self.study_dir / 'param_importances.png', dpi=300)

            print(f"优化历史和参数重要性图已保存到: {self.study_dir}")

        except Exception as e:
            print(f"绘图失败: {str(e)}")

    def run_grid_search(self, param_grid, train_data, val_data):
        """运行网格搜索"""
        from sklearn.model_selection import ParameterGrid

        print(f"\n开始网格搜索 (共{len(list(ParameterGrid(param_grid)))}组参数)")

        best_score = 0
        best_params = None

        for params in ParameterGrid(param_grid):
            print(f"测试参数: {params}")

            # 创建模型并训练
            # ... (实现类似于bayesian)

            if score > best_score:
                best_score = score
                best_params = params
                print(f"  新最佳: {score:.4f}")

        print(f"\n网格搜索完成!")
        print(f"最佳F1: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        return best_params, best_score
