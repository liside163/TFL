"""
全自动批量实验运行器
支持断点续跑、异常处理、进度保存
"""

import os
import json
import time
import logging
import traceback
import torch
import gc
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from config import Config


class AutoExperimentRunner:
    """全自动批量实验运行器"""

    def __init__(self, config=None):
        self.config = config or Config()

        # 创建目录
        self.checkpoint_dir = Path(self.config.SAVE_DIR) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.setup_logging()

        # 进度跟踪
        self.progress_file = self.checkpoint_dir / 'progress.json'
        self.progress = self.load_progress()

        # 结果汇总
        self.results_summary = []
        self.failed_experiments = []

    def setup_logging(self):
        """设置多层日志系统"""
        log_dir = Path(self.config.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 主日志
        self.main_logger = logging.getLogger('AutoRunner')
        self.main_logger.setLevel(logging.INFO)

        handler = logging.FileHandler(
            log_dir / f'auto_runner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.main_logger.addHandler(handler)

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.main_logger.addHandler(console_handler)

    def load_progress(self):
        """加载进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'completed': [], 'failed': [], 'current_index': 0}

    def save_progress(self):
        """保存进度"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def run_all_experiments(self, experiment_matrix):
        """
        运行所有实验

        参数:
            experiment_matrix: List[dict], 每个元素包含:
                {
                    'exp_id': 'E001',
                    'method': 'baseline',
                    'architecture': 'cnn',
                    'target_state': 1,
                    'hyperparams': {...}
                }
        """
        total = len(experiment_matrix)
        self.main_logger.info(f"="*60)
        self.main_logger.info(f"开始批量实验: 共 {total} 组")
        self.main_logger.info(f"="*60)

        for idx, exp_config in enumerate(tqdm(experiment_matrix, desc="实验进度")):
            exp_id = exp_config['exp_id']

            # 跳过已完成
            if exp_id in self.progress['completed']:
                self.main_logger.info(f"跳过已完成: {exp_id}")
                continue

            self.main_logger.info(f"\n{'='*60}")
            self.main_logger.info(f"实验 {exp_id} ({idx+1}/{total})")
            self.main_logger.info(f"配置: {exp_config}")
            self.main_logger.info(f"{'='*60}")

            try:
                # 运行单个实验
                result = self.run_single_experiment(exp_config)

                # 记录成功
                self.progress['completed'].append(exp_id)
                self.results_summary.append({
                    **exp_config,
                    'result': result,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                })

                self.main_logger.info(f"实验 {exp_id} 完成: F1={result['f1_macro']:.4f}")

            except Exception as e:
                # 记录失败
                self.progress['failed'].append(exp_id)
                error_info = {
                    **exp_config,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }
                self.failed_experiments.append(error_info)

                self.main_logger.error(f"实验 {exp_id} 失败: {str(e)}")
                self.main_logger.error(traceback.format_exc())

            finally:
                # 保存进度
                self.save_progress()
                self.save_results_summary()

                # 清理GPU
                self.cleanup()

        # 生成报告
        self.main_logger.info(f"\n{'='*60}")
        self.main_logger.info(f"所有实验完成!")
        self.main_logger.info(f"成功: {len(self.progress['completed'])}, "
                             f"失败: {len(self.progress['failed'])}")
        self.main_logger.info(f"{'='*60}")

    def run_single_experiment(self, exp_config):
        """
        运行单个实验

        参数:
            exp_config: 实验配置字典

        返回:
            result: dict, 包含指标和训练时间等
        """
        start_time = time.time()

        # 创建实验专用日志
        exp_log_dir = Path(self.config.LOG_DIR) / exp_config['exp_id']
        exp_log_dir.mkdir(parents=True, exist_ok=True)

        exp_logger = logging.getLogger(exp_config['exp_id'])
        exp_logger.addHandler(logging.FileHandler(exp_log_dir / 'experiment.log', encoding='utf-8'))
        exp_logger.setLevel(logging.INFO)

        exp_logger.info(f"开始实验: {exp_config}")

        # 这里调用实际的训练代码
        # 需要在实际使用时实现具体的训练逻辑
        # 示例:
        # from experiments.run_single_exp import run_experiment
        # result = run_experiment(exp_config, exp_logger)

        # 占位符结果
        result = {
            'f1_macro': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

        # 记录时间
        elapsed_time = time.time() - start_time
        result['elapsed_time'] = elapsed_time

        exp_logger.info(f"实验完成，耗时: {elapsed_time/60:.2f}分钟")

        return result

    def cleanup(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def save_results_summary(self):
        """保存结果汇总"""
        results_file = Path(self.config.SAVE_DIR) / 'results_summary.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results_summary, f, indent=2, ensure_ascii=False)

        failed_file = Path(self.config.SAVE_DIR) / 'failed_experiments.json'
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(self.failed_experiments, f, indent=2, ensure_ascii=False)
