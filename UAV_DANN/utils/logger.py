# -*- coding: utf-8 -*-
"""
==============================================================================
日志记录模块
==============================================================================
功能：训练日志和TensorBoard日志记录
- 控制台日志输出
- 文件日志记录
- TensorBoard可视化

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Optional, Any
import numpy as np

# 尝试导入TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[警告] TensorBoard未安装，将禁用TensorBoard日志")


def setup_logger(
    name: str = 'UAV-DANN',
    log_dir: str = './logs',
    log_level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        log_level: 日志级别
        console_output: 是否输出到控制台
    
    Returns:
        logger: 配置好的日志记录器
    
    使用方式:
    ---------
    logger = setup_logger('UAV-DANN', './logs')
    logger.info('开始训练')
    logger.warning('学习率较低')
    logger.error('发生错误')
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    
    return logger


class TensorBoardLogger:
    """
    TensorBoard日志记录器
    
    封装TensorBoard的SummaryWriter，提供便捷的日志记录接口
    
    Attributes:
        writer: TensorBoard SummaryWriter实例
        log_dir: 日志保存目录
        enabled: TensorBoard是否可用
    
    使用方式:
    ---------
    tb_logger = TensorBoardLogger('./runs/experiment1')
    
    # 记录标量
    tb_logger.log_scalar('loss/train', loss_value, epoch)
    
    # 记录多个标量
    tb_logger.log_scalars({
        'accuracy/train': train_acc,
        'accuracy/val': val_acc,
        'accuracy/target': target_acc
    }, epoch)
    
    # 记录直方图
    tb_logger.log_histogram('features', feature_tensor, epoch)
    
    # 关闭
    tb_logger.close()
    """
    
    def __init__(
        self,
        log_dir: str = './runs',
        experiment_name: Optional[str] = None,
        enabled: bool = True
    ):
        """
        初始化TensorBoard日志记录器
        
        Args:
            log_dir: 日志根目录
            experiment_name: 实验名称 (可选)
            enabled: 是否启用TensorBoard
        """
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        
        if self.enabled:
            # 创建带时间戳的日志目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if experiment_name:
                self.log_dir = os.path.join(log_dir, f'{experiment_name}_{timestamp}')
            else:
                self.log_dir = os.path.join(log_dir, timestamp)
            
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"[TensorBoard] 日志目录: {self.log_dir}")
            print(f"[TensorBoard] 运行 'tensorboard --logdir={log_dir}' 查看")
        else:
            self.writer = None
            self.log_dir = None
            if enabled and not TENSORBOARD_AVAILABLE:
                print("[警告] TensorBoard不可用，请安装: pip install tensorboard")
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int
    ) -> None:
        """
        记录标量值
        
        Args:
            tag: 标签名 (例如 'loss/train', 'accuracy/val')
            value: 数值
            step: 步数 (通常是epoch)
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(
        self,
        scalars: Dict[str, float],
        step: int
    ) -> None:
        """
        记录多个标量值
        
        Args:
            scalars: 标签-数值字典
            step: 步数
        """
        if self.enabled and self.writer is not None:
            for tag, value in scalars.items():
                self.writer.add_scalar(tag, value, step)
    
    def log_histogram(
        self,
        tag: str,
        values: Any,
        step: int
    ) -> None:
        """
        记录直方图
        
        Args:
            tag: 标签名
            values: 数据 (Tensor或ndarray)
            step: 步数
        """
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(
        self,
        tag: str,
        image: Any,
        step: int,
        dataformats: str = 'HWC'
    ) -> None:
        """
        记录图像
        
        Args:
            tag: 标签名
            image: 图像数据
            step: 步数
            dataformats: 图像格式 ('HWC', 'CHW', 'HW')
        """
        if self.enabled and self.writer is not None:
            self.writer.add_image(tag, image, step, dataformats=dataformats)
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: int
    ) -> None:
        """
        记录文本
        
        Args:
            tag: 标签名
            text: 文本内容
            step: 步数
        """
        if self.enabled and self.writer is not None:
            self.writer.add_text(tag, text, step)
    
    def log_hparams(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> None:
        """
        记录超参数和对应的指标
        
        Args:
            hparams: 超参数字典
            metrics: 指标字典
        """
        if self.enabled and self.writer is not None:
            # 过滤不支持的类型
            filtered_hparams = {}
            for k, v in hparams.items():
                if isinstance(v, (int, float, str, bool)):
                    filtered_hparams[k] = v
                elif v is None:
                    filtered_hparams[k] = 'None'
                else:
                    filtered_hparams[k] = str(v)
            
            self.writer.add_hparams(filtered_hparams, metrics)
    
    def log_model_graph(
        self,
        model: Any,
        input_data: Any
    ) -> None:
        """
        记录模型计算图
        
        Args:
            model: PyTorch模型
            input_data: 输入数据示例
        """
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_graph(model, input_data)
            except Exception as e:
                print(f"[警告] 无法记录模型图: {e}")
    
    def flush(self) -> None:
        """刷新缓冲区"""
        if self.enabled and self.writer is not None:
            self.writer.flush()
    
    def close(self) -> None:
        """关闭日志记录器"""
        if self.enabled and self.writer is not None:
            self.writer.close()
            print("[TensorBoard] 日志记录器已关闭")


class ProgressBar:
    """
    训练进度条
    
    在控制台显示训练进度
    """
    
    def __init__(self, total: int, prefix: str = '', width: int = 40):
        """
        初始化进度条
        
        Args:
            total: 总步数
            prefix: 前缀文本
            width: 进度条宽度
        """
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0
        self.metrics = {}
    
    def update(self, step: int = 1, **metrics) -> None:
        """
        更新进度条
        
        Args:
            step: 步进量
            **metrics: 要显示的指标
        """
        self.current += step
        self.metrics.update(metrics)
        
        # 计算进度
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        
        # 构建指标字符串
        metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in self.metrics.items()])
        
        # 打印进度条
        print(f'\r{self.prefix} [{bar}] {self.current}/{self.total} | {metrics_str}', end='')
        
        if self.current >= self.total:
            print()  # 换行
    
    def reset(self) -> None:
        """重置进度条"""
        self.current = 0
        self.metrics = {}


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    metrics: Dict[str, float],
    elapsed_time: float
) -> None:
    """
    打印epoch摘要
    
    Args:
        epoch: 当前epoch
        total_epochs: 总epoch数
        metrics: 指标字典
        elapsed_time: 用时(秒)
    """
    print("\n" + "-" * 70)
    print(f"Epoch [{epoch}/{total_epochs}] 完成 | 用时: {elapsed_time:.1f}s")
    print("-" * 70)
    
    # 分组打印指标
    loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k.lower()}
    acc_metrics = {k: v for k, v in metrics.items() if 'accuracy' in k.lower() or 'acc' in k.lower()}
    other_metrics = {k: v for k, v in metrics.items() if k not in loss_metrics and k not in acc_metrics}
    
    if loss_metrics:
        print("损失:", " | ".join([f"{k}: {v:.4f}" for k, v in loss_metrics.items()]))
    if acc_metrics:
        print("准确率:", " | ".join([f"{k}: {v:.4f}" for k, v in acc_metrics.items()]))
    if other_metrics:
        print("其他:", " | ".join([f"{k}: {v:.4f}" for k, v in other_metrics.items()]))
    
    print("-" * 70)


if __name__ == "__main__":
    """
    测试日志模块
    """
    print("=" * 60)
    print("日志模块测试")
    print("=" * 60)
    
    # 测试logger
    print("\n>>> 测试标准日志记录器:")
    logger = setup_logger('test', './logs')
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.debug("这是一条调试日志 (不会显示)")
    
    # 测试TensorBoard记录器
    print("\n>>> 测试TensorBoard记录器:")
    tb_logger = TensorBoardLogger('./runs', 'test_experiment')
    
    # 模拟训练过程
    for epoch in range(5):
        tb_logger.log_scalars({
            'loss/train': 1.0 / (epoch + 1),
            'loss/val': 1.2 / (epoch + 1),
            'accuracy/train': 0.5 + epoch * 0.1,
            'accuracy/val': 0.45 + epoch * 0.1,
        }, epoch)
    
    tb_logger.close()
    
    # 测试进度条
    print("\n>>> 测试进度条:")
    pbar = ProgressBar(10, prefix='训练')
    for i in range(10):
        import time
        time.sleep(0.1)
        pbar.update(1, loss=1.0/(i+1), acc=0.1*(i+1))
    
    # 测试epoch摘要
    print("\n>>> 测试epoch摘要:")
    print_epoch_summary(
        epoch=1,
        total_epochs=100,
        metrics={
            'train_loss': 0.5432,
            'train_accuracy': 0.7823,
            'val_loss': 0.6123,
            'val_accuracy': 0.7456,
            'target_accuracy': 0.6789,
            'f1_score': 0.7234
        },
        elapsed_time=45.6
    )
    
    print("\n测试完成！")
