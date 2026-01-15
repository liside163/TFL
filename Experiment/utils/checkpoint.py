"""
模型检查点管理
"""

import torch
from pathlib import Path


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, save_dir, exp_id):
        """
        参数:
            save_dir: 保存目录
            exp_id: 实验ID
        """
        self.save_dir = Path(save_dir) / exp_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = 0

    def save(self, model, optimizer, epoch, score, filename='checkpoint.pth'):
        """
        保存检查点

        参数:
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            score: 当前指标分数
            filename: 保存文件名
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
        }
        torch.save(checkpoint, self.save_dir / filename)

        # 如果是最佳模型，额外保存
        if score > self.best_score:
            self.best_score = score
            torch.save(checkpoint, self.save_dir / 'best_model.pth')

    def load(self, model, optimizer=None, filename='best_model.pth'):
        """
        加载检查点

        参数:
            model: 模型
            optimizer: 优化器 (可选)
            filename: 加载文件名

        返回:
            epoch: 检查点保存的epoch
            score: 检查点保存的分数
        """
        checkpoint = torch.load(self.save_dir / filename)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['score']

    def get_checkpoint_path(self, filename='best_model.pth'):
        """获取检查点文件路径"""
        return self.save_dir / filename
