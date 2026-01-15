"""
MMD (Maximum Mean Discrepancy) 域适应方法
在再生核希尔伯特空间中最小化源域和目标域的分布距离
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from config import Config


class MMDLoss(nn.Module):
    """MMD损失函数"""

    def __init__(self, sigmas=[0.1, 1, 10, 100]):
        super(MMDLoss, self).__init__()
        self.sigmas = sigmas

    def gaussian_kernel(self, x, y, sigma):
        """
        高斯核计算

        参数:
            x: [n, d] 源域特征
            y: [m, d] 目标域特征
            sigma: 高斯核带宽

        返回:
            核矩阵: [n, m]
        """
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        x_norm = torch.sum(x**2, dim=1, keepdim=True)  # [n, 1]
        y_norm = torch.sum(y**2, dim=1, keepdim=True)  # [m, 1]
        dist = x_norm + y_norm.T - 2 * torch.mm(x, y.T)  # [n, m]
        return torch.exp(-dist / (2 * sigma**2))

    def forward(self, source_features, target_features):
        """
        计算MMD^2距离

        参数:
            source_features: [n_s, feature_dim] 源域特征
            target_features: [n_t, feature_dim] 目标域特征

        返回:
            mmd_loss: 标量，MMD距离的平方
        """
        n_s = source_features.shape[0]
        n_t = target_features.shape[0]

        # 多核MMD (组合多个带宽的高斯核)
        mmd_squared = 0

        for sigma in self.sigmas:
            # 源域-源域核矩阵
            k_ss = self.gaussian_kernel(source_features, source_features, sigma)
            # 目标域-目标域核矩阵
            k_tt = self.gaussian_kernel(target_features, target_features, sigma)
            # 源域-目标域核矩阵
            k_st = self.gaussian_kernel(source_features, target_features, sigma)

            # MMD^2 = E[k(x_s, x_s')] + E[k(x_t, x_t')] - 2E[k(x_s, x_t)]
            mmd_squared += (
                k_ss.sum() / (n_s * n_s) +
                k_tt.sum() / (n_t * n_t) -
                2 * k_st.sum() / (n_s * n_t)
            )

        return torch.clamp(mmd_squared, min=0.0)  # 确保非负


class MMDAdapter:
    """MMD域适应方法"""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else 'cpu')

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.mmd_loss = MMDLoss(sigmas=self.config.MMD_KERNEL_SIGMAS)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

    def train_epoch(self, source_loader, target_loader, lambda_mmd=1.0):
        """
        训练一个epoch

        参数:
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器
            lambda_mmd: MMD损失权重

        返回:
            avg_loss: 平均总损失
            avg_cls_loss: 平均分类损失
            avg_mmd_loss: 平均MMD损失
        """
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_mmd_loss = 0

        # 创建目标域迭代器 (循环)
        target_iter = iter(target_loader)

        for source_data, source_labels in tqdm(source_loader, desc="MMD Training"):
            # 获取目标域batch
            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)

            source_data = source_data.to(self.device)
            source_labels = source_labels.to(self.device)
            target_data = target_data.to(self.device)

            self.optimizer.zero_grad()

            # 源域前向传播
            source_logits, source_features = self.model(source_data, return_features=True)
            cls_loss = self.criterion(source_logits, source_labels)

            # 目标域前向传播 (只提取特征)
            with torch.no_grad():
                _, target_features = self.model(target_data, return_features=True)

            # 计算MMD损失
            mmd_loss = self.mmd_loss(source_features, target_features)

            # 总损失 = 分类损失 + lambda * MMD损失
            loss = cls_loss + lambda_mmd * mmd_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_mmd_loss += mmd_loss.item()

        n = len(source_loader)
        return total_loss / n, total_cls_loss / n, total_mmd_loss / n

    def train(self, source_loader, target_train_loader, target_test_loader,
              lambda_mmd=None, num_epochs=None):
        """
        完整训练流程

        参数:
            source_loader: 源域数据
            target_train_loader: 目标域微调数据
            target_test_loader: 目标域测试数据
            lambda_mmd: MMD权重 (默认从config读取)
            num_epochs: 训练轮数

        返回:
            history: 训练历史
            best_metrics: 最佳指标
        """
        lambda_mmd = lambda_mmd or self.config.MMD_LAMBDA
        num_epochs = num_epochs or self.config.NUM_EPOCHS

        print(f"\nMMD域适应训练")
        print(f"Lambda MMD: {lambda_mmd}, Epochs: {num_epochs}")

        history = {
            'train_loss': [],
            'cls_loss': [],
            'mmd_loss': [],
            'val_f1': []
        }

        best_f1 = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # 训练
            train_loss, cls_loss, mmd_loss = self.train_epoch(
                source_loader, target_train_loader, lambda_mmd
            )

            history['train_loss'].append(train_loss)
            history['cls_loss'].append(cls_loss)
            history['mmd_loss'].append(mmd_loss)

            print(f"Loss: {train_loss:.4f} (Cls: {cls_loss:.4f}, MMD: {mmd_loss:.4f})")

            # 验证
            if target_test_loader:
                val_f1 = self.evaluate(target_test_loader)
                history['val_f1'].append(val_f1)
                print(f"Val F1: {val_f1:.4f}")

                # 早停
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_mmd_model.pth')
                    print(f"  保存最佳模型")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.EARLY_STOP_PATIENCE:
                        print(f"  早停")
                        break

        # 加载最佳模型
        if target_test_loader:
            self.model.load_state_dict(torch.load('best_mmd_model.pth'))

        print(f"\n训练完成! 最佳F1: {best_f1:.4f}")

        return history, {'f1_macro': best_f1}

    def evaluate(self, test_loader):
        """
        评估模型

        参数:
            test_loader: 测试数据加载器

        返回:
            f1_macro: 宏平均F1分数
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                preds = outputs.argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        from sklearn.metrics import f1_score
        return f1_score(all_labels, all_preds, average='macro')
