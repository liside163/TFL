"""
Baseline方法 - 无迁移对照组
直接在目标域上训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config


class BaselineMethod:
    """Baseline方法: 直接在目标域训练，不使用源域数据"""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else 'cpu')

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # 学习率调度器
        if self.config.LR_SCHEDULER == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **self.config.LR_SCHEDULER_PARAMS
            )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': []
        }

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc="Training")):
            data, labels = data.to(self.device), labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        return avg_loss, all_preds, all_labels

    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        return avg_loss, all_preds, all_labels

    def compute_metrics(self, preds, labels):
        """计算评估指标"""
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        accuracy = accuracy_score(labels, preds)

        return {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }

    def train(self, train_loader, val_loader=None, num_epochs=None):
        """
        完整训练流程

        参数:
            train_loader: 目标域训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数

        返回:
            history: 训练历史
            best_metrics: 最佳指标
        """
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        best_f1 = 0
        patience_counter = 0

        print(f"\n开始Baseline训练 (目标域直接训练)")
        print(f"Epochs: {num_epochs}, Device: {self.device}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # 训练
            train_loss, train_preds, train_labels = self.train_epoch(train_loader)
            train_metrics = self.compute_metrics(train_preds, train_labels)

            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_metrics['f1_macro'])

            print(f"Train Loss: {train_loss:.4f}, F1: {train_metrics['f1_macro']:.4f}")

            # 验证
            if val_loader:
                val_loss, val_preds, val_labels = self.validate(val_loader)
                val_metrics = self.compute_metrics(val_preds, val_labels)

                self.history['val_loss'].append(val_loss)
                self.history['val_f1'].append(val_metrics['f1_macro'])

                print(f"Val Loss: {val_loss:.4f}, F1: {val_metrics['f1_macro']:.4f}")

                # 学习率调度
                self.scheduler.step(val_metrics['f1_macro'])

                # 早停
                if val_metrics['f1_macro'] > best_f1:
                    best_f1 = val_metrics['f1_macro']
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    print(f"  保存最佳模型 (F1={best_f1:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.EARLY_STOP_PATIENCE:
                        print(f"  早停: {self.config.EARLY_STOP_PATIENCE}轮无改善")
                        break

        # 加载最佳模型
        if val_loader:
            self.model.load_state_dict(torch.load('best_model.pth'))
            final_metrics = self.compute_metrics(val_preds, val_labels)
        else:
            final_metrics = train_metrics

        print("\n训练完成!")
        print(f"最佳F1: {best_f1:.4f}")

        return self.history, final_metrics
