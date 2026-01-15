"""
预训练+微调方法
在源域预训练，冻结特征提取器，微调分类头
"""

import torch
import torch.nn as nn
import torch.optim as optim
from methods.baseline import BaselineMethod
from config import Config


class PretrainFinetuneMethod:
    """预训练+微调方法"""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else 'cpu')

        # 源域训练器
        self.source_trainer = BaselineMethod(model, config)

        # 微调优化器 (只优化分类头)
        self.finetune_optimizer = None

        # 记录源域F1
        self.source_f1 = None

    def pretrain(self, source_loader, val_loader=None):
        """
        阶段1: 在源域上预训练

        参数:
            source_loader: 源域数据加载器
            val_loader: 验证数据加载器

        返回:
            source_f1: 源域F1分数
        """
        print("\n" + "="*60)
        print("阶段1: 源域预训练")
        print("="*60)

        history, best_metrics = self.source_trainer.train(
            source_loader,
            val_loader,
            num_epochs=self.config.NUM_EPOCHS
        )

        self.source_f1 = best_metrics['f1_macro']

        print(f"\n源域预训练完成!")
        print(f"源域F1: {self.source_f1:.4f}")

        return history, best_metrics

    def finetune(self, target_loader, val_loader=None):
        """
        阶段2: 在目标域上微调

        参数:
            target_loader: 目标域微调数据加载器
            val_loader: 验证数据加载器

        返回:
            history: 微调历史
            target_f1: 目标域F1分数
        """
        print("\n" + "="*60)
        print("阶段2: 目标域微调")
        print("="*60)

        # 冻结特征提取器
        print("冻结特征提取器...")

        # 冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻分类头 (最后两层)
        if hasattr(self.model, 'fc2'):
            self.model.fc2.weight.requires_grad = True
            self.model.fc2.bias.requires_grad = True
            self.model.fc1.weight.requires_grad = True
            self.model.fc1.bias.requires_grad = True

        # 创建微调优化器 (只优化可训练参数)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.finetune_optimizer = optim.Adam(
            trainable_params,
            lr=self.config.FINETUNE_LR,  # 更小的学习率
            weight_decay=self.config.WEIGHT_DECAY
        )

        print(f"可训练参数: {sum(p.numel() for p in trainable_params)}")

        # 微调训练
        criterion = nn.CrossEntropyLoss()
        history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
        best_f1 = 0

        for epoch in range(self.config.FINETUNE_EPOCHS):
            self.model.train()
            total_loss = 0
            all_preds = []
            all_labels = []

            for data, labels in target_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                self.finetune_optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                self.finetune_optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # 计算指标
            from sklearn.metrics import f1_score
            train_f1 = f1_score(all_labels, all_preds, average='macro')
            avg_loss = total_loss / len(target_loader)

            history['train_loss'].append(avg_loss)
            history['train_f1'].append(train_f1)

            print(f"Epoch {epoch+1}/{self.config.FINETUNE_EPOCHS}: "
                  f"Loss={avg_loss:.4f}, F1={train_f1:.4f}")

            if val_loader:
                # 验证
                self.model.eval()
                with torch.no_grad():
                    val_preds = []
                    val_labels = []
                    val_loss = 0
                    for data, labels in val_loader:
                        data, labels = data.to(self.device), labels.to(self.device)
                        outputs = self.model(data)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())

                    val_f1 = f1_score(val_labels, val_preds, average='macro')
                    history['val_f1'].append(val_f1)
                    history['val_loss'].append(val_loss / len(val_loader))

                    if val_f1 > best_f1:
                        best_f1 = val_f1

                    print(f"  Val F1: {val_f1:.4f}")

        target_f1 = best_f1 if val_loader else train_f1

        print(f"\n微调完成!")
        print(f"目标域F1: {target_f1:.4f}")

        # 计算迁移率
        transfer_rate = (target_f1 / self.source_f1) * 100
        print(f"迁移率: {transfer_rate:.2f}%")

        return history, {'f1_macro': target_f1, 'transfer_rate': transfer_rate}

    def run_full_pipeline(self, source_loader, target_train_loader, target_test_loader):
        """
        运行完整的预训练+微调流程

        参数:
            source_loader: 源域训练数据
            target_train_loader: 目标域微调数据
            target_test_loader: 目标域测试数据

        返回:
            results: 完整结果字典
        """
        # 阶段1: 预训练
        source_history, source_metrics = self.pretrain(source_loader)

        # 阶段2: 微调
        finetune_history, target_metrics = self.finetune(target_train_loader, target_test_loader)

        results = {
            'source_f1': source_metrics['f1_macro'],
            'target_f1': target_metrics['f1_macro'],
            'transfer_rate': target_metrics['transfer_rate'],
            'source_history': source_history,
            'finetune_history': finetune_history
        }

        return results
