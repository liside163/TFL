# =====================================================================
# 故障分类器模块
# 全连接层实现7分类
# =====================================================================
#
# 维度变换:
# 输入: [B, 256] (特征提取器输出)
# 隐藏层: [B, 128]
# 输出: [B, 7] (7类故障概率)
# =====================================================================

import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import CLASSIFIER_CONFIG


class FaultClassifier(nn.Module):
    """
    故障分类器
    
    结构: FC -> ReLU -> Dropout -> FC
    
    维度变换:
    输入: [B, 256]
    隐藏层: [B, 128]
    输出: [B, 7]
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        if config is None:
            config = CLASSIFIER_CONFIG
        
        self.classifier = nn.Sequential(
            # 第一层全连接: 256 -> 128
            nn.Linear(config["input_size"], config["hidden_size"]),
            nn.ReLU(inplace=True),
            nn.Dropout(config["dropout"]),
            
            # 第二层全连接: 128 -> 7
            nn.Linear(config["hidden_size"], config["num_classes"])
        )
        
        self.num_classes = config["num_classes"]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, input_size] = [B, 256]
        
        Returns:
            [B, num_classes] = [B, 7] (logits，未经softmax)
        """
        return self.classifier(x)


if __name__ == "__main__":
    # 测试代码
    print("测试故障分类器...")
    
    batch_size = 4
    input_size = 256
    
    x = torch.randn(batch_size, input_size)
    print(f"输入形状: {x.shape}")
    
    classifier = FaultClassifier()
    output = classifier(x)
    print(f"输出形状: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"参数量: {total_params:,}")
