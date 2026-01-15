"""
实验矩阵生成器
生成所有实验配置的组合
"""

import itertools
import json
from pathlib import Path
from config import Config


def generate_experiment_matrix(config=None):
    """
    生成完整的实验矩阵

    参数:
        config: 配置对象

    返回:
        experiment_matrix: List[dict], 每个元素是一个实验配置
    """
    config = config or Config()

    # 定义实验维度
    methods = ['baseline', 'pretrain', 'mmd']  # 5种迁移方法
    architectures = ['cnn', 'lstm', 'cnn_lstm']  # 3种架构
    target_states = config.TARGET_STATES  # 5种目标状态 [1, 2, 3, 4, 5]

    # 生成所有组合
    experiment_matrix = []
    exp_id = 1

    for method, arch, target_state in itertools.product(methods, architectures, target_states):
        config_dict = {
            'exp_id': f'E{exp_id:03d}',
            'method': method,
            'architecture': arch,
            'target_state': target_state,
            'target_state_name': config.STATE_NAMES[target_state],
            'hyperparams': {
                'learning_rate': config.LEARNING_RATE,
                'batch_size': config.BATCH_SIZE,
                'dropout': 0.2,
            }
        }

        # 特定方法的超参数
        if method == 'mmd':
            config_dict['hyperparams']['lambda_mmd'] = config.MMD_LAMBDA
        elif method == 'dann':
            config_dict['hyperparams']['lambda_dann'] = config.DANN_LAMBDA_MAX

        experiment_matrix.append(config_dict)
        exp_id += 1

    print(f"生成实验矩阵: 共 {len(experiment_matrix)} 组实验")
    print(f"  方法: {len(methods)}")
    print(f"  架构: {len(architectures)}")
    print(f"  目标状态: {len(target_states)}")

    return experiment_matrix


def generate_quick_test_matrix():
    """生成快速测试矩阵 (少量实验)"""
    return [
        {
            'exp_id': 'E001',
            'method': 'baseline',
            'architecture': 'cnn',
            'target_state': 1,
            'hyperparams': {}
        },
        {
            'exp_id': 'E002',
            'method': 'pretrain',
            'architecture': 'cnn',
            'target_state': 1,
            'hyperparams': {}
        },
    ]


if __name__ == '__main__':
    matrix = generate_experiment_matrix()

    # 保存到JSON
    save_dir = Path(Config().SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'experiment_matrix.json', 'w', encoding='utf-8') as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)

    print(f"\n实验矩阵已保存到: {save_dir / 'experiment_matrix.json'}")

    # 打印前5个
    print("\n前5个实验:")
    for exp in matrix[:5]:
        print(f"  {exp}")
