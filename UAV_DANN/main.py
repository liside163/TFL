# -*- coding: utf-8 -*-
"""
==============================================================================
UAV-DANN 项目入口文件
==============================================================================
功能：提供便捷的命令行接口
- 数据预处理
- 模型训练
- 模型评估

使用方式:
---------
# 数据预处理
python main.py preprocess --config ./config/config.yaml

# 训练模型
python main.py train --config ./config/config.yaml

# 评估模型
python main.py evaluate --config ./config/config.yaml --checkpoint ./checkpoints/best.pth

# 完整流程
python main.py all --config ./config/config.yaml

作者：UAV-DANN项目
日期：2025年
==============================================================================
"""

import os
import sys
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def run_preprocess(args):
    """运行数据预处理"""
    from data.preprocess import DataPreprocessor
    
    print("=" * 70)
    print("UAV-DANN 数据预处理")
    print("=" * 70)
    
    preprocessor = DataPreprocessor(config_path=args.config)
    data_dict = preprocessor.process(save_processed=True)
    
    print("\n数据预处理完成！")


def run_train(args):
    """运行训练"""
    from train import train
    
    train(config_path=args.config, resume_path=args.resume)


def run_evaluate(args):
    """运行评估"""
    from evaluate import evaluate
    
    if args.checkpoint is None:
        # 尝试找到最优模型
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        checkpoint_dir = config['logging']['checkpoint_dir']
        experiment_name = config['logging']['experiment_name']
        
        # 查找best模型
        best_path = None
        for file in os.listdir(checkpoint_dir):
            if 'best' in file and file.endswith('.pth'):
                best_path = os.path.join(checkpoint_dir, file)
                break
        
        if best_path is None:
            print("[错误] 未找到模型检查点，请使用 --checkpoint 指定")
            return
        
        args.checkpoint = best_path
        print(f"自动选择检查点: {args.checkpoint}")
    
    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output
    )


def run_all(args):
    """运行完整流程"""
    print("=" * 70)
    print("UAV-DANN 完整流程")
    print("=" * 70)
    
    # 1. 数据预处理
    print("\n[1/3] 数据预处理...")
    run_preprocess(args)
    
    # 2. 训练
    print("\n[2/3] 模型训练...")
    args.resume = None
    run_train(args)
    
    # 3. 评估
    print("\n[3/3] 模型评估...")
    args.checkpoint = None
    args.output = None
    run_evaluate(args)
    
    print("\n" + "=" * 70)
    print("完整流程执行完成！")
    print("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='UAV-DANN 无人机故障诊断迁移学习',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
---------
# 仅数据预处理
python main.py preprocess --config ./config/config.yaml

# 训练模型
python main.py train --config ./config/config.yaml

# 从检查点恢复训练
python main.py train --config ./config/config.yaml --resume ./checkpoints/epoch50.pth

# 评估模型
python main.py evaluate --config ./config/config.yaml --checkpoint ./checkpoints/best.pth

# 完整流程（预处理+训练+评估）
python main.py all --config ./config/config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # -------------------- preprocess 子命令 --------------------
    parser_preprocess = subparsers.add_parser('preprocess', help='数据预处理')
    parser_preprocess.add_argument(
        '--config',
        type=str,
        default='./config/config.yaml',
        help='配置文件路径'
    )
    
    # -------------------- train 子命令 --------------------
    parser_train = subparsers.add_parser('train', help='训练模型')
    parser_train.add_argument(
        '--config',
        type=str,
        default='./config/config.yaml',
        help='配置文件路径'
    )
    parser_train.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的检查点路径'
    )
    
    # -------------------- evaluate 子命令 --------------------
    parser_evaluate = subparsers.add_parser('evaluate', help='评估模型')
    parser_evaluate.add_argument(
        '--config',
        type=str,
        default='./config/config.yaml',
        help='配置文件路径'
    )
    parser_evaluate.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型检查点路径'
    )
    parser_evaluate.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出目录'
    )
    
    # -------------------- all 子命令 --------------------
    parser_all = subparsers.add_parser('all', help='运行完整流程')
    parser_all.add_argument(
        '--config',
        type=str,
        default='./config/config.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # 获取绝对路径
    if hasattr(args, 'config') and not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    
    # 执行对应命令
    if args.command == 'preprocess':
        run_preprocess(args)
    elif args.command == 'train':
        run_train(args)
    elif args.command == 'evaluate':
        run_evaluate(args)
    elif args.command == 'all':
        run_all(args)


if __name__ == "__main__":
    main()
