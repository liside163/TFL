"""
全自动批量迁移学习实验主脚本
"""

import argparse
import os
from experiments.experiment_matrix import generate_experiment_matrix
from experiments.auto_experiment_runner import AutoExperimentRunner
from evaluators.report_generator import ReportGenerator
from config import Config


def main():
    parser = argparse.ArgumentParser(description='RflyMAD 迁移学习自动化实验系统')
    parser.add_argument('--mode', choices=['all', 'quick', 'single', 'report'],
                       default='all', help='运行模式')
    parser.add_argument('--exp-id', type=str, help='单个实验ID')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复')

    args = parser.parse_args()

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 加载配置
    config = Config()
    print(f"\n{'='*60}")
    print("RflyMAD 跨飞行状态迁移学习自动化实验系统")
    print(f"{'='*60}")
    print(f"数据集路径: {config.HIL_DATA_DIR}")
    print(f"结果保存: {config.SAVE_DIR}")
    print(f"运行模式: {args.mode}")

    if args.mode == 'all':
        # 运行所有实验
        print("\n生成实验矩阵...")
        experiment_matrix = generate_experiment_matrix(config)

        print(f"\n开始批量实验 ({len(experiment_matrix)} 组)...")
        runner = AutoExperimentRunner(config)
        runner.run_all_experiments(experiment_matrix)

        # 生成报告
        if runner.results_summary:
            print("\n生成实验报告...")
            generator = ReportGenerator(runner.results_summary, config)
            generator.generate_all_reports()

    elif args.mode == 'quick':
        # 快速测试 (少量实验)
        print("\n快速测试模式...")
        from experiments.experiment_matrix import generate_quick_test_matrix
        experiment_matrix = generate_quick_test_matrix()

        runner = AutoExperimentRunner(config)
        runner.run_all_experiments(experiment_matrix)

        if runner.results_summary:
            generator = ReportGenerator(runner.results_summary, config)
            generator.generate_all_reports()

    elif args.mode == 'report':
        # 仅生成报告
        print("\n从已有结果生成报告...")
        import json
        from pathlib import Path

        results_file = Path(config.SAVE_DIR) / 'results_summary.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            generator = ReportGenerator(results, config)
            generator.generate_all_reports()
        else:
            print(f"错误: 未找到结果文件 {results_file}")

    print("\n所有任务完成!")


if __name__ == '__main__':
    main()
