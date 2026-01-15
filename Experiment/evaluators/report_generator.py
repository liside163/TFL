"""
自动报告生成器
生成CSV表格、可视化图表、Markdown报告
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
from config import Config


class ReportGenerator:
    """自动报告生成器"""

    def __init__(self, results, config=None):
        """
        参数:
            results: List[dict], 实验结果列表
            config: 配置对象
        """
        self.results = results
        self.config = config or Config()
        self.df = pd.DataFrame(results)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_all_reports(self):
        """生成所有报告"""
        print("\n生成实验报告...")

        # 1. 保存详细CSV
        self.save_detailed_csv()

        # 2. 生成对比表格
        self.generate_comparison_tables()

        # 3. 生成可视化
        self.generate_visualizations()

        # 4. 生成Markdown报告
        self.generate_markdown_report()

        print(f"所有报告已保存到: {self.config.FIG_DIR}")

    def save_detailed_csv(self):
        """保存详细CSV"""
        csv_path = Path(self.config.SAVE_DIR) / 'detailed_results.csv'
        self.df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  详细CSV: {csv_path}")

    def generate_comparison_tables(self):
        """生成对比表格"""
        # 展平嵌套的result字段
        df_flat = self.df.copy()

        # 提取result中的字段到顶层
        if 'result' in df_flat.columns and len(df_flat) > 0:
            result_data = pd.DataFrame(df_flat['result'].tolist())
            df_flat = pd.concat([df_flat.drop('result', axis=1), result_data], axis=1)

        # 方法对比
        if 'f1_macro' in df_flat.columns:
            method_comp = df_flat.groupby(['method', 'architecture'])['f1_macro'].agg(
                ['mean', 'std', 'min', 'max', 'count']
            ).round(4)
            method_comp.to_csv(Path(self.config.SAVE_DIR) / 'method_comparison.csv'))

            # 状态对比
            state_comp = df_flat.groupby('target_state')['f1_macro'].agg(
                ['mean', 'std', 'min', 'max']
            ).round(4)
            state_comp.to_csv(Path(self.config.SAVE_DIR) / 'state_comparison.csv')

    def generate_visualizations(self):
        """生成可视化图表"""
        fig_dir = Path(self.config.FIG_DIR)
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 展平result字段
        df_flat = self.df.copy()
        if 'result' in df_flat.columns and len(df_flat) > 0:
            result_data = pd.DataFrame(df_flat['result'].tolist())
            df_flat = pd.concat([df_flat.drop('result', axis=1), result_data], axis=1)

        # 创建大图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 图1: 方法对比箱线图
        if 'f1_macro' in df_flat.columns:
            sns.boxplot(
                data=df_flat, x='method', y='f1_macro',
                hue='architecture', ax=axes[0, 0]
            )
            axes[0, 0].set_title('迁移方法对比', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_xlabel('方法', fontsize=12)
            axes[0, 0].set_ylabel('F1 Score (Macro)', fontsize=12)
            axes[0, 0].legend(title='架构')

            # 图2: 架构对比
            sns.boxplot(
                data=df_flat, x='architecture', y='f1_macro',
                ax=axes[0, 1]
            )
            axes[0, 1].set_title('模型架构对比', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].set_xlabel('架构', fontsize=12)
            axes[0, 1].set_ylabel('F1 Score (Macro)', fontsize=12)

            # 图3: 目标状态难度
            state_order = sorted(df_flat['target_state'].unique())
            sns.boxplot(
                data=df_flat, x='target_state', y='f1_macro',
                order=state_order, ax=axes[1, 0]
            )
            axes[1, 0].set_title('目标状态迁移难度', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('飞行状态', fontsize=12)
            axes[1, 0].set_ylabel('F1 Score (Macro)', fontsize=12)
            axes[1, 0].set_xticklabels([
                self.config.STATE_NAMES.get(int(x.get_text()), f'State{x.get_text()}')
                for x in axes[1, 0].get_xticklabels()
            ], rotation=15)

            # 图4: 迁移率分布
            if 'transfer_rate' in df_flat.columns:
                sns.boxplot(
                    data=df_flat, x='method', y='transfer_rate',
                    ax=axes[1, 1]
                )
                axes[1, 1].set_title('迁移率分布', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('方法', fontsize=12)
                axes[1, 1].set_ylabel('迁移率 (%)', fontsize=12)
                axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='优秀(80%)')
                axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(fig_dir / 'comparison_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  可视化图表: {fig_dir}")

    def generate_markdown_report(self):
        """生成Markdown报告"""
        # 展平result字段
        df_flat = self.df.copy()
        if 'result' in df_flat.columns and len(df_flat) > 0:
            result_data = pd.DataFrame(df_flat['result'].tolist())
            df_flat = pd.concat([df_flat.drop('result', axis=1), result_data], axis=1)

        # 计算汇总统计
        success_count = len(df_flat[df_flat['status'] == 'success']) if 'status' in df_flat.columns else len(df_flat)
        failed_count = len(df_flat[df_flat['status'] == 'failed']) if 'status' in df_flat.columns else 0

        # 找最佳配置
        if 'f1_macro' in df_flat.columns:
            top5 = df_flat.nlargest(5, 'f1_macro')

            # 方法对比
            method_stats = df_flat.groupby('method')['f1_macro'].agg(['mean', 'std']).round(4)
            best_method = method_stats['mean'].idxmax()
            worst_method = method_stats['mean'].idxmin()

            # 状态难度排序
            state_difficulty = df_flat.groupby('target_state')['f1_macro'].mean().sort_values()
        else:
            top5 = []
            method_stats = None
            best_method = 'N/A'
            worst_method = 'N/A'
            state_difficulty = pd.Series()

        # 生成报告
        report = f"""# RflyMAD 迁移学习实验报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 实验概览

- **总实验数**: {len(df_flat)}
- **成功**: {success_count}
- **失败**: {failed_count}

---

## 最佳结果 (Top 5)

| 排名 | 实验ID | 方法 | 架构 | 目标状态 | F1-Score | 迁移率 |
|------|--------|------|------|----------|----------|--------|
"""

        if len(top5) > 0:
            for idx, (i, row) in enumerate(top5.iterrows(), 1):
                transfer_rate = row.get('transfer_rate', 'N/A')
                if transfer_rate != 'N/A':
                    transfer_rate = f"{transfer_rate:.1f}%"
                report += f"| {idx} | {row['exp_id']} | {row['method']} | {row['architecture']} | {row['target_state_name']} | {row['f1_macro']:.4f} | {transfer_rate} |\n"
        else:
            report += "| - | 暂无数据 | - | - | - | - | - |\n"

        report += f"""

---

## 方法对比

"""

        if method_stats is not None:
            report += "| 方法 | 平均F1 | 标准差 | 最小值 | 最大值 |\n"
            report += "|------|--------|--------|--------|--------|\n"

            for method, stats in method_stats.iterrows():
                report += f"| {method} | {stats['mean']:.4f} | {stats['std']:.4f} |\n"

            report += f"""

### 关键发现

1. **最佳迁移方法**: {best_method}
2. **最差迁移方法**: {worst_method}
3. **目标状态难度排序** (从易到难):
"""

            for state, f1 in state_difficulty.items():
                state_name = self.config.STATE_NAMES.get(state, f'State{state}')
                report += f"   - {state_name}: {f1:.4f}\n"

        # 架构对比
        if 'f1_macro' in df_flat.columns:
            arch_stats = df_flat.groupby('architecture')['f1_macro'].agg(['mean', 'std']).round(4)

            report += f"""

---

## 架构对比

| 架构 | 平均F1 | 标准差 |
|------|--------|--------|
"""
            for arch, stats in arch_stats.iterrows():
                report += f"| {arch} | {stats['mean']:.4f} | {stats['std']:.4f} |\n"

        report += """

---

## 结论与建议

### 主要结论

"""

        # 自动分析结论
        if method_stats is not None:
            if best_method == 'mmd':
                report += "- MMD方法表现最佳，说明基于统计矩的域适应对飞行状态迁移有效\n"
            elif best_method == 'pretrain':
                report += "- 预训练+微调方法表现最佳，说明冻结特征提取器是有效的迁移策略\n"
            elif best_method == 'dann':
                report += "- DANN方法表现最佳，说明对抗学习能有效学习域不变特征\n"

            if 'f1_macro' in df_flat.columns:
                arch_stats = df_flat.groupby('architecture')['f1_macro'].mean()
                if arch_stats.get('cnn_lstm', 0) > arch_stats.get('cnn', 0):
                    report += "- 混合架构优于纯CNN，说明融合局部和全局特征有助于迁移\n"

        report += """

### 改进建议

1. 对于难以迁移的目标状态，考虑增加目标域微调数据量
2. 尝试集成学习，组合多种迁移方法的预测结果
3. 进一步优化超参数，特别是域适应损失权重λ

---

*报告由 AutoExperimentRunner 自动生成*
"""

        # 保存报告
        report_path = Path(self.config.SAVE_DIR) / 'auto_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"  Markdown报告: {report_path}")
