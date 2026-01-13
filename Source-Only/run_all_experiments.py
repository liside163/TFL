# =====================================================================
# 批量运行所有飞行状态的Source-Only实验
# =====================================================================

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))
from config import AVAILABLE_FLIGHT_STATES, RESULT_DIR

# 使用配置中的可用飞行状态 (REAL数据中dece没有数据，已排除)
FLIGHT_STATES = AVAILABLE_FLIGHT_STATES

def run_experiment(flight_state: str, test_run: bool = False):
    """运行单个飞行状态的实验"""
    print(f"\n{'='*60}")
    print(f"开始实验: {flight_state}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "train_source_only.py",
        "--flight_state", flight_state,
        "--tsne"
    ]
    
    if test_run:
        cmd.append("--test_run")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"警告: {flight_state} 实验失败")
        return False
    return True


def collect_all_results():
    """
    收集所有实验结果JSON文件并生成汇总表格
    
    Returns:
        所有实验结果的列表
    """
    results_list = []
    
    # 遍历results目录下的所有子目录
    if not RESULT_DIR.exists():
        print("结果目录不存在")
        return results_list
    
    for exp_dir in RESULT_DIR.iterdir():
        if exp_dir.is_dir():
            json_file = exp_dir / "experiment_result.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        results_list.append(result)
                except Exception as e:
                    print(f"警告: 无法读取 {json_file}: {e}")
    
    return results_list


def generate_summary_table(results_list: list, save_path: Path):
    """
    生成实验结果汇总表格
    
    Args:
        results_list: 所有实验结果的列表
        save_path: 保存路径（不含扩展名）
    """
    if not results_list:
        print("没有找到实验结果")
        return
    
    # 按飞行状态和时间排序
    results_list.sort(key=lambda x: (x.get("飞行状态", ""), x.get("实验时间", "")))
    
    # 转换为DataFrame
    table_data = []
    for r in results_list:
        table_data.append({
            "实验时间": r["实验时间"],
            "飞行状态": r["飞行状态"],
            "方法": r["方法"],
            "源域验证ACC": float(r["源域结果"]["验证准确率"]),
            "源域验证F1": float(r["源域结果"]["验证F1"]),
            "目标域ACC": float(r["目标域结果"]["准确率"]),
            "目标域F1": float(r["目标域结果"]["F1"]),
            "性能下降": float(r['源域结果']['验证准确率']) - float(r['目标域结果']['准确率']),
            "备注": "基线实验"
        })
    
    df = pd.DataFrame(table_data)
    
    # 保存为CSV
    csv_path = save_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总表格已保存 (CSV): {csv_path}")
    
    # 保存为Markdown
    md_path = save_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Source-Only 实验结果汇总\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"总实验数: {len(results_list)}\n\n")
        
        # 按飞行状态分组统计
        f.write("## 按飞行状态统计\n\n")
        state_summary = df.groupby("飞行状态").agg({
            "源域验证ACC": "mean",
            "目标域ACC": "mean",
            "性能下降": "mean"
        }).round(4)
        f.write(state_summary.to_markdown())
        f.write("\n\n")
        
        # 完整结果表格
        f.write("## 完整实验结果\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    
    print(f"汇总表格已保存 (Markdown): {md_path}")
    
    # 打印到控制台
    print(f"\n{'='*80}")
    print("实验结果汇总")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")


def main():
    """运行所有实验并生成汇总报告"""
    import argparse
    
    parser = argparse.ArgumentParser(description="批量运行Source-Only实验")
    parser.add_argument(
        "--states",
        nargs="+",
        default=FLIGHT_STATES,
        help="要运行的飞行状态列表"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="测试模式"
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="只生成汇总表格，不运行实验"
    )
    
    args = parser.parse_args()
    
    if not args.summary_only:
        # 运行实验
        print(f"\n{'#'*60}")
        print(f"# 批量运行 Source-Only 实验")
        print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# 飞行状态: {args.states}")
        print(f"{'#'*60}")
        
        results = {}
        for state in args.states:
            success = run_experiment(state, args.test_run)
            results[state] = "成功" if success else "失败"
        
        # 打印运行汇总
        print(f"\n{'='*60}")
        print("实验运行汇总")
        print(f"{'='*60}")
        for state, status in results.items():
            print(f"  {state}: {status}")
        print(f"{'='*60}\n")
    
    # 收集并生成汇总表格
    print("\n正在收集所有实验结果...")
    all_results = collect_all_results()
    
    if all_results:
        print(f"找到 {len(all_results)} 个实验结果")
        summary_path = RESULT_DIR / "all_experiments_summary"
        generate_summary_table(all_results, summary_path)
    else:
        print("未找到任何实验结果")
    
    print(f"\n{'='*60}")
    print("全部完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
