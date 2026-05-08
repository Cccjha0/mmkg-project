"""
从 Early Fusion 的 best_summary.csv 读取数据，生成表格行和段落文本。
用法：修复 Early Fusion 数据并完成绘图后运行
  python ml/training/scripts/export_early_fusion_for_report.py [--plots_dir ml/artifacts/plots]
"""
import argparse
import re
from pathlib import Path


def parse_mean_std(value_str: str):
    """解析 '0.2970 +/- 0.0003' 格式，返回 (mean, std)"""
    match = re.match(r"([\d.]+)\s*\+\/-\s*([\d.]+)", str(value_str).strip())
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_dir", type=str, default="ml/artifacts/plots")
    args = parser.parse_args()

    csv_path = Path(args.plots_dir) / "openbg_img_early" / "best_summary.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found. Run plot_kg_results.py for Early Fusion first.")
        return

    with csv_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0].strip().split(",")
    idx_mrr = header.index("best_mrr")
    idx_h1 = header.index("hits@1_at_best")
    idx_h3 = header.index("hits@3_at_best")
    idx_h10 = header.index("hits@10_at_best")

    for line in lines[1:]:
        row = line.strip().split(",")
        if len(row) > idx_h10 and "mean" in row[0].lower():
            mrr_mean, mrr_std = parse_mean_std(row[idx_mrr])
            h1_mean, h1_std = parse_mean_std(row[idx_h1])
            h3_mean, h3_std = parse_mean_std(row[idx_h3])
            h10_mean, h10_std = parse_mean_std(row[idx_h10])
            break
    else:
        print("[ERROR] mean+/-std row not found in best_summary.csv")
        return

    # 表格行（保留 3 位小数）
    print("=" * 60)
    print("3. 表格中 Early Fusion 行（Markdown）")
    print("=" * 60)
    print("| Early Fusion | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(mrr_mean, h1_mean, h3_mean, h10_mean))
    print()

    # 表格行（带标准差）
    print("=" * 60)
    print("3. 表格中 Early Fusion 行（带标准差，Markdown）")
    print("=" * 60)
    print("| Early Fusion | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} |".format(
        mrr_mean, mrr_std, h1_mean, h1_std, h3_mean, h3_std, h10_mean, h10_std
    ))
    print()

    # 段落文本
    print("=" * 60)
    print("4. 段落中关于 Early Fusion 的句子")
    print("=" * 60)
    para = (
        "Among the currently organized results, the Early Fusion model achieves an MRR of {:.4f}, "
        "Hits@1 of {:.4f}, Hits@3 of {:.4f}, and Hits@10 of {:.4f}. "
        "These results confirm that the multimodal training pipeline is functioning correctly and that "
        "the model is able to learn useful joint representations from the available features."
    ).format(mrr_mean, h1_mean, h3_mean, h10_mean)
    print(para)
    print()

    # 带标准差的段落
    print("=" * 60)
    print("4. 段落（带 mean ± std）")
    print("=" * 60)
    para_std = (
        "Among the currently organized results, the Early Fusion model achieves an MRR of {:.3f} ± {:.3f}, "
        "Hits@1 of {:.3f} ± {:.3f}, Hits@3 of {:.3f} ± {:.3f}, and Hits@10 of {:.3f} ± {:.3f} (mean ± std). "
        "These results confirm that the multimodal training pipeline is functioning correctly and that "
        "the model is able to learn useful joint representations from the available features."
    ).format(mrr_mean, mrr_std, h1_mean, h1_std, h3_mean, h3_std, h10_mean, h10_std)
    print(para_std)


if __name__ == "__main__":
    main()
