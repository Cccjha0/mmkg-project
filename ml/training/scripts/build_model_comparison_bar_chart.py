"""
从各模型的 best_summary.csv 读取 MRR，生成五模型对比柱状图。
用法：
  python ml/training/scripts/build_model_comparison_bar_chart.py [--plots_dir ml/artifacts/plots] [--output model_comparison_mrr.png]
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 模型目录名 -> 显示名称
MODEL_DISPLAY_NAMES = {
    "text-only": "Text-ComplEx",
    "text-rgcn": "Text-RGCN",
    "openbg_img_early": "Early Fusion",
    "gate-only": "Gate Fusion",
    "gate+residual": "Gate+Residual",
}

# 柱状图顺序（按此顺序绘制）
MODEL_ORDER = [
    "text-only",
    "text-rgcn",
    "openbg_img_early",
    "gate-only",
    "gate+residual",
]


def parse_mean_std(value_str: str) -> tuple[float, float]:
    """解析 '0.2173 +/- 0.0074' 格式，返回 (mean, std)"""
    match = re.match(r"([\d.]+)\s*\+\/-\s*([\d.]+)", str(value_str).strip())
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0


def load_mrr_from_summary(csv_path: Path) -> tuple[float, float]:
    """从 best_summary.csv 读取 mean+/-std 行的 best_mrr"""
    with csv_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 2:
        return 0.0, 0.0
    header = lines[0].strip().split(",")
    if "best_mrr" not in header:
        return 0.0, 0.0
    idx = header.index("best_mrr")
    for line in lines[1:]:
        row = line.strip().split(",")
        if len(row) > idx and "mean" in row[0].lower():
            return parse_mean_std(row[idx])
    return 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description="Build model comparison bar chart from best_summary.csv")
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="ml/artifacts/plots",
        help="Directory containing model subdirs (text-only, gate-only, etc.) with best_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/artifacts/plots/model_comparison_mrr.png",
        help="Output path for the bar chart",
    )
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    models = []
    mrr_mean = []
    mrr_std = []

    for model_key in MODEL_ORDER:
        csv_path = plots_dir / model_key / "best_summary.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found, skipping {model_key}")
            continue
        mean, std = load_mrr_from_summary(csv_path)
        models.append(MODEL_DISPLAY_NAMES.get(model_key, model_key))
        mrr_mean.append(mean)
        mrr_std.append(std)

    if not models:
        raise FileNotFoundError(f"No best_summary.csv found under {plots_dir}")

    x = np.arange(len(models))
    plt.figure(figsize=(8, 5))
    plt.bar(x, mrr_mean, yerr=mrr_std, capsize=5)
    plt.xticks(x, models)
    plt.ylabel("MRR")
    plt.title("Model Comparison on OpenBG-IMG")
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
