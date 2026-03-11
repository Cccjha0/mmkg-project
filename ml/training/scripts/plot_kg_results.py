from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def set_paper_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def load_csvs(input_dir: Path):
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "epoch" not in df.columns:
            raise ValueError(f"'epoch' column not found in {csv_file.name}")
        df = df.sort_values("epoch").reset_index(drop=True)
        dfs.append((csv_file.stem, df))
    return dfs


def align_by_epoch(dfs):
    """
    Inner join on epoch to keep only epochs shared by all seeds.
    """
    base = dfs[0][1][["epoch"]].copy()
    for _, df in dfs[1:]:
        base = base.merge(df[["epoch"]], on="epoch", how="inner")
    common_epochs = base["epoch"].tolist()

    aligned = []
    for name, df in dfs:
        sub = df[df["epoch"].isin(common_epochs)].sort_values("epoch").reset_index(drop=True)
        aligned.append((name, sub))
    return aligned


def compute_mean_std(aligned_dfs, metric):
    epochs = aligned_dfs[0][1]["epoch"].values
    values = []

    for name, df in aligned_dfs:
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in {name}.csv")
        values.append(df[metric].values)

    values = np.vstack(values)  # [num_seeds, num_epochs]
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    return epochs, mean, std, values


def plot_single_metric_mean_std(aligned_dfs, metric, ylabel, title, output_path):
    epochs, mean, std, _ = compute_mean_std(aligned_dfs, metric)

    plt.figure()
    plt.plot(epochs, mean, label=f"{metric} mean")
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.2, label="± std")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_seed_comparison(aligned_dfs, metric, ylabel, title, output_path):
    plt.figure()
    for name, df in aligned_dfs:
        if metric not in df.columns:
            continue
        plt.plot(df["epoch"], df[metric], label=name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_hits_mean_std(aligned_dfs, output_path):
    hit_metrics = ["hits@1", "hits@3", "hits@10"]
    plt.figure()

    for metric in hit_metrics:
        epochs, mean, std, _ = compute_mean_std(aligned_dfs, metric)
        plt.plot(epochs, mean, label=metric)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Epoch")
    plt.ylabel("Hits")
    plt.title("Hits@K vs Epoch (Mean ± Std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_gate_mean(aligned_dfs, output_path):
    metrics = ["g_mean_all", "g_mean_img", "g_mean_noimg"]
    plt.figure()

    for metric in metrics:
        epochs, mean, std, _ = compute_mean_std(aligned_dfs, metric)
        plt.plot(epochs, mean, label=metric)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Epoch")
    plt.ylabel("Gate Mean")
    plt.title("Gate Mean Analysis (Mean ± Std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_gate_std(aligned_dfs, output_path):
    metrics = ["g_std_all", "g_std_img", "g_std_noimg"]
    plt.figure()

    for metric in metrics:
        epochs, mean, std, _ = compute_mean_std(aligned_dfs, metric)
        plt.plot(epochs, mean, label=metric)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Epoch")
    plt.ylabel("Gate Std")
    plt.title("Gate Std Analysis (Mean ± Std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_image_fraction(aligned_dfs, output_path):
    metric = "g_frac_img_in_sample"
    epochs, mean, std, _ = compute_mean_std(aligned_dfs, metric)

    plt.figure()
    plt.plot(epochs, mean, label=metric)
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.2, label="± std")
    plt.xlabel("Epoch")
    plt.ylabel("Fraction")
    plt.title("Fraction of Image-Available Entities in Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def summarize_best(aligned_dfs, out_csv):
    rows = []
    for name, df in aligned_dfs:
        best_idx = df["mrr"].idxmax()
        row = {
            "seed": name,
            "best_epoch": int(df.loc[best_idx, "epoch"]),
            "best_mrr": float(df.loc[best_idx, "mrr"]),
            "hits@1_at_best": float(df.loc[best_idx, "hits@1"]),
            "hits@3_at_best": float(df.loc[best_idx, "hits@3"]),
            "hits@10_at_best": float(df.loc[best_idx, "hits@10"]),
            "loss_at_best": float(df.loc[best_idx, "avg_loss"]),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    agg_row = {
        "seed": "mean±std",
        "best_epoch": np.nan,
        "best_mrr": f"{summary_df['best_mrr'].mean():.4f} ± {summary_df['best_mrr'].std(ddof=0):.4f}",
        "hits@1_at_best": f"{summary_df['hits@1_at_best'].mean():.4f} ± {summary_df['hits@1_at_best'].std(ddof=0):.4f}",
        "hits@3_at_best": f"{summary_df['hits@3_at_best'].mean():.4f} ± {summary_df['hits@3_at_best'].std(ddof=0):.4f}",
        "hits@10_at_best": f"{summary_df['hits@10_at_best'].mean():.4f} ± {summary_df['hits@10_at_best'].std(ddof=0):.4f}",
        "loss_at_best": f"{summary_df['loss_at_best'].mean():.4f} ± {summary_df['loss_at_best'].std(ddof=0):.4f}",
    }

    summary_out = pd.concat([summary_df, pd.DataFrame([agg_row])], ignore_index=True)
    summary_out.to_csv(out_csv, index=False)
    print("\nBest-result summary:")
    print(summary_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing seed CSV files")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    dfs = load_csvs(input_dir)
    aligned_dfs = align_by_epoch(dfs)

    print("Loaded seeds:")
    for name, df in aligned_dfs:
        print(f"  - {name}: {len(df)} rows, epochs = {df['epoch'].tolist()}")

    # 1. Loss
    plot_single_metric_mean_std(
        aligned_dfs,
        metric="avg_loss",
        ylabel="Average Loss",
        title="Training Loss vs Epoch (Mean ± Std)",
        output_path=output_dir / "loss_mean_std.png"
    )

    # 2. MRR
    plot_single_metric_mean_std(
        aligned_dfs,
        metric="mrr",
        ylabel="MRR",
        title="MRR vs Epoch (Mean ± Std)",
        output_path=output_dir / "mrr_mean_std.png"
    )

    # 3. Seed-wise MRR comparison
    plot_seed_comparison(
        aligned_dfs,
        metric="mrr",
        ylabel="MRR",
        title="MRR vs Epoch (Per Seed)",
        output_path=output_dir / "mrr_per_seed.png"
    )

    # 4. Hits@K
    plot_hits_mean_std(
        aligned_dfs,
        output_path=output_dir / "hits_mean_std.png"
    )

    # 5. Gate mean
    plot_gate_mean(
        aligned_dfs,
        output_path=output_dir / "gate_mean_analysis.png"
    )

    # 6. Gate std
    plot_gate_std(
        aligned_dfs,
        output_path=output_dir / "gate_std_analysis.png"
    )

    # 7. Image fraction
    if "g_frac_img_in_sample" in aligned_dfs[0][1].columns:
        plot_image_fraction(
            aligned_dfs,
            output_path=output_dir / "image_fraction.png"
        )

    # 8. Best summary
    summarize_best(
        aligned_dfs,
        out_csv=output_dir / "best_summary.csv"
    )

    print(f"\nAll figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()