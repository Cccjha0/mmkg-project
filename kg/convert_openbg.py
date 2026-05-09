from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "datasets" / "openbg_img" / "raw" / "OpenBG-IMG_train.tsv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "datasets" / "openbg_img" / "processed" / "data.csv"


def convert_openbg(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    show_progress: bool = True,
) -> int:
    """Convert OpenBG-IMG train triples from TSV into the CSV format used by KG search."""
    train_path = Path(train_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path, sep="\t", header=None)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        rows = df.itertuples(index=False)
        iterator = tqdm(rows, total=len(df), desc="Processing Triples", ncols=100) if show_progress else rows
        for h, r, t in iterator:
            file.write(f"{h},{r},{t}\n")

    return len(df)


def main() -> None:
    print("=" * 50)
    print("Reading OpenBG-IMG_train.tsv ...")
    print("=" * 50)
    count = convert_openbg()
    print("\n" + "=" * 50)
    print("data.csv generated successfully.")
    print(f"Total triples: {count}")
    print(f"Saved to: {DEFAULT_OUTPUT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
