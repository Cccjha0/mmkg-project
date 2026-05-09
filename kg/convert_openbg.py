import pandas as pd
import os
from tqdm import tqdm

# =========================
# OpenBG train 文件路径
# =========================

train_path = os.path.join(
    "..",
    "data",
    "datasets",
    "openbg_img",
    "raw",
    "OpenBG-IMG_train.tsv"
)

# =========================
# 输出 data.csv 路径
# =========================

output_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "datasets",
    "openbg_img",
    "processed",
    "data.csv"
)

print("=" * 50)
print("开始读取 OpenBG-IMG_train.tsv ...")
print("=" * 50)

# =========================
# 读取 TSV
# =========================

df = pd.read_csv(
    train_path,
    sep="\t",
    header=None
)

print(f"\n读取完成！")
print(f"总三元组数量: {len(df)}")

print("\n开始写入 data.csv ...")

# =========================
# 手动写入（支持进度条）
# =========================

with open(output_path, "w", encoding="utf-8") as f:

    for row in tqdm(
            df.itertuples(index=False),
            total=len(df),
            desc="Processing Triples",
            ncols=100
    ):

        h, r, t = row

        f.write(f"{h},{r},{t}\n")

print("\n" + "=" * 50)
print("data.csv 生成成功！")
print(f"保存位置: {output_path}")
print("=" * 50)

print("\n前5条数据：")
print(df.head())