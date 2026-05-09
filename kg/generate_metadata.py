import json
import os

import pandas as pd

from tqdm import tqdm

# =====================================================
# entity files (Chinese and English)
# =====================================================

ENTITY_FILE_ZH = (
    "../data/datasets/openbg_img/raw/"
    "OpenBG-IMG_entity2text.tsv"
)

ENTITY_FILE_EN = (
    "../data/datasets/openbg_img/raw/"
    "OpenBG-IMG_entity2text_en.tsv"
)

# =====================================================
# relation files (Chinese and English)
# =====================================================

RELATION_FILE_ZH = (
    "../data/datasets/openbg_img/raw/"
    "OpenBG-IMG_relation2text.tsv"
)

RELATION_FILE_EN = (
    "../data/datasets/openbg_img/raw/"
    "OpenBG-IMG_relation2text_en.tsv"
)

# =====================================================
# image root
# =====================================================

IMAGE_ROOT = (
    "../data/datasets/openbg_img/raw/"
    "OpenBG-IMG_images"
)

# =====================================================
# output
# =====================================================

OUTPUT_FILE = "../data/datasets/openbg_img/processed/metadata.json"

# =====================================================
# read entity labels (Chinese)
# =====================================================

print("=" * 60)
print("Reading Chinese entity labels...")
print("=" * 60)

entity_df_zh = pd.read_csv(
    ENTITY_FILE_ZH,
    sep="\t",
    header=None,
    names=["entity", "label"]
)

# =====================================================
# read entity labels (English)
# =====================================================

print("=" * 60)
print("Reading English entity labels...")
print("=" * 60)

entity_df_en = pd.read_csv(
    ENTITY_FILE_EN,
    sep="\t",
    header=None,
    names=["entity", "label"]
)

# =====================================================
# read relation labels (Chinese)
# =====================================================

print("=" * 60)
print("Reading Chinese relation labels...")
print("=" * 60)

relation_df_zh = pd.read_csv(
    RELATION_FILE_ZH,
    sep="\t",
    header=None,
    names=["relation", "label"]
)

# =====================================================
# read relation labels (English)
# =====================================================

print("=" * 60)
print("Reading English relation labels...")
print("=" * 60)

relation_df_en = pd.read_csv(
    RELATION_FILE_EN,
    sep="\t",
    header=None,
    names=["relation", "label"]
)

# =====================================================
# metadata dict
# =====================================================

metadata = {}

# =====================================================
# entities - build with both Chinese and English
# =====================================================

print("=" * 60)
print("Generating entity metadata...")
print("=" * 60)

# Create dictionaries for quick lookup
entity_labels_zh = dict(zip(entity_df_zh["entity"], entity_df_zh["label"]))
entity_labels_en = dict(zip(entity_df_en["entity"], entity_df_en["label"]))

# Get all unique entity IDs
all_entity_ids = set(entity_df_zh["entity"].unique()) | set(entity_df_en["entity"].unique())

for entity_id in tqdm(all_entity_ids):

    entity_id_str = str(entity_id)

    label_zh = str(entity_labels_zh.get(entity_id, ""))
    label_en = str(entity_labels_en.get(entity_id, ""))

    # Use English label as default if available, otherwise Chinese
    default_label = label_en if label_en else label_zh

    # ==========================================
    # image url
    # ==========================================

    image_url = (
        f"/images/{entity_id_str}/image_0.jpg"
    )

    metadata[entity_id_str] = {
        "label": default_label,
        "label_zh": label_zh,
        "label_en": label_en,
        "image": image_url
    }

# =====================================================
# relations - build with both Chinese and English
# =====================================================

print("=" * 60)
print("Generating relation metadata...")
print("=" * 60)

# Create dictionaries for quick lookup
relation_labels_zh = dict(zip(relation_df_zh["relation"], relation_df_zh["label"]))
relation_labels_en = dict(zip(relation_df_en["relation"], relation_df_en["label"]))

# Get all unique relation IDs
all_relation_ids = set(relation_df_zh["relation"].unique()) | set(relation_df_en["relation"].unique())

for relation_id in tqdm(all_relation_ids):

    relation_id_str = str(relation_id)

    label_zh = str(relation_labels_zh.get(relation_id, ""))
    label_en = str(relation_labels_en.get(relation_id, ""))

    # Use English label as default if available, otherwise Chinese
    default_label = label_en if label_en else label_zh

    metadata[relation_id_str] = {
        "label": default_label,
        "label_zh": label_zh,
        "label_en": label_en
    }

# =====================================================
# save json
# =====================================================

print("=" * 60)
print("Saving metadata.json ...")
print("=" * 60)

with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

# =====================================================
# done
# =====================================================

print("=" * 60)
print("Done!")
print(f"Total metadata items: {len(metadata)}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 60)
