from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "datasets" / "openbg_img" / "raw"
DEFAULT_ENTITY_FILE_ZH = RAW_DIR / "OpenBG-IMG_entity2text.tsv"
DEFAULT_ENTITY_FILE_EN = RAW_DIR / "OpenBG-IMG_entity2text_en.tsv"
DEFAULT_RELATION_FILE_ZH = RAW_DIR / "OpenBG-IMG_relation2text.tsv"
DEFAULT_RELATION_FILE_EN = RAW_DIR / "OpenBG-IMG_relation2text_en.tsv"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "datasets" / "openbg_img" / "processed" / "metadata.json"


def _read_label_map(path: str | Path, key_name: str) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t", header=None, names=[key_name, "label"])
    return {str(key): str(label) for key, label in zip(df[key_name], df["label"])}


def generate_metadata(
    entity_file_zh: str | Path = DEFAULT_ENTITY_FILE_ZH,
    entity_file_en: str | Path = DEFAULT_ENTITY_FILE_EN,
    relation_file_zh: str | Path = DEFAULT_RELATION_FILE_ZH,
    relation_file_en: str | Path = DEFAULT_RELATION_FILE_EN,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    *,
    show_progress: bool = True,
) -> dict[str, dict[str, str]]:
    """Generate entity/relation display metadata for the KG Flask service."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    entity_labels_zh = _read_label_map(entity_file_zh, "entity")
    entity_labels_en = _read_label_map(entity_file_en, "entity")
    relation_labels_zh = _read_label_map(relation_file_zh, "relation")
    relation_labels_en = _read_label_map(relation_file_en, "relation")

    metadata: dict[str, dict[str, str]] = {}

    entity_ids = sorted(set(entity_labels_zh) | set(entity_labels_en))
    entity_iterator = tqdm(entity_ids, desc="Entities") if show_progress else entity_ids
    for entity_id in entity_iterator:
        label_zh = entity_labels_zh.get(entity_id, "")
        label_en = entity_labels_en.get(entity_id, "")
        metadata[entity_id] = {
            "label": label_en or label_zh,
            "label_zh": label_zh,
            "label_en": label_en,
            "image": f"/images/{entity_id}/image_0.jpg",
        }

    relation_ids = sorted(set(relation_labels_zh) | set(relation_labels_en))
    relation_iterator = tqdm(relation_ids, desc="Relations") if show_progress else relation_ids
    for relation_id in relation_iterator:
        label_zh = relation_labels_zh.get(relation_id, "")
        label_en = relation_labels_en.get(relation_id, "")
        metadata[relation_id] = {
            "label": label_en or label_zh,
            "label_zh": label_zh,
            "label_en": label_en,
        }

    output_file.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    print("=" * 60)
    print("Generating metadata.json ...")
    print("=" * 60)
    metadata = generate_metadata()
    print("=" * 60)
    print("Done!")
    print(f"Total metadata items: {len(metadata)}")
    print(f"Saved to: {DEFAULT_OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
