from __future__ import annotations

import json
from pathlib import Path

from kg.convert_openbg import convert_openbg
from kg.generate_metadata import generate_metadata


def test_convert_openbg_writes_search_csv(tmp_path: Path) -> None:
    train_path = tmp_path / "OpenBG-IMG_train.tsv"
    output_path = tmp_path / "processed" / "data.csv"
    train_path.write_text(
        "ent_000001\trel_0001\tent_000002\n"
        "ent_000002\trel_0002\tent_000003\n",
        encoding="utf-8",
    )

    count = convert_openbg(train_path, output_path, show_progress=False)

    assert count == 2
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "ent_000001,rel_0001,ent_000002",
        "ent_000002,rel_0002,ent_000003",
    ]


def test_generate_metadata_merges_bilingual_labels(tmp_path: Path) -> None:
    entity_zh = tmp_path / "entity_zh.tsv"
    entity_en = tmp_path / "entity_en.tsv"
    relation_zh = tmp_path / "relation_zh.tsv"
    relation_en = tmp_path / "relation_en.tsv"
    output = tmp_path / "processed" / "metadata.json"

    entity_zh.write_text("ent_000001\t裤子\nent_000002\t无图商品\n", encoding="utf-8")
    entity_en.write_text("ent_000001\tPants\nent_000002\tNo Image Product\n", encoding="utf-8")
    relation_zh.write_text("rel_0001\t类别\n", encoding="utf-8")
    relation_en.write_text("rel_0001\tcategory\n", encoding="utf-8")

    metadata = generate_metadata(
        entity_zh,
        entity_en,
        relation_zh,
        relation_en,
        output,
        show_progress=False,
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == metadata
    assert metadata["ent_000001"]["label"] == "Pants"
    assert metadata["ent_000001"]["label_zh"] == "裤子"
    assert metadata["ent_000002"]["image"] == "/images/ent_000002/image_0.jpg"
    assert metadata["rel_0001"]["label_en"] == "category"
