from __future__ import annotations

import json
from pathlib import Path

from app.services.search import Search


def test_search_returns_bilingual_graph_and_respects_limits(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    metadata_path = tmp_path / "metadata.json"
    csv_path.write_text(
        "\n".join(
            [
                "ent_000001,rel_0001,ent_000002",
                "ent_000001,rel_0001,ent_000003",
                "ent_000002,rel_0002,ent_000004",
            ]
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "ent_000001": {"label": "Pants", "label_zh": "裤子", "label_en": "Pants"},
                "ent_000002": {"label": "Black Pants", "label_zh": "黑色裤子", "label_en": "Black Pants"},
                "ent_000003": {"label": "No Image Product", "label_zh": "无图商品", "label_en": "No Image Product"},
                "ent_000004": {"label": "Cotton", "label_zh": "棉", "label_en": "Cotton"},
                "rel_0001": {"label": "category", "label_zh": "类别", "label_en": "category"},
                "rel_0002": {"label": "material", "label_zh": "材质", "label_en": "material"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    search = Search(str(csv_path)).load_metadata(str(metadata_path))
    result = search(query="Pants", k=1, n=2, p=999, lang="en")

    assert result["nodes"]
    assert result["links"]
    assert len(result["nodes"]) <= Search.MAX_GRAPH_NODES
    assert len(result["links"]) <= Search.MAX_GRAPH_LINKS
    assert result["nodes"][0]["label"] == "Pants"
    assert result["links"][0]["relation_en"] == "category"


def test_explore_limited_caps_dense_graph(tmp_path: Path) -> None:
    csv_path = tmp_path / "dense.csv"
    csv_path.write_text(
        "\n".join(f"ent_000000,rel_0001,ent_{idx:06d}" for idx in range(1, 400)),
        encoding="utf-8",
    )

    search = Search(str(csv_path))
    pairs = search.explore_limited("ent_000000", max_depth=2, prune=999)

    assert len(pairs) <= Search.MAX_GRAPH_LINKS
