from __future__ import annotations

import csv
import math
import mimetypes
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from app.deps import openbg_img_cache_path, openbg_img_raw_path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _token_to_int(token: str, prefix: str) -> int:
    if not token.startswith(prefix):
        raise ValueError(f"Invalid token {token!r}; expected prefix {prefix!r}")
    return int(token[len(prefix) :])


def entity_sort_key(entity: str) -> int:
    return _token_to_int(entity, "ent_")


def relation_sort_key(relation: str) -> int:
    return _token_to_int(relation, "rel_")


@lru_cache(maxsize=1)
def load_entity_text_map() -> dict[str, str]:
    return _load_text_map(openbg_img_raw_path("OpenBG-IMG_entity2text.tsv"))


@lru_cache(maxsize=1)
def load_entity_text_en_map() -> dict[str, str]:
    return _load_text_map(openbg_img_raw_path("OpenBG-IMG_entity2text_en.tsv"))


@lru_cache(maxsize=1)
def load_relation_text_map() -> dict[str, str]:
    return _load_text_map(openbg_img_raw_path("OpenBG-IMG_relation2text.tsv"))


@lru_cache(maxsize=1)
def load_relation_text_en_map() -> dict[str, str]:
    return _load_text_map(openbg_img_raw_path("OpenBG-IMG_relation2text_en.tsv"))


def _load_text_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        for row in csv.reader(file, delimiter="\t"):
            if len(row) >= 2:
                out[row[0].strip()] = row[1].strip()
    return out


@lru_cache(maxsize=1)
def load_train_triples() -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with openbg_img_raw_path("OpenBG-IMG_train.tsv").open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


@lru_cache(maxsize=1)
def triples_by_head() -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for head, relation, tail in load_train_triples():
        out[head].append((relation, tail))
    return dict(out)


@lru_cache(maxsize=1)
def triples_by_pair() -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for head, relation, tail in load_train_triples():
        out[(head, relation)].append(tail)
    return dict(out)


@lru_cache(maxsize=1)
def triples_by_entity() -> dict[str, list[tuple[str, str, str]]]:
    out: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for head, relation, tail in load_train_triples():
        out[head].append((head, relation, tail))
        out[tail].append((head, relation, tail))
    return dict(out)


@lru_cache(maxsize=1)
def relation_frequency() -> Counter[str]:
    return Counter(relation for _, relation, _ in load_train_triples())


@lru_cache(maxsize=1)
def has_image_entities() -> set[str]:
    root = openbg_img_raw_path("OpenBG-IMG_images")
    if not root.is_dir():
        return set()
    return {
        child.name
        for child in root.iterdir()
        if child.is_dir() and any((child / f"image_0{ext}").is_file() for ext in IMAGE_EXTENSIONS)
    }


def entity_text(entity: str) -> str | None:
    return load_entity_text_map().get(entity)


def entity_text_en(entity: str) -> str | None:
    return load_entity_text_en_map().get(entity)


def relation_text(relation: str) -> str | None:
    return load_relation_text_map().get(relation)


def relation_text_en(relation: str) -> str | None:
    return load_relation_text_en_map().get(relation)


def image_path_for_entity(entity: str) -> str | None:
    directory = openbg_img_raw_path("OpenBG-IMG_images", entity)
    for ext in IMAGE_EXTENSIONS:
        path = directory / f"image_0{ext}"
        if path.is_file():
            return f"/static/openbg_img/{entity}/{path.name}"
    return None


def image_media_type(path: str | None) -> str | None:
    if path is None:
        return None
    return mimetypes.guess_type(path)[0]


def entity_exists(entity: str) -> bool:
    return entity in load_entity_text_map()


def relation_exists(relation: str) -> bool:
    return relation in load_relation_text_map()


def top_attribute_relations(limit: int = 8) -> list[str]:
    return [relation for relation, _ in relation_frequency().most_common(limit)]


@lru_cache(maxsize=1)
def relation_tail_candidates() -> dict[str, list[str]]:
    candidates: dict[str, set[str]] = defaultdict(set)
    for _, relation, tail in load_train_triples():
        candidates[relation].add(tail)

    return {
        relation: sorted(values, key=entity_sort_key)
        for relation, values in candidates.items()
    }


def relation_tail_count(relation: str) -> int:
    return len(relation_tail_candidates().get(relation, []))


def entity_embedding_summary(entity: str) -> dict:
    summary: dict = {}
    text_meta = openbg_img_cache_path("text_meta.json")
    img_meta = openbg_img_cache_path("img_meta.json")
    if text_meta.is_file():
        summary["text"] = {"dim": 256, "l2_norm": 1.0, "mean": None, "std": None}
    if image_path_for_entity(entity) is not None and img_meta.is_file():
        summary["image"] = {"dim": 256, "l2_norm": 1.0, "mean": None, "std": None}
    return summary


def similar_by_text_embedding(entity: str, topk: int) -> list[tuple[str, float]]:
    return []
