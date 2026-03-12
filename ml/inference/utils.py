import json
from pathlib import Path
from typing import Any


def parse_token_id(value: int | str, *, prefix: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith(prefix):
        return int(value[len(prefix):])
    raise ValueError(f"Unsupported value for prefix '{prefix}': {value}")


def format_token_id(value: int, *, prefix: str, width: int) -> str:
    return f"{prefix}{int(value):0{width}d}"


def parse_entity_id(value: int | str) -> int:
    return parse_token_id(value, prefix="ent_")


def parse_relation_id(value: int | str) -> int:
    return parse_token_id(value, prefix="rel_")


def format_entity_id(value: int) -> str:
    return format_token_id(value, prefix="ent_", width=6)


def format_relation_id(value: int) -> str:
    return format_token_id(value, prefix="rel_", width=4)


def infer_text_map_paths(train_path: str | Path | None) -> tuple[Path | None, Path | None]:
    if not train_path:
        return None, None

    train_file = Path(train_path)
    raw_dir = train_file.parent
    stem = train_file.stem
    prefix = stem[:-6] if stem.endswith("_train") else stem
    return raw_dir / f"{prefix}_entity2text.tsv", raw_dir / f"{prefix}_relation2text.tsv"


def load_tsv_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}

    file_path = Path(path)
    if not file_path.is_file():
        return {}

    out: dict[str, str] = {}
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            key, value = parts
            out[key.strip()] = value.strip()
    return out


def build_response(
    *,
    task: str,
    model: str,
    device: str,
    inputs: dict[str, Any],
    results: Any,
    latency_ms: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "task": task,
        "model": model,
        "device": device,
        "inputs": inputs,
        "results": results,
    }
    if latency_ms is not None:
        out["latency_ms"] = float(latency_ms)
    if extra:
        out.update(extra)
    return out


def to_pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
