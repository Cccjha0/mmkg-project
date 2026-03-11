import copy
import yaml


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(exp_path: str, common_path: str = "ml/configs/common.yaml") -> dict:
    common = load_yaml(common_path) if common_path else {}
    exp = load_yaml(exp_path)
    cfg = _deep_update(common, exp)
    cfg["_config_paths"] = {"common": common_path, "exp": exp_path}
    return cfg
