from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_runtime_config() -> dict[str, str]:
    """
    这里只返回基础配置。
    实际开发时可以在这里初始化 predictor 或统一配置对象。
    """
    return {
        "run_dir": os.getenv(
            "MMKG_RUN_DIR",
            "ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260308_123356_seed1",
        ),
        "device": os.getenv("MMKG_DEVICE", "cpu"),
        "dataset_name": "openbg_img",
        "model_name": "residual_gate",
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
