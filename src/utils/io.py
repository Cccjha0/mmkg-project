import os
import json
import shutil
from datetime import datetime


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def now_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(root_dir: str, exp_name: str, seed: int):
    run_id = now_run_id()
    out_dir = os.path.join(root_dir, exp_name, f"{run_id}_seed{seed}")
    ensure_dir(out_dir)
    return out_dir


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv(path: str, row: dict, header_order=None):
    exists = os.path.exists(path)
    keys = header_order or list(row.keys())
    line = ",".join(str(row.get(k, "")) for k in keys)
    if not exists:
        header = ",".join(keys)
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write(line + "\n")
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def copy_file(src: str, dst_dir: str, new_name: str = None):
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, new_name or os.path.basename(src))
    shutil.copy2(src, dst)
    return dst