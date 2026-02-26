import os
import sys
import argparse
import torch

sys.path.append(os.path.abspath("."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.tsv_reader import read_allow_2or3
from src.data.build_true_facts import build_true_facts
from src.models.build_model import build_model
from src.train.trainer_yaml import TrainerYAML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to experiment yaml, e.g., configs/openbg_img_gated.yaml")
    ap.add_argument("--common", default="configs/common.yaml", help="path to common yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, args.common)

    # seed
    seed = cfg["system"].get("seed", 1)
    deterministic = cfg["system"].get("deterministic", False)
    set_seed(seed, deterministic=deterministic)

    # device
    device = cfg["system"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda not available, switching to cpu")
        device = "cpu"
        cfg["system"]["device"] = "cpu"

    # load triples
    train_path = cfg["dataset"]["train"]
    dev_path = cfg["dataset"]["dev"]

    train3, _, bad_train = read_allow_2or3(train_path)
    dev3, _, bad_dev = read_allow_2or3(dev_path)

    if bad_train or bad_dev:
        print(f"[WARN] malformed lines skipped: train={bad_train}, dev={bad_dev}")

    if len(train3) == 0 or len(dev3) == 0:
        raise RuntimeError("Train/Dev must contain 3-column triples for training/evaluation.")

    print(f"train triples: {len(train3)} | dev triples: {len(dev3)}")

    # build filtered facts (train+dev)
    true_tails, true_heads = build_true_facts(train3 + dev3)

    # build model from config
    model, num_entities = build_model(cfg)
    model = model.to(device)

    # run trainer
    trainer = TrainerYAML(
        model=model,
        train_triples=train3,
        dev_triples=dev3,
        num_entities=num_entities,
        true_tails=true_tails,
        true_heads=true_heads,
        cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()