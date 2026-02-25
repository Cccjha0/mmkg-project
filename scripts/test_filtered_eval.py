import os
import sys
import torch

sys.path.append(os.path.abspath("."))

from src.data.build_true_facts import build_true_facts
from src.eval.filtered_ranking import filtered_ranking_eval
from src.models.debug_random import RandomScorer


def parse_ent(ent_str: str) -> int:
    if not ent_str.startswith("ent_"):
        raise ValueError(f"Bad entity: {ent_str}")
    return int(ent_str.replace("ent_", ""))


def parse_rel(rel_str: str) -> int:
    if not rel_str.startswith("rel_"):
        raise ValueError(f"Bad relation: {rel_str}")
    return int(rel_str.replace("rel_", ""))


def read_file_allow_2or3(path: str):
    """
    Returns:
      triples3: list[(h,r,t)] int
      queries2: list[(h,r)] int
    """
    triples3 = []
    queries2 = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                h, r, t = parts
                triples3.append((parse_ent(h), parse_rel(r), parse_ent(t)))
            elif len(parts) == 2:
                h, r = parts
                queries2.append((parse_ent(h), parse_rel(r)))
            else:
                bad += 1
                print(f"[WARN] {path} line {ln}: cols={len(parts)} | {repr(line)}")
    if bad:
        print(f"[WARN] total bad lines skipped in {path}: {bad}")
    return triples3, queries2


def main():
    train_path = "datasets/openbg_img/raw/OpenBG-IMG_train.tsv"
    dev_path = "datasets/openbg_img/raw/OpenBG-IMG_dev.tsv"
    test_path = "datasets/openbg_img/raw/OpenBG-IMG_test.tsv"

    print("Loading files (allow 2 or 3 columns)...")

    train3, train2 = read_file_allow_2or3(train_path)
    dev3, dev2 = read_file_allow_2or3(dev_path)
    test3, test2 = read_file_allow_2or3(test_path)

    print(f"Train triples(3-col): {len(train3)} | Train queries(2-col): {len(train2)}")
    print(f"Dev   triples(3-col): {len(dev3)}   | Dev   queries(2-col): {len(dev2)}")
    print(f"Test  triples(3-col): {len(test3)}  | Test  queries(2-col): {len(test2)}")

    if len(dev3) == 0:
        raise RuntimeError("Dev file has no 3-column triples, cannot compute MRR/Hits.")

    if len(test2) > 0 and len(test3) == 0:
        print("[INFO] Test file is a query set (2 columns). It will be used for top-k prediction, not for MRR.")

    # Build filtered dictionaries using all known true triples (train+dev [+test if any])
    print("Building filtered dictionaries from train+dev (+test3 if exists)...")
    all_true_triples = train3 + dev3 + test3
    true_tails, true_heads = build_true_facts(all_true_triples)

    # num_entities from cache (best aligned with your embeddings)
    num_entities = torch.load("cache/openbg_img/text_emb.pt").shape[0]
    print("num_entities =", num_entities)

    # Sanity eval on a small dev subset
    dev_subset = torch.tensor(dev3[:200], dtype=torch.long)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running filtered eval on dev subset (n={len(dev_subset)}) using RandomScorer on {device}...")

    model = RandomScorer(seed=1)

    metrics = filtered_ranking_eval(
        model=model,
        triples=dev_subset,
        true_tails=true_tails,
        true_heads=true_heads,
        num_entities=num_entities,
        chunk_size=10000,
        device=device,
        ks=(1, 3, 10),
    )

    print("\n=== Filtered Eval Results (Random Model on DEV subset) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    print("\n[OK] Filtered eval pipeline works. Next: implement real scorer (ComplEx) and train Gated Fusion.")


if __name__ == "__main__":
    main()