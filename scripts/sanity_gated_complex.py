import os
import sys
import torch
import random

sys.path.append(os.path.abspath("."))

from src.models.openbg_img_gated_lp import OpenBGImgGatedLP


def parse_ent(s: str) -> int:
    return int(s.replace("ent_", ""))


def parse_rel(s: str) -> int:
    return int(s.replace("rel_", ""))


def read_triples_3col(path: str, limit: int = None):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((parse_ent(h), parse_rel(r), parse_ent(t)))
            if limit and len(triples) >= limit:
                break
    return triples


def corrupt_tail(pos_batch, num_entities):
    # pos_batch: [B,3]
    neg = pos_batch.clone()
    neg[:, 2] = torch.randint(0, num_entities, (neg.size(0),), device=neg.device)
    return neg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # load cache
    text_emb = torch.load("cache/openbg_img/text_emb.pt")
    img_emb = torch.load("cache/openbg_img/img_emb.pt")
    has_img = torch.load("cache/openbg_img/has_img.pt")

    num_entities = text_emb.shape[0]
    num_relations = 136

    model = OpenBGImgGatedLP(
        text_emb=text_emb,
        img_emb=img_emb,
        has_img=has_img,
        num_relations=num_relations,
        d=256,
        use_layernorm=True,
    ).to(device)

    train_path = "datasets/openbg_img/raw/OpenBG-IMG_train.tsv"
    triples = read_triples_3col(train_path, limit=1024)
    pos = torch.tensor(triples[:256], dtype=torch.long, device=device)

    # score
    with torch.no_grad():
        s = model.score(pos)
    print("score shape:", tuple(s.shape), "mean:", float(s.mean()), "std:", float(s.std()))

    # forward loss with simple negatives
    neg = corrupt_tail(pos, num_entities)
    loss = model(pos, neg)
    print("loss:", float(loss))

    print("[OK] sanity passed")


if __name__ == "__main__":
    main()