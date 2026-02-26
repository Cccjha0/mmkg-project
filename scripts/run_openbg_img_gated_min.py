import os
import sys
import torch

sys.path.append(os.path.abspath("."))

from src.models.openbg_img_gated_lp import OpenBGImgGatedLP
from src.train.trainer import Trainer, read_triples_3col
from src.data.build_true_facts import build_true_facts


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # load cache
    text_emb = torch.load("cache/openbg_img/text_emb.pt")
    img_emb = torch.load("cache/openbg_img/img_emb.pt")
    has_img = torch.load("cache/openbg_img/has_img.pt")
    num_entities = text_emb.shape[0]
    num_relations = 136

    # load triples
    train_path = "datasets/openbg_img/raw/OpenBG-IMG_train.tsv"
    dev_path = "datasets/openbg_img/raw/OpenBG-IMG_dev.tsv"

    train_triples = read_triples_3col(train_path)
    dev_triples = read_triples_3col(dev_path)

    print("train:", len(train_triples), "dev:", len(dev_triples))
    true_tails, true_heads = build_true_facts(train_triples + dev_triples)

    model = OpenBGImgGatedLP(
        text_emb=text_emb,
        img_emb=img_emb,
        has_img=has_img,
        num_relations=num_relations,
        d=256,
        use_layernorm=True,
    ).to(device)

    trainer = Trainer(
        model=model,
        train_triples=train_triples,
        dev_triples=dev_triples,
        num_entities=num_entities,
        true_tails=true_tails,
        true_heads=true_heads,
        device=device,
        lr=1e-3,
        batch_size=1024,
        neg_ratio=10,
        epochs=50,
        eval_every=5,
        dev_eval_limit=5000,
        chunk_size=10000,
        patience=10,
        out_dir="outputs/openbg_img/gated/min_run",
    )

    trainer.train()


if __name__ == "__main__":
    main()