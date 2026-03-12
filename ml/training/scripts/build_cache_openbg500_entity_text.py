import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def parse_ent_id(ent_str: str) -> int:
    if not ent_str.startswith("ent_"):
        raise ValueError(f"Bad entity id: {ent_str}")
    return int(ent_str.replace("ent_", ""))


def pick_device(device_arg: str) -> torch.device:
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_entity_texts(data_path: str) -> tuple[list[str], int]:
    df = pd.read_csv(data_path, sep="\t", header=None, names=["entity_id", "text"])
    df["entity_id"] = df["entity_id"].astype(str).map(parse_ent_id)
    df["text"] = df["text"].fillna("").astype(str)

    max_entity_id = int(df["entity_id"].max())
    num_entities = max_entity_id + 1
    texts = [""] * num_entities

    for row in df.itertuples(index=False):
        texts[int(row.entity_id)] = row.text

    return texts, num_entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/datasets/openbg500/raw/OpenBG500_entity2text.tsv")
    parser.add_argument("--save_path", default="data/cache/openbg500/entity_bert_emb.pt")
    parser.add_argument("--model_name", default="bert-base-chinese")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto/cuda/mps/cpu")
    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name).to(device)
    model.eval()

    texts, num_entities = load_entity_texts(args.data_path)
    print(f"Starting feature extraction, total {num_entities} entities...")

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, num_entities, args.batch_size)):
            batch_texts = texts[i:i + args.batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls_embeddings)

    all_embeddings = torch.cat(embeddings, dim=0)
    if all_embeddings.shape[0] != num_entities:
        raise RuntimeError(
            f"Embedding row count mismatch: got {all_embeddings.shape[0]}, expected {num_entities}"
        )

    torch.save(all_embeddings, args.save_path)
    print(f"Feature extraction completed! Saved to: {args.save_path}")
    print(f"Vector shape: {all_embeddings.shape}")


if __name__ == "__main__":
    main()
