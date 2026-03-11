import os
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def pick_device(device_arg: str) -> torch.device:
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/datasets/openbg500/raw/OpenBG500_entity2text.tsv")
    parser.add_argument("--save_path", default="data/cache/openbg500/entity_bert_emb.pt")
    parser.add_argument("--model_name", default="bert-base-chinese")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto/cuda/mps/cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name).to(device)
    model.eval()

    df = pd.read_csv(args.data_path, sep="\t", header=None, names=["entity_id", "text"])
    df["text"] = df["text"].fillna("").astype(str)
    texts = df["text"].tolist()

    embeddings = []
    print(f"Starting feature extraction, total {len(texts)} entries...")

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size)):
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
    torch.save(all_embeddings, args.save_path)

    print(f"Feature extraction completed! Saved to: {args.save_path}")
    print(f"Vector shape: {all_embeddings.shape}")


if __name__ == "__main__":
    main()
