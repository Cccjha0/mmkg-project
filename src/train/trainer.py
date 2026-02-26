import os
import torch
from tqdm import tqdm

from src.data.sampler import negative_sample
from src.eval.filtered_ranking import filtered_ranking_eval
from src.data.build_true_facts import build_true_facts


def read_triples_3col(path: str):
    def pe(x): return int(x.replace("ent_", ""))
    def pr(x): return int(x.replace("rel_", ""))
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
            triples.append((pe(h), pr(r), pe(t)))
    return triples


class Trainer:
    def __init__(
            self,
            model,
            train_triples,
            dev_triples,
            num_entities,
            true_tails,
            true_heads,
            device="cuda",
            lr=1e-3,
            batch_size=1024,
            neg_ratio=10,
            epochs=50,
            eval_every=5,
            dev_eval_limit=1000,
            chunk_size=10000,
            patience=10,
            out_dir="outputs/openbg_img/gated/run1",
    ):
        self.model = model
        self.train_triples = train_triples
        self.dev_triples = dev_triples
        self.num_entities = num_entities
        self.true_tails = true_tails
        self.true_heads = true_heads

        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.epochs = epochs
        self.eval_every = eval_every
        self.dev_eval_limit = dev_eval_limit
        self.chunk_size = chunk_size
        self.patience = patience
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        best_mrr = -1.0
        bad_epochs = 0

        train_tensor = torch.tensor(self.train_triples, dtype=torch.long)
        dev_tensor = torch.tensor(self.dev_triples[: self.dev_eval_limit], dtype=torch.long)

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            # shuffle indices
            perm = torch.randperm(train_tensor.size(0))
            total_loss = 0.0
            steps = 0

            for i in tqdm(range(0, train_tensor.size(0), self.batch_size), desc=f"epoch {epoch}"):
                idx = perm[i : i + self.batch_size]
                pos = train_tensor[idx].to(self.device)

                neg = negative_sample(pos, num_entities=self.num_entities, neg_ratio=self.neg_ratio)

                loss = self.model(pos, neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += float(loss.item())
                steps += 1

            avg_loss = total_loss / max(1, steps)
            print(f"[Train] epoch={epoch} avg_loss={avg_loss:.6f}")

            # eval
            if epoch % self.eval_every == 0:
                self.model.eval()
                metrics = filtered_ranking_eval(
                    model=self.model,
                    triples=dev_tensor,
                    true_tails=self.true_tails,
                    true_heads=self.true_heads,
                    num_entities=self.num_entities,
                    chunk_size=self.chunk_size,
                    device=self.device,
                    ks=(1, 3, 10),
                )
                print(f"[Dev] epoch={epoch} " + " ".join([f"{k}={v:.6f}" for k, v in metrics.items()]))

                mrr = metrics["mrr"]
                if mrr > best_mrr:
                    best_mrr = mrr
                    bad_epochs = 0
                    ckpt_path = os.path.join(self.out_dir, "best.ckpt")
                    torch.save(self.model.state_dict(), ckpt_path)
                    print(f"[CKPT] saved best to {ckpt_path}")
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        print("[EarlyStop] triggered.")
                        break

        print(f"[Done] best_dev_mrr={best_mrr:.6f}")
        return best_mrr