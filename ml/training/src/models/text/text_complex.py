import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.src.models.decoders.complex import ComplEx


class TextComplEx(nn.Module):
    def __init__(self, config, num_entities, num_relations, bert_emb_path):
        super().__init__()

        self.d = config["model"]["embedding_dim"]
        if self.d % 2 != 0:
            raise ValueError(f"TextComplEx requires even embedding_dim, got {self.d}")
        self.normalize_bert = bool(config["model"].get("normalize_bert", False))

        print(f"Member A: Loading BERT embeddings from {bert_emb_path}")
        bert_emb = torch.load(bert_emb_path, map_location="cpu").float()
        if bert_emb.shape[0] != num_entities:
            raise ValueError(
                f"BERT embedding size mismatch: cache has {bert_emb.shape[0]} rows, "
                f"expected num_entities={num_entities}"
            )
        self.register_buffer("bert_base", bert_emb)

        self.re_map = nn.Linear(768, self.d // 2)
        self.im_map = nn.Linear(768, self.d // 2)
        nn.init.xavier_uniform_(self.re_map.weight)
        nn.init.xavier_uniform_(self.im_map.weight)
        nn.init.zeros_(self.re_map.bias)
        nn.init.zeros_(self.im_map.bias)

        self.decoder = ComplEx(num_relations, self.d)

    def get_entity_emb(self, entity_ids):
        base = self.bert_base[entity_ids]
        if self.normalize_bert:
            base = F.normalize(base, dim=-1)
        h_re = self.re_map(base)
        h_im = self.im_map(base)
        return torch.cat([h_re, h_im], dim=-1)

    def score(self, triples):
        h_emb = self.get_entity_emb(triples[:, 0])
        r_id = triples[:, 1]
        t_emb = self.get_entity_emb(triples[:, 2])
        return self.decoder.score(h_emb, r_id, t_emb)

    def forward(self, pos_triples, neg_triples):
        pos_scores = self.score(pos_triples)
        neg_scores = self.score(neg_triples)

        batch_size = pos_scores.size(0)
        neg_scores = neg_scores.view(batch_size, -1)
        return -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-10).mean()
