import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion.gated import GatedFusion
from src.models.decoders.complex import ComplEx


class OpenBGImgGatedLP(nn.Module):
    def __init__(self, text_emb: torch.Tensor, img_emb: torch.Tensor, has_img: torch.Tensor,
                 num_relations: int, d: int = 256, use_layernorm: bool = True):
        super().__init__()
        self.d = d

        # register cached embeddings as buffers (not trainable)
        self.register_buffer("text_emb", text_emb)  # [N,d]
        self.register_buffer("img_emb", img_emb)    # [N,d]
        self.register_buffer("has_img", has_img)    # [N]

        # missing image token (trainable)
        self.v_missing = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.v_missing, mean=0.0, std=0.02)

        self.fusion = GatedFusion(d=d, use_layernorm=use_layernorm)
        self.decoder = ComplEx(num_relations=num_relations, d=d)

    def _entity_text(self, eids: torch.Tensor) -> torch.Tensor:
        return self.text_emb[eids]  # [B,d]

    def _entity_image(self, eids: torch.Tensor) -> torch.Tensor:
        v = self.img_emb[eids]  # [B,d] (zero for missing in cache)
        mask = self.has_img[eids].unsqueeze(-1)  # [B,1] bool
        # if missing -> v_missing
        v = torch.where(mask, v, self.v_missing.unsqueeze(0).expand_as(v))
        return v

    def _fused(self, eids: torch.Tensor):
        t = self._entity_text(eids)
        v = self._entity_image(eids)
        z, g = self.fusion(t, v)
        return z, g

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        """
        triples: [B,3] on device
        returns scores: [B]
        (Training needs gradients -> DO NOT use torch.no_grad here)
        """
        h = triples[:, 0]
        r = triples[:, 1]
        t = triples[:, 2]
        zh, _ = self._fused(h)
        zt, _ = self._fused(t)
        return self.decoder.score(zh, r, zt)

    @torch.no_grad()
    def score_eval(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score(triples)

    def forward(self, pos_triples: torch.LongTensor, neg_triples: torch.LongTensor) -> torch.Tensor:
        """
        pos_triples: [B,3], neg_triples: [B*neg_ratio,3] (or same B with multiple negatives flattened)
        We'll do BCE loss: pos label=1, neg label=0
        """
        pos_scores = self.score(pos_triples)
        neg_scores = self.score(neg_triples)

        y_pos = torch.ones_like(pos_scores)
        y_neg = torch.zeros_like(neg_scores)

        loss = F.binary_cross_entropy_with_logits(pos_scores, y_pos) + \
               F.binary_cross_entropy_with_logits(neg_scores, y_neg)
        return loss