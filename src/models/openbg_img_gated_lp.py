import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion.gated import RelAwareGatedFusion
from src.models.decoders.complex import ComplEx


class OpenBGImgGatedLP(nn.Module):
    def __init__(self, text_emb: torch.Tensor, img_emb: torch.Tensor, has_img: torch.Tensor,
                 num_relations: int, d: int = 256, use_layernorm: bool = True):
        super().__init__()
        self.d = d
        self.num_relations = num_relations
        num_entities = text_emb.shape[0]

        # register cached embeddings as buffers (not trainable)
        self.register_buffer("text_emb", text_emb)  # [N,d]
        self.register_buffer("img_emb", img_emb)    # [N,d]
        self.register_buffer("has_img", has_img)    # [N]

        # missing image token (trainable)
        self.v_missing = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.v_missing, mean=0.0, std=0.02)

        self.entity_residual = nn.Embedding(num_entities, d)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))  # softplus(-2)≈0.127
        nn.init.xavier_uniform_(self.entity_residual.weight)

        self.fusion = RelAwareGatedFusion(d=d, num_relations=num_relations, use_layernorm=use_layernorm)
        self.decoder = ComplEx(num_relations=num_relations, d=d)

    def _entity_text(self, eids: torch.Tensor) -> torch.Tensor:
        return self.text_emb[eids]  # [B,d]

    def _entity_image(self, eids: torch.Tensor) -> torch.Tensor:
        v = self.img_emb[eids]  # [B,d] (zero for missing in cache)
        mask = self.has_img[eids].unsqueeze(-1)  # [B,1] bool
        # if missing -> v_missing
        v = torch.where(mask, v, self.v_missing.unsqueeze(0).expand_as(v))
        return v

    def _fused_with_r(self, eids: torch.LongTensor, rids: torch.LongTensor):
        t = self._entity_text(eids)
        v = self._entity_image(eids)
        z_fused, g = self.fusion(t, v, rids)
        res = self.entity_residual(eids)
        scale = F.softplus(self.residual_scale)  # >0
        z = z_fused + scale * res
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
        zh, _ = self._fused_with_r(h, r)
        zt, _ = self._fused_with_r(t, r)
        return self.decoder.score(zh, r, zt)

    @torch.no_grad()
    def gate_for_entities(self, eids: torch.LongTensor) -> torch.Tensor:
        """
        eids: [B] entity ids on device
        return: gate g [B] in [0,1] under random relation ids
        """
        rids = torch.randint(0, self.num_relations, (eids.size(0),), device=eids.device)
        t = self._entity_text(eids)
        v = self._entity_image(eids)
        _, g = self.fusion(t, v, rids)  # [B,d]
        return g.mean(dim=-1)   # [B]

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

        bce_loss = F.binary_cross_entropy_with_logits(pos_scores, y_pos) + \
                   F.binary_cross_entropy_with_logits(neg_scores, y_neg)
        l2 = 1e-6 * self.entity_residual.weight.pow(2).mean()
        scale = F.softplus(self.residual_scale)
        scale_l2 = 1e-4 * scale.pow(2)
        loss = bce_loss + l2 + scale_l2
        return loss
