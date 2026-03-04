import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion.gated import RelAwareGatedFusion
from src.models.decoders.complex import ComplEx


class OpenBGImgGatedLP(nn.Module):
    def __init__(self, text_emb: torch.Tensor, img_emb: torch.Tensor, has_img: torch.Tensor,
                 num_relations: int, d: int = 256, use_layernorm: bool = True,
                 neg_ratio: int = 10, adv_temperature: float = 1.0, img_dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.num_relations = num_relations
        self.neg_ratio = neg_ratio
        self.adv_temperature = adv_temperature
        self.img_dropout = float(img_dropout)
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
        has_img = self.has_img[eids]  # [B] bool
        mask = has_img.unsqueeze(-1)  # [B,1] bool
        # if missing -> v_missing
        v = torch.where(mask, v, self.v_missing.unsqueeze(0).expand_as(v))

        # modality dropout: only drop image modality for entities that actually have images
        if self.training and self.img_dropout > 0:
            drop_mask = (torch.rand(eids.size(0), device=eids.device) < self.img_dropout) & has_img
            if drop_mask.any():
                v = v.clone()
                v[drop_mask] = self.v_missing
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

    def self_adversarial_loss(self, pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
        """
        pos_logits: [B]
        neg_logits: [B * neg_ratio]
        """
        B = pos_logits.size(0)
        neg = neg_logits.view(B, self.neg_ratio)  # [B,K]

        # -log(sigmoid(pos))
        pos_loss = F.softplus(-pos_logits)

        # softmax(alpha * neg) with detached weights
        with torch.no_grad():
            w = F.softmax(self.adv_temperature * neg, dim=1)  # [B,K]

        # sum_i w_i * -log(sigmoid(-neg_i))
        neg_loss = (w * F.softplus(neg)).sum(dim=1)

        return (pos_loss + neg_loss).mean()

    def forward(self, pos_triples: torch.LongTensor, neg_triples: torch.LongTensor) -> torch.Tensor:
        """
        pos_triples: [B,3], neg_triples: [B*neg_ratio,3]
        """
        pos_scores = self.score(pos_triples)
        neg_scores = self.score(neg_triples)

        main_loss = self.self_adversarial_loss(pos_scores, neg_scores)
        l2 = 1e-6 * self.entity_residual.weight.pow(2).mean()
        scale = F.softplus(self.residual_scale)
        scale_l2 = 1e-4 * scale.pow(2)
        loss = main_loss + l2 + scale_l2
        return loss
