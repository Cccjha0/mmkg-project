import torch
import torch.nn as nn


class RelAwareGatedFusion(nn.Module):
    def __init__(self, d: int, num_relations: int, use_layernorm: bool = True):
        super().__init__()
        self.d = d
        self.use_ln = use_layernorm
        if use_layernorm:
            self.ln_t = nn.LayerNorm(d)
            self.ln_v = nn.LayerNorm(d)
        self.gate = nn.Linear(2 * d, d)
        self.rel_gate = nn.Embedding(num_relations, d)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.rel_gate.weight)

    def forward(self, t: torch.Tensor, v: torch.Tensor, r: torch.LongTensor):
        """
        t: [B,d], v: [B,d], r: [B]
        returns:
          z: [B,d], g: [B,d]
        """
        if self.use_ln:
            t = self.ln_t(t)
            v = self.ln_v(v)
        base = self.gate(torch.cat([t, v], dim=-1))  # [B,d]
        rbias = self.rel_gate(r)  # [B,d]
        g = torch.sigmoid(base + rbias)  # [B,d]
        z = g * t + (1.0 - g) * v
        return z, g
