import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, d: int, use_layernorm: bool = True):
        super().__init__()
        self.d = d
        self.use_ln = use_layernorm
        if use_layernorm:
            self.ln_t = nn.LayerNorm(d)
            self.ln_v = nn.LayerNorm(d)
        self.gate = nn.Linear(2 * d, 1)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, t: torch.Tensor, v: torch.Tensor):
        """
        t: [B,d], v: [B,d]
        returns:
          z: [B,d], g: [B,1]
        """
        if self.use_ln:
            t = self.ln_t(t)
            v = self.ln_v(v)
        g = torch.sigmoid(self.gate(torch.cat([t, v], dim=-1)))  # [B,1]
        z = g * t + (1.0 - g) * v
        return z, g