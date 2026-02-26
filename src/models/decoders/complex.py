import torch
import torch.nn as nn

class ComplEx(nn.Module):
    """
    ComplEx score: Re(<h, r, conj(t)>)
    Using real+imag parts concatenated in a single vector of dim d (must be even).
    """
    def __init__(self, num_relations: int, d: int):
        super().__init__()
        assert d % 2 == 0, "ComplEx requires even embedding dim"
        self.d = d
        self.k = d // 2
        self.rel = nn.Embedding(num_relations, d)
        nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h: torch.Tensor, r_id: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        h: [B,d], t: [B,d], r_id: [B]
        returns: [B]
        """
        r = self.rel(r_id)  # [B,d]

        hr, hi = h[:, :self.k], h[:, self.k:]
        rr, ri = r[:, :self.k], r[:, self.k:]
        tr, ti = t[:, :self.k], t[:, self.k:]

        # Re( (hr + i hi) * (rr + i ri) * conj(tr + i ti) )
        # = sum( (hr*rr - hi*ri)*tr + (hr*ri + hi*rr)*ti )
        re1 = hr * rr - hi * ri
        re2 = hr * ri + hi * rr
        score = (re1 * tr + re2 * ti).sum(dim=-1)
        return score