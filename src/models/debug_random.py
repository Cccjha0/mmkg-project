import torch
import torch.nn as nn

class RandomScorer(nn.Module):
    def __init__(self, seed=1):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        self.g = g

    @torch.no_grad()
    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        # return random scores
        return torch.rand(triples.size(0), generator=self.g, device=triples.device)