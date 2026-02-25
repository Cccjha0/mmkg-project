import torch
import torch.nn as nn


class RandomScorer(nn.Module):
    def __init__(self, seed=1):
        super().__init__()
        torch.manual_seed(seed)

    @torch.no_grad()
    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        # return random scores directly on same device
        return torch.rand(triples.size(0), device=triples.device)