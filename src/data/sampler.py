import torch
import random

def negative_sample(pos_batch: torch.LongTensor, num_entities: int, neg_ratio: int = 10):
    """
    pos_batch: [B,3] on device
    returns neg_triples: [B*neg_ratio,3]
    Strategy: corrupt head/tail with 50/50
    (Unfiltered negatives for MVP. Later we can add filtered negatives.)
    """
    B = pos_batch.size(0)
    device = pos_batch.device

    # repeat positives
    neg = pos_batch.repeat_interleave(neg_ratio, dim=0)  # [B*neg_ratio,3]

    # decide corrupt head or tail
    flip = torch.rand(B * neg_ratio, device=device) < 0.5

    rand_ent = torch.randint(0, num_entities, (B * neg_ratio,), device=device)

    # corrupt head where flip=True else corrupt tail
    neg[flip, 0] = rand_ent[flip]
    neg[~flip, 2] = rand_ent[~flip]

    return neg