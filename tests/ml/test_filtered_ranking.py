from __future__ import annotations

import torch

from eval.filtered_ranking import filtered_ranking_eval, prepare_true_tails_index


class TailIdScorer:
    def eval(self) -> None:
        return None

    def score(self, triples: torch.Tensor) -> torch.Tensor:
        return triples[:, 2].float()


def test_prepare_true_tails_index_sorts_cpu_tensors() -> None:
    prepared = prepare_true_tails_index({(0, 0): {3, 1, 2}})

    assert prepared[(0, 0)].tolist() == [1, 2, 3]
    assert prepared[(0, 0)].device.type == "cpu"


def test_filtered_ranking_filters_other_true_tails() -> None:
    triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
    true_tails = {(0, 0): {1, 2}}

    metrics = filtered_ranking_eval(
        TailIdScorer(),
        triples,
        true_tails=true_tails,
        true_heads={},
        num_entities=3,
        chunk_size=2,
        query_batch_size=1,
        device="cpu",
        ks=(1, 3),
    )

    assert metrics["mrr"] == 1.0
    assert metrics["hits@1"] == 1.0
    assert metrics["hits@3"] == 1.0
