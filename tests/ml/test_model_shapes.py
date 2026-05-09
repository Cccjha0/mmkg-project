from __future__ import annotations

import torch

from ml.training.src.models.fusion.early import OpenBGImgEarlyLP
from ml.training.src.models.openbg_img_gated_lp import OpenBGImgGatedLP


def _embeddings(num_entities: int = 6, d: int = 8):
    generator = torch.Generator().manual_seed(7)
    text_emb = torch.randn(num_entities, d, generator=generator)
    img_emb = torch.randn(num_entities, d, generator=generator)
    has_img = torch.tensor([True, False, True, False, True, False])
    return text_emb, img_emb, has_img


def _triples():
    pos = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
    neg = torch.tensor([[0, 0, 2], [0, 0, 3], [2, 1, 4], [2, 1, 5]], dtype=torch.long)
    return pos, neg


def _assert_link_prediction_contract(model) -> None:
    pos, neg = _triples()

    scores = model.score(pos)
    loss = model(pos, neg)

    assert scores.shape == (2,)
    assert loss.shape == ()
    assert torch.isfinite(scores).all()
    assert torch.isfinite(loss)


def test_early_fusion_forward_handles_missing_images() -> None:
    text_emb, img_emb, has_img = _embeddings()
    model = OpenBGImgEarlyLP(
        text_emb=text_emb,
        img_emb=img_emb,
        has_img=has_img,
        num_relations=3,
        d=8,
        neg_ratio=2,
    )

    _assert_link_prediction_contract(model)


def test_gate_and_residual_forward_shapes() -> None:
    text_emb, img_emb, has_img = _embeddings()
    model = OpenBGImgGatedLP(
        text_emb=text_emb,
        img_emb=img_emb,
        has_img=has_img,
        num_relations=3,
        d=8,
        neg_ratio=2,
        use_fusion=True,
        use_residual=True,
    )

    _assert_link_prediction_contract(model)
    gate = model.gate_for_entities(torch.tensor([0, 1, 2], dtype=torch.long))
    assert gate.shape == (3,)
    assert torch.all((gate >= 0) & (gate <= 1))


def test_gate_only_forward_shapes() -> None:
    text_emb, img_emb, has_img = _embeddings()
    model = OpenBGImgGatedLP(
        text_emb=text_emb,
        img_emb=img_emb,
        has_img=has_img,
        num_relations=3,
        d=8,
        neg_ratio=2,
        use_fusion=True,
        use_residual=False,
    )

    _assert_link_prediction_contract(model)


def test_residual_only_forward_shapes() -> None:
    text_emb, img_emb, has_img = _embeddings()
    model = OpenBGImgGatedLP(
        text_emb=text_emb,
        img_emb=img_emb,
        has_img=has_img,
        num_relations=3,
        d=8,
        neg_ratio=2,
        use_fusion=False,
        use_residual=True,
    )

    _assert_link_prediction_contract(model)
    gate = model.gate_for_entities(torch.tensor([0, 1], dtype=torch.long))
    assert gate.tolist() == [0.0, 0.0]
