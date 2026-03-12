import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ml.inference.utils import (
    build_response,
    format_entity_id,
    format_relation_id,
    infer_text_map_paths,
    load_tsv_map,
    parse_entity_id,
    parse_relation_id,
)


class Predictor:
    def __init__(
        self,
        model,
        cfg: dict[str, Any],
        device: str,
        num_entities: int,
        run_dir: str | Path,
    ):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_entities = int(num_entities)
        self.run_dir = Path(run_dir)
        self.model_name = cfg.get("model", {}).get("name", type(model).__name__)
        self.default_chunk_size = int(cfg.get("evaluation", {}).get("chunk_size", 4096))
        self.entity_text_map, self.relation_text_map = self._load_text_maps()
        self.default_attribute_relations = self._load_default_attribute_relations()

    def parse_entity(self, value: int | str) -> int:
        return parse_entity_id(value)

    def parse_relation(self, value: int | str) -> int:
        return parse_relation_id(value)

    def format_entity(self, entity_id: int) -> str:
        return format_entity_id(entity_id)

    def format_relation(self, relation_id: int) -> str:
        return format_relation_id(relation_id)

    def entity_text(self, entity_id: int) -> str | None:
        return self.entity_text_map.get(self.format_entity(entity_id))

    def relation_text(self, relation_id: int) -> str | None:
        return self.relation_text_map.get(self.format_relation(relation_id))

    @torch.inference_mode()
    def predict_tail(
        self,
        head_id: int | str,
        rel_id: int | str,
        *,
        topk: int = 10,
        chunk_size: int | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        head = self.parse_entity(head_id)
        rel = self.parse_relation(rel_id)
        chunk = int(chunk_size or self.default_chunk_size)

        scores_parts = []
        for s in range(0, self.num_entities, chunk):
            e = min(self.num_entities, s + chunk)
            cand = torch.arange(s, e, dtype=torch.long, device=self.device)
            h = torch.full((cand.numel(),), head, dtype=torch.long, device=self.device)
            r = torch.full((cand.numel(),), rel, dtype=torch.long, device=self.device)
            triples = torch.stack([h, r, cand], dim=1)
            scores_parts.append(self.model.score(triples).detach().cpu())

        scores = torch.cat(scores_parts, dim=0)
        k = min(int(topk), self.num_entities)
        top_scores, top_indices = torch.topk(scores, k=k, largest=True)
        latency_ms = (time.perf_counter() - start) * 1000.0

        results = []
        for entity_id, score in zip(top_indices.tolist(), top_scores.tolist()):
            results.append(
                {
                    "entity_id": entity_id,
                    "entity": self.format_entity(entity_id),
                    "entity_text": self.entity_text(entity_id),
                    "score": float(score),
                }
            )

        return build_response(
            task="tail",
            model=self.model_name,
            device=self.device.type,
            inputs={
                "head_id": head,
                "head": self.format_entity(head),
                "head_text": self.entity_text(head),
                "relation_id": rel,
                "relation": self.format_relation(rel),
                "relation_text": self.relation_text(rel),
            },
            results=results,
            latency_ms=latency_ms,
        )

    @torch.inference_mode()
    def predict_tail_batch(
        self,
        pairs: list[tuple[int | str, int | str]],
        *,
        topk: int = 10,
        chunk_size: int | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        queries = [
            self.predict_tail(head_id=h, rel_id=r, topk=topk, chunk_size=chunk_size)
            for h, r in pairs
        ]
        latency_ms = (time.perf_counter() - start) * 1000.0
        return build_response(
            task="tail_batch",
            model=self.model_name,
            device=self.device.type,
            inputs={
                "num_queries": len(pairs),
                "topk": int(topk),
                "chunk_size": int(chunk_size or self.default_chunk_size),
            },
            results=queries,
            latency_ms=latency_ms,
        )

    @torch.inference_mode()
    def complete_attributes(
        self,
        entity_id: int | str,
        *,
        relation_ids: list[int | str] | None = None,
        topk: int = 5,
        chunk_size: int | None = None,
        max_relations: int = 5,
    ) -> dict[str, Any]:
        entity = self.parse_entity(entity_id)
        rel_values = relation_ids or self.default_attribute_relations[:max_relations]
        outputs = []
        for rel in rel_values:
            pred = self.predict_tail(entity, rel, topk=topk, chunk_size=chunk_size)
            outputs.append(pred)
        return build_response(
            task="attr",
            model=self.model_name,
            device=self.device.type,
            inputs={
                "entity_id": entity,
                "entity": self.format_entity(entity),
                "entity_text": self.entity_text(entity),
                "relation_ids": [self.parse_relation(rel) for rel in rel_values],
                "relations": [self.format_relation(self.parse_relation(rel)) for rel in rel_values],
                "relation_texts": [self.relation_text(self.parse_relation(rel)) for rel in rel_values],
            },
            results=outputs,
        )

    @torch.inference_mode()
    def get_entity_multimodal(self, entity_id: int | str) -> dict[str, Any]:
        entity = self.parse_entity(entity_id)
        entity_token = self.format_entity(entity)
        results = {
            "entity_id": entity,
            "entity": entity_token,
            "entity_text": self.entity_text(entity),
            "has_text_embedding": bool(hasattr(self.model, "text_emb") or hasattr(self.model, "text_base")),
            "has_image_embedding": bool(hasattr(self.model, "img_emb")),
            "has_image": self._has_image(entity),
            "image_path": self._resolve_image_path(entity),
            "available_spaces": self._available_spaces(entity),
            "embedding_summary": self._embedding_summary(entity),
        }

        fused_summary = self._fused_summary(entity)
        if fused_summary is not None:
            results["fused_summary"] = fused_summary

        gate_summary = self._gate_summary(entity)
        if gate_summary is not None:
            results["gate_summary"] = gate_summary

        out = build_response(
            task="entity",
            model=self.model_name,
            device=self.device.type,
            inputs={
                "entity_id": entity,
                "entity": entity_token,
                "entity_text": self.entity_text(entity),
            },
            results=results,
        )
        return out

    @torch.inference_mode()
    def similar_entities(
        self,
        entity_id: int | str,
        *,
        topk: int = 10,
        space: str = "fused",
        chunk_size: int | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        entity = self.parse_entity(entity_id)
        chunk = int(chunk_size or self.default_chunk_size)

        query = self._entity_vector(entity, space=space).detach().cpu()
        query = F.normalize(query.unsqueeze(0), dim=1).squeeze(0)

        sims_parts = []
        for s in range(0, self.num_entities, chunk):
            e = min(self.num_entities, s + chunk)
            ids = torch.arange(s, e, dtype=torch.long, device=self.device)
            mat = self._entity_matrix(ids, space=space).detach().cpu()
            mat = F.normalize(mat, dim=1)
            sims_parts.append(torch.mv(mat, query))

        sims = torch.cat(sims_parts, dim=0)
        sims[entity] = float("-inf")
        k = min(int(topk), max(0, self.num_entities - 1))
        top_scores, top_indices = torch.topk(sims, k=k, largest=True)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return build_response(
            task="similar",
            model=self.model_name,
            device=self.device.type,
            inputs={
                "entity_id": entity,
                "entity": self.format_entity(entity),
                "entity_text": self.entity_text(entity),
                "space": space,
            },
            results=[
                {
                    "entity_id": idx,
                    "entity": self.format_entity(idx),
                    "entity_text": self.entity_text(idx),
                    "score": float(score),
                }
                for idx, score in zip(top_indices.tolist(), top_scores.tolist())
            ],
            latency_ms=latency_ms,
        )

    def _load_text_maps(self) -> tuple[dict[str, str], dict[str, str]]:
        entity_file, relation_file = infer_text_map_paths(self.cfg.get("dataset", {}).get("train"))
        return load_tsv_map(entity_file), load_tsv_map(relation_file)

    def _load_default_attribute_relations(self) -> list[str]:
        configured = self.cfg.get("inference", {}).get("attribute_relations")
        if configured:
            return [self.format_relation(self.parse_relation(rel)) for rel in configured]
        return self._discover_default_attribute_relations()

    def _discover_default_attribute_relations(self, topn: int = 10) -> list[str]:
        train_path = self.cfg.get("dataset", {}).get("train")
        if not train_path:
            return []

        rel_counter: Counter[int] = Counter()
        with Path(train_path).open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                rel = parts[1].strip()
                try:
                    rel_counter[self.parse_relation(rel)] += 1
                except ValueError:
                    continue

        return [self.format_relation(rel_id) for rel_id, _ in rel_counter.most_common(topn)]

    def _entity_vector(self, entity_id: int, *, space: str) -> torch.Tensor:
        ids = torch.tensor([entity_id], dtype=torch.long, device=self.device)
        return self._entity_matrix(ids, space=space)[0]

    def _has_image(self, entity_id: int) -> bool | None:
        if hasattr(self.model, "has_img"):
            return bool(self.model.has_img[entity_id].detach().cpu().item())
        return None

    def _resolve_image_path(self, entity_id: int) -> str | None:
        train_path = self.cfg.get("dataset", {}).get("train")
        if not train_path:
            return None

        raw_dir = Path(train_path).parent
        images_root = raw_dir / "OpenBG-IMG_images"
        if not images_root.is_dir():
            return None

        entity_token = self.format_entity(entity_id)
        image_path = images_root / entity_token / "image_0.jpg"
        if image_path.is_file():
            return str(image_path)
        return None

    def _available_spaces(self, entity_id: int) -> list[str]:
        spaces = []
        for space in ("text", "image", "fused", "entity_repr"):
            try:
                self._entity_vector(entity_id, space=space)
            except Exception:
                continue
            spaces.append(space)
        return spaces

    def _embedding_summary(self, entity_id: int) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for space in ("text", "image", "fused", "entity_repr"):
            try:
                vec = self._entity_vector(entity_id, space=space).detach().cpu()
            except Exception:
                continue
            summary[space] = {
                "dim": int(vec.numel()),
                "l2_norm": float(vec.norm(p=2).item()),
                "mean": float(vec.mean().item()),
                "std": float(vec.std(unbiased=False).item()) if vec.numel() > 1 else 0.0,
            }
        return summary

    def _fused_summary(self, entity_id: int) -> dict[str, Any] | None:
        try:
            vec = self._entity_vector(entity_id, space="fused").detach().cpu()
        except Exception:
            return None
        return {
            "dim": int(vec.numel()),
            "l2_norm": float(vec.norm(p=2).item()),
        }

    def _gate_summary(self, entity_id: int) -> dict[str, Any] | None:
        if not hasattr(self.model, "gate_for_entities"):
            return None
        eids = torch.tensor([entity_id], dtype=torch.long, device=self.device)
        gate = self.model.gate_for_entities(eids).detach().cpu()
        return {
            "mean_gate": float(gate.mean().item()),
        }

    def _entity_matrix(self, entity_ids: torch.Tensor, *, space: str) -> torch.Tensor:
        if space == "text":
            if hasattr(self.model, "text_emb"):
                return self.model.text_emb[entity_ids]
            if hasattr(self.model, "text_base") and self.model.text_base is not None:
                return self.model.text_proj(self.model.text_base[entity_ids].to(self.device))
        elif space == "image":
            if hasattr(self.model, "_entity_image"):
                return self.model._entity_image(entity_ids)
            if hasattr(self.model, "img_emb"):
                return self.model.img_emb[entity_ids]
        elif space == "fused":
            if hasattr(self.model, "_fused"):
                return self.model._fused(entity_ids)
            if hasattr(self.model, "_fused_with_r"):
                rids = torch.zeros(entity_ids.size(0), dtype=torch.long, device=self.device)
                fused, _ = self.model._fused_with_r(entity_ids, rids)
                return fused
        elif space == "entity_repr":
            if hasattr(self.model, "_encode_all_entities"):
                return self.model._encode_all_entities()[entity_ids]

        raise ValueError(f"Unsupported similarity space '{space}' for model {self.model_name}.")
