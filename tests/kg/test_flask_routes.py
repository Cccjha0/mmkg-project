from __future__ import annotations

import json
from collections import defaultdict

from backend import flask_app


class FakeSearch:
    MAX_GRAPH_LINKS = 500

    def __init__(self, file: str) -> None:
        self.file = file
        self.metadata = {
            "ent_000000": {"label": "Pants", "label_zh": "裤子", "label_en": "Pants"},
        }
        for idx in range(1, 700):
            self.metadata[f"ent_{idx:06d}"] = {
                "label": f"Node {idx}",
                "label_zh": f"节点 {idx}",
                "label_en": f"Node {idx}",
            }
        self.triples = defaultdict(tuple)
        self.triples["ent_000000"] = tuple(f"ent_{idx:06d}" for idx in range(1, 700))
        self.relations = defaultdict(list)
        for idx in range(1, 700):
            self.relations[f"ent_000000_ent_{idx:06d}"] = ["rel_0001"]
        self.metadata["rel_0001"] = {"label": "category", "label_zh": "类别", "label_en": "category"}

    def save(self, path: str):
        return self

    def load_metadata(self, path: str):
        return self

    def get_relation_label(self, relation_id: str, lang: str = "en") -> str:
        meta = self.metadata.get(relation_id, {})
        return meta.get(f"label_{lang}", meta.get("label", relation_id))

    def __call__(self, query: str, k: int, n: int, p: int, lang: str = "en"):
        limit = min(self.MAX_GRAPH_LINKS, 10 if n < 2 else self.MAX_GRAPH_LINKS)
        nodes = [{"id": "ent_000000", "label": "Pants", "metadata": self.metadata["ent_000000"]}]
        links = []
        for idx in range(1, limit + 1):
            entity = f"ent_{idx:06d}"
            nodes.append({"id": entity, "label": f"Node {idx}", "metadata": self.metadata[entity]})
            links.append(
                {
                    "source": "ent_000000",
                    "target": entity,
                    "value": 1,
                    "relation": self.get_relation_label("rel_0001", lang),
                    "relation_zh": self.get_relation_label("rel_0001", "zh"),
                    "relation_en": self.get_relation_label("rel_0001", "en"),
                }
            )
        return {"nodes": nodes, "links": links}


def _client(monkeypatch):
    monkeypatch.setattr(flask_app, "Search", FakeSearch)
    monkeypatch.setattr(flask_app.os.path, "exists", lambda path: False)
    return flask_app.create_app().test_client()


def test_search_route_returns_graph_and_caps_dense_request(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.get("/search/1/2/999/Pants?lang=en")

    assert response.status_code == 200
    payload = json.loads(response.data.decode("utf-8"))
    assert payload["nodes"][0]["label"] == "Pants"
    assert len(payload["links"]) <= FakeSearch.MAX_GRAPH_LINKS
    assert len(payload["nodes"]) <= FakeSearch.MAX_GRAPH_LINKS + 1
    assert payload["links"][0]["relation_en"] == "category"


def test_node_connections_route_caps_dense_connections(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.get("/node_connections/ent_000000?lang=en")

    assert response.status_code == 200
    payload = json.loads(response.data.decode("utf-8"))
    assert payload["total"] == FakeSearch.MAX_GRAPH_LINKS
    assert len(payload["connections"]) == FakeSearch.MAX_GRAPH_LINKS
    assert payload["connections"][0]["relation"]["relation_en"] == "category"
