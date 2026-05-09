import collections
import json
import os
import pathlib
import pickle

from functools import lru_cache

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["Search", "create_app", "save_metadata"]


# =====================================================
# save metadata
# =====================================================

def save_metadata(origin, source):

    with open(origin, "r", encoding="utf-8") as f:

        metadata = json.load(f)

    with open(source, "w", encoding="utf-8") as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )


class SimpleTfIdfRetriever:
    def __init__(self, documents, vectorizer):
        self.documents = documents
        self.vectorizer = vectorizer
        labels = [doc["label"] for doc in documents]
        self.matrix = vectorizer.fit_transform(labels) if labels else None

    def __call__(self, query: str):
        if self.matrix is None or not query:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1]

        return [
            self.documents[index]
            for index in ranked_indices
            if scores[index] > 0
        ]


# =====================================================
# Search
# =====================================================

class Search:
    MAX_GRAPH_NODES = 250
    MAX_GRAPH_LINKS = 500
    NEIGHBORS_PER_PRUNE_STEP = 50

    def __init__(self, file: str) -> None:

        self.colors = [
            "#00A36C",
            "#9370DB",
            "#bbae98",
            "#7393B3",
            "#677179",
            "#318ce7",
            "#088F8F",
        ]

        self.metadata = {}

        triples = pd.read_csv(
            file,
            header=None
        )

        self.triples = collections.defaultdict(tuple)

        self.relations = collections.defaultdict(list)

        # ==========================================
        # build graph
        # ==========================================

        for h, r, t in triples.to_records(index=False).tolist():

            h = str(h)
            r = str(r)
            t = str(t)

            self.triples[h] += tuple([t])

            self.triples[t] += tuple([h])

            self.relations[f"{h}_{t}"].append(r)

        self.explore.cache_clear()

    # =====================================================
    # save pickle
    # =====================================================

    def save(self, path):

        with open(path, "wb") as f:

            pickle.dump(self, f)

        return self

    # =====================================================
    # load metadata
    # =====================================================

    def load_metadata(self, path):

        with open(path, "r", encoding="utf-8") as f:

            self.metadata = json.load(f)

        documents = []

        # ==========================================
        # only entity searchable
        # ==========================================

        for entity_id, meta in self.metadata.items():

            if entity_id.startswith("ent_"):

                # Add both Chinese and English labels for searching
                label_zh = meta.get("label_zh", "")
                label_en = meta.get("label_en", "")

                # Add document with Chinese label
                if label_zh:
                    documents.append(
                        {
                            "key": entity_id,
                            "label": label_zh,
                        }
                    )

                # Add document with English label (if different)
                if label_en and label_en != label_zh:
                    documents.append(
                        {
                            "key": entity_id,
                            "label": label_en,
                        }
                    )

        # ==========================================
        # TF-IDF search
        # ==========================================

        self.retriever = SimpleTfIdfRetriever(
            documents=documents,
            vectorizer=TfidfVectorizer(
                lowercase=True,
                analyzer="char",
                ngram_range=(2, 5),
            ),
        )

        return self

    # =====================================================
    # graph explore
    # =====================================================

    @lru_cache(maxsize=10000)

    def explore(
            self,
            entities,
            neighbours,
            entity,
            depth,
            max_depth
    ):

        depth += 1

        for neighbour in neighbours:

            entities += tuple([tuple([entity, neighbour])])

            if depth < max_depth:

                entities = self.explore(
                    entities=entities,
                    neighbours=self.triples.get(
                        neighbour,
                        tuple([])
                    ),
                    entity=neighbour,
                    depth=depth,
                    max_depth=max_depth,
                )

        return entities

    # =====================================================
    # relation label
    # =====================================================

    def get_relation_label(self, relation_id, lang="en"):

        meta = self.metadata.get(
            relation_id,
            {}
        )

        if lang == "zh":
            return meta.get(
                "label_zh",
                meta.get("label", relation_id)
            )
        else:
            return meta.get(
                "label_en",
                meta.get("label", relation_id)
            )

    def explore_limited(
            self,
            entity_id: str,
            max_depth: int,
            prune: int,
            max_nodes: int = MAX_GRAPH_NODES,
            max_links: int = MAX_GRAPH_LINKS,
    ):
        max_depth = max(1, int(max_depth))
        max_nodes = max(1, int(max_nodes))
        max_links = max(1, int(max_links))
        per_node_limit = max(1, int(prune)) * self.NEIGHBORS_PER_PRUNE_STEP

        pairs = []
        visited_nodes = {entity_id}
        frontier = [entity_id]

        for _ in range(max_depth):
            next_frontier = []

            for current in frontier:
                neighbours = self.triples.get(current, tuple([]))[:per_node_limit]

                for neighbour in neighbours:
                    pairs.append((current, neighbour))

                    if neighbour not in visited_nodes:
                        visited_nodes.add(neighbour)
                        next_frontier.append(neighbour)

                    if len(visited_nodes) >= max_nodes or len(pairs) >= max_links:
                        return pairs

            frontier = next_frontier
            if not frontier:
                break

        return pairs

    # =====================================================
    # search call
    # =====================================================

    def __call__(self, query: str, k: int, n: int, p: int, lang: str = "en"):

        nodes = []

        links = []

        entities = {}

        h_r_t = {}

        candidates = []

        seen = {}

        # ==========================================
        # search entity label
        # ==========================================

        for q in query.split(";"):

            answer = self.retriever(q.strip())[: int(k)]

            for candidate in answer:

                if candidate["key"] not in seen:

                    candidates.append(candidate)

                    seen[candidate["key"]] = True

        # ==========================================
        # query nodes
        # ==========================================

        for group, e in enumerate(candidates):

            entity_id = e["key"]

            metadata = self.metadata.get(
                entity_id,
                {}
            )

            # Get labels based on language
            if lang == "zh":
                entity_label = metadata.get(
                    "label_zh",
                    metadata.get("label", entity_id)
                )
            else:
                entity_label = metadata.get(
                    "label_en",
                    metadata.get("label", entity_id)
                )

            nodes.append(
                {
                    "id": entity_id,
                    "label": entity_label,
                    "label_zh": metadata.get("label_zh", ""),
                    "label_en": metadata.get("label_en", ""),
                    "group": group,
                    "color": "#960018",
                    "metadata": metadata,
                }
            )

            entities[entity_id] = True

        # ==========================================
        # graph expand
        # ==========================================

        for group, e in enumerate(candidates):

            entity_id = e["key"]

            color = self.colors[
                group % len(self.colors)
                ]

            match = self.explore_limited(
                entity_id=entity_id,
                max_depth=n,
                prune=p,
            )

            for h, t in list(match):
                relation_source = h
                relation_target = t
                relation_ids = self.relations.get(f"{h}_{t}", [])

                if not relation_ids:
                    relation_ids = self.relations.get(f"{t}_{h}", [])
                    relation_source = t
                    relation_target = h

                if not relation_ids:
                    continue

                # ==================================
                # source node
                # ==================================

                if h not in entities:

                    metadata = self.metadata.get(h, {})

                    # Get labels based on language
                    if lang == "zh":
                        node_label = metadata.get(
                            "label_zh",
                            metadata.get("label", h)
                        )
                    else:
                        node_label = metadata.get(
                            "label_en",
                            metadata.get("label", h)
                        )

                    nodes.append(
                        {
                            "id": h,
                            "label": node_label,
                            "label_zh": metadata.get("label_zh", ""),
                            "label_en": metadata.get("label_en", ""),
                            "group": group,
                            "color": color,
                            "metadata": metadata,
                        }
                    )

                    entities[h] = True

                # ==================================
                # target node
                # ==================================

                if t not in entities:

                    metadata = self.metadata.get(t, {})

                    # Get labels based on language
                    if lang == "zh":
                        node_label = metadata.get(
                            "label_zh",
                            metadata.get("label", t)
                        )
                    else:
                        node_label = metadata.get(
                            "label_en",
                            metadata.get("label", t)
                        )

                    nodes.append(
                        {
                            "id": t,
                            "label": node_label,
                            "label_zh": metadata.get("label_zh", ""),
                            "label_en": metadata.get("label_en", ""),
                            "group": group,
                            "color": color,
                            "metadata": metadata,
                        }
                    )

                    entities[t] = True

                # ==================================
                # relation
                # ==================================

                for r in relation_ids:

                    relation_label = self.get_relation_label(r, lang)

                    relation_label_zh = self.get_relation_label(r, "zh")
                    relation_label_en = self.get_relation_label(r, "en")

                    edge_key = f"{relation_source}_{r}_{relation_target}"

                    if edge_key not in h_r_t:

                        links.append(
                            {
                                "source": relation_source,
                                "target": relation_target,
                                "value": 1,
                                "relation": relation_label,
                                "relation_zh": relation_label_zh,
                                "relation_en": relation_label_en,
                            }
                        )

                        h_r_t[edge_key] = True

        return {
            "nodes": nodes,
            "links": links,
        }


# =====================================================
# Flask App
# =====================================================

def create_app():
    from flask import Flask
    from flask import request
    from flask import send_from_directory
    from flask_cors import CORS
    from flask_cors import cross_origin

    app = Flask(__name__)

    CORS(app)

    # =====================================================
    # image route
    # =====================================================

    @app.route('/images/<entity_id>/<filename>')

    def serve_image(entity_id, filename):

        image_root = os.path.abspath(

            os.path.join(
                os.path.dirname(__file__),
                "../../../data/datasets/openbg_img/raw/OpenBG-IMG_images"
            )
        )

        folder = os.path.join(
            image_root,
            entity_id
        )

        return send_from_directory(
            folder,
            filename
        )

    # =====================================================
    # search route
    # =====================================================

    @app.route("/search/<k>/<n>/<p>/<query>")

    @cross_origin()

    def get(k, n, p, query):

        lang = request.args.get('lang', 'en')

        path = pathlib.Path(__file__).parent.joinpath(
            "./../data"
        )

        search_path = os.path.join(
            path,
            "search.pkl"
        )

        # ==========================================
        # load cache
        # ==========================================

        if os.path.exists(search_path):

            with open(search_path, "rb") as f:

                search = pickle.load(f)

        else:

            search = Search(
                file=os.path.join(path, "data.csv")
            ).save(search_path)

        # ==========================================
        # load metadata
        # ==========================================

        metadata_path = os.path.join(
            path,
            "metadata.json"
        )

        if os.path.exists(metadata_path):

            search = search.load_metadata(
                metadata_path
            )

        # ==========================================
        # return graph
        # ==========================================

        return json.dumps(

            search(
                query=query,
                k=int(k),
                n=int(n),
                p=int(p),
                lang=lang,
            )

        )

    return app
