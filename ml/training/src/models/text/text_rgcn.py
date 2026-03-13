import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import RGCNConv
except ImportError as e:
    raise ImportError(
        "TextRGCN requires torch-geometric. Install it before running text_rgcn experiments."
    ) from e

from ml.training.src.models.decoders.complex import ComplEx


class TextRGCN(nn.Module):
    """
    True RGCN encoder over the training graph with a ComplEx decoder.
    The graph is built from training triples and message passing runs over
    all entities, then triples are scored from the resulting node states.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_type: torch.Tensor,
        text_emb_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 2,
        num_bases: int = 8,
        sample_neighbors: int | list[int] = 10,
        eval_on_cpu: bool = False,
        init_ent_emb: torch.Tensor | None = None,
    ):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("TextRGCN requires even hidden_dim for ComplEx decoding.")

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.conv_num_rels = num_relations * 2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_on_cpu = bool(eval_on_cpu)
        if isinstance(sample_neighbors, int):
            self.sample_neighbors = [sample_neighbors] * num_layers
        else:
            vals = list(sample_neighbors)
            if len(vals) == 1:
                vals = vals * num_layers
            if len(vals) != num_layers:
                raise ValueError(
                    f"sample_neighbors must have len 1 or num_layers={num_layers}, got {len(vals)}"
                )
            self.sample_neighbors = [int(v) for v in vals]

        self.entity_embeddings = nn.Embedding(num_entities, hidden_dim)
        self.text_proj = nn.Linear(text_emb_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.rgcn_layers = nn.ModuleList(
            [
                RGCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_relations=self.conv_num_rels,
                    num_bases=min(num_bases, self.conv_num_rels),
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = ComplEx(num_relations=num_relations, d=hidden_dim)

        self.register_buffer("edge_src", edge_src.long())
        self.register_buffer("edge_dst", edge_dst.long())
        self.register_buffer("edge_type", edge_type.long())
        sort_in = torch.argsort(self.edge_dst)
        self.register_buffer("in_src_sorted", self.edge_src[sort_in])
        self.register_buffer("in_dst_sorted", self.edge_dst[sort_in])
        self.register_buffer("in_type_sorted", self.edge_type[sort_in])
        in_counts = torch.bincount(self.in_dst_sorted, minlength=num_entities)
        self.register_buffer(
            "in_ptr",
            torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(in_counts, dim=0)]),
        )

        if init_ent_emb is not None:
            if init_ent_emb.dim() != 2:
                raise ValueError(f"init_ent_emb must be 2D, got shape={tuple(init_ent_emb.shape)}")
            if init_ent_emb.size(0) != num_entities or init_ent_emb.size(1) != text_emb_dim:
                raise ValueError(
                    "init_ent_emb shape mismatch: "
                    f"expected ({num_entities}, {text_emb_dim}), got {tuple(init_ent_emb.shape)}"
                )
            self.register_buffer("text_base", init_ent_emb.float())
        else:
            self.text_base = None

        self._edge_index_cache = None
        self._edge_index_cache_device = None
        self._eval_entity_cache = None

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

    def _invalidate_eval_cache(self):
        self._eval_entity_cache = None

    def train(self, mode: bool = True):
        super().train(mode)
        self._invalidate_eval_cache()
        return self

    def _get_edge_index(self, device: torch.device):
        if self._edge_index_cache is not None and self._edge_index_cache_device == str(device):
            return self._edge_index_cache

        edge_index = torch.stack([self.edge_src.to(device), self.edge_dst.to(device)], dim=0)
        self._edge_index_cache = edge_index
        self._edge_index_cache_device = str(device)
        return edge_index

    def _base_entity_features(self) -> torch.Tensor:
        x = self.entity_embeddings.weight
        if self.text_base is not None:
            x = x + self.text_proj(self.text_base.to(x.device))
        x = self.input_norm(x)
        return x

    def _encode_all_entities(self) -> torch.Tensor:
        if not self.training and self._eval_entity_cache is not None:
            return self._eval_entity_cache

        x = self._base_entity_features()
        edge_index = self._get_edge_index(x.device)
        rel = self.edge_type.to(x.device)

        for layer in self.rgcn_layers:
            x = layer(x, edge_index, rel)
            x = F.relu(x)
            x = self.dropout(x)

        if not self.training:
            self._eval_entity_cache = x
        return x

    def _sample_subgraph(self, seed_nodes: torch.Tensor):
        device = seed_nodes.device
        frontier = seed_nodes.detach().cpu().long().unique()
        all_nodes = [frontier]
        sampled_src = []
        sampled_dst = []
        sampled_type = []

        for k in range(self.num_layers):
            fanout = self.sample_neighbors[k]
            node_ids = frontier.tolist()
            starts = self.in_ptr[node_ids].cpu().numpy()
            ends = self.in_ptr[torch.tensor(node_ids) + 1].cpu().numpy()
            degs = ends - starts

            all_picks = []
            for start, end, deg in zip(starts, ends, degs):
                if deg <= 0:
                    continue
                if fanout > 0 and deg > fanout:
                    pick = torch.randperm(deg)[:fanout] + start
                else:
                    pick = torch.arange(start, end, dtype=torch.long)
                all_picks.append(pick)

            if not all_picks:
                break

            pick = torch.cat(all_picks)
            sampled_src.append(self.in_src_sorted[pick].cpu())
            sampled_dst.append(self.in_dst_sorted[pick].cpu())
            sampled_type.append(self.in_type_sorted[pick].cpu())

            next_frontier = self.in_src_sorted[pick].cpu()
            if next_frontier.numel() == 0:
                break
            frontier = next_frontier.long().unique()
            all_nodes.append(frontier)

        all_nodes = [x.cpu().long() for x in all_nodes]
        local_nodes = torch.cat(all_nodes, dim=0).unique()
        if sampled_src:
            sub_src = torch.cat(sampled_src)
            sub_dst = torch.cat(sampled_dst)
            sub_type = torch.cat(sampled_type)
        else:
            sub_src = torch.empty(0, dtype=torch.long)
            sub_dst = torch.empty(0, dtype=torch.long)
            sub_type = torch.empty(0, dtype=torch.long)

        node_map = torch.full((self.num_entities,), -1, dtype=torch.long)
        node_map[local_nodes] = torch.arange(local_nodes.numel(), dtype=torch.long)

        edge_index = torch.stack([node_map[sub_src], node_map[sub_dst]], dim=0) if sub_src.numel() > 0 else torch.empty((2, 0), dtype=torch.long)
        return (
            local_nodes.to(device),
            edge_index.to(device),
            sub_type.to(device),
            node_map.to(device),
        )

    def _encode_sampled_entities(self, seed_nodes: torch.Tensor):
        local_nodes, edge_index, edge_type, node_map = self._sample_subgraph(seed_nodes)
        x = self.entity_embeddings(local_nodes)
        if self.text_base is not None:
            x = x + self.text_proj(self.text_base[local_nodes].to(x.device))
        x = self.input_norm(x)

        for layer in self.rgcn_layers:
            if edge_index.size(1) > 0:
                x = layer(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)
        return x, local_nodes, node_map

    def score(self, triples: torch.Tensor) -> torch.Tensor:
        if not self.training and self.eval_on_cpu and triples.device.type == "cuda":
            return self._score_eval_cpu(triples)

        entity_repr = self._encode_all_entities()
        h = entity_repr[triples[:, 0].long()]
        r = triples[:, 1].long()
        t = entity_repr[triples[:, 2].long()]
        return self.decoder.score(h, r, t)

    @torch.inference_mode()
    def _score_eval_cpu(self, triples: torch.Tensor) -> torch.Tensor:
        cpu_triples = triples.detach().cpu()
        self._invalidate_eval_cache()
        cpu_repr = self._encode_all_entities_cpu()
        h = cpu_repr[cpu_triples[:, 0].long()]
        r = cpu_triples[:, 1].long()
        t = cpu_repr[cpu_triples[:, 2].long()]
        out = self.decoder.score(h, r, t)
        return out.to(triples.device)

    @torch.inference_mode()
    def _encode_all_entities_cpu(self) -> torch.Tensor:
        weight_device = self.entity_embeddings.weight.device
        orig_training = self.training

        self.train(False)
        self.cpu()
        try:
            self._invalidate_eval_cache()
            x = self._base_entity_features()
            edge_index = torch.stack([self.edge_src, self.edge_dst], dim=0)
            rel = self.edge_type
            for layer in self.rgcn_layers:
                x = layer(x, edge_index, rel)
                x = F.relu(x)
            return x
        finally:
            self.to(weight_device)
            self.train(orig_training)

    def forward(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor) -> torch.Tensor:
        self._invalidate_eval_cache()
        all_nodes = torch.cat(
            [pos_triples[:, 0], pos_triples[:, 2], neg_triples[:, 0], neg_triples[:, 2]], dim=0
        ).long().unique()
        entity_repr, _, node_map = self._encode_sampled_entities(all_nodes)

        def score_local(triples: torch.Tensor) -> torch.Tensor:
            h_idx = node_map[triples[:, 0].long()]
            t_idx = node_map[triples[:, 2].long()]
            h = entity_repr[h_idx]
            r = triples[:, 1].long()
            t = entity_repr[t_idx]
            return self.decoder.score(h, r, t)

        pos_scores = score_local(pos_triples)
        neg_scores = score_local(neg_triples)

        batch_size = pos_scores.size(0)
        neg_scores = neg_scores.view(batch_size, -1)
        pos_scores = pos_scores.unsqueeze(1)
        return -F.logsigmoid(pos_scores - neg_scores).mean()
