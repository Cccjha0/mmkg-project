import torch

@torch.no_grad()
def filtered_ranking_eval(
        model,
        triples: torch.LongTensor,          # [N,3] on CPU
        true_tails: dict,
        true_heads: dict,
        num_entities: int,
        chunk_size: int = 10000,
        device: str = "cuda",
        ks=(1, 3, 10),
):
    """
    model.score(triples) -> scores (higher is better)
    We compute filtered ranks for both head and tail prediction.
    """

    model.eval()
    device = torch.device(device)

    mrr_sum = 0.0
    hits = {k: 0 for k in ks}
    count = 0

    # Precreate all entity ids once (on CPU), chunk later
    all_entities = torch.arange(num_entities, dtype=torch.long)

    for (h, r, t) in triples.tolist():
        # ---------- Tail prediction: (h, r, ?) ----------
        target_score = None
        greater = 0
        equal = 0  # optional tie handling

        # filtered set for (h,r)
        filt = true_tails.get((h, r), set())

        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            cand = all_entities[start:end].clone()

            # apply filtering: remove other true tails by setting score=-inf
            # keep target t
            mask = torch.zeros_like(cand, dtype=torch.bool)
            # mark filtered candidates
            for tt in filt:
                if start <= tt < end and tt != t:
                    mask[tt - start] = True

            # build batch triples [B,3]
            batch = torch.stack([
                torch.full((end-start,), h, dtype=torch.long),
                torch.full((end-start,), r, dtype=torch.long),
                cand
            ], dim=1).to(device)

            scores = model.score(batch).detach().cpu()  # [B]
            scores[mask] = -1e30

            # capture target score if within chunk
            if start <= t < end:
                target_score = scores[t - start].item()

            # count how many candidates have score > target (we'll finalize after target_score known)
            # can't do until target_score known; so we store chunk scores if needed
            # simplest: after we know target_score, do a second pass (slow).
            # better: compute rank in one pass by using target_score from chunk containing t.
            if target_score is not None:
                greater += int((scores > target_score).sum().item())

        rank_tail = greater + 1

        # ---------- Head prediction: (?, r, t) ----------
        target_score = None
        greater = 0

        filt = true_heads.get((r, t), set())

        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            cand = all_entities[start:end].clone()

            mask = torch.zeros_like(cand, dtype=torch.bool)
            for hh in filt:
                if start <= hh < end and hh != h:
                    mask[hh - start] = True

            batch = torch.stack([
                cand,
                torch.full((end-start,), r, dtype=torch.long),
                torch.full((end-start,), t, dtype=torch.long),
            ], dim=1).to(device)

            scores = model.score(batch).detach().cpu()
            scores[mask] = -1e30

            if start <= h < end:
                target_score = scores[h - start].item()

            if target_score is not None:
                greater += int((scores > target_score).sum().item())

        rank_head = greater + 1

        # ---------- accumulate metrics ----------
        for rank in (rank_tail, rank_head):
            mrr_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1
            count += 1

    mrr = mrr_sum / count
    out = {"mrr": mrr}
    for k in ks:
        out[f"hits@{k}"] = hits[k] / count
    return out