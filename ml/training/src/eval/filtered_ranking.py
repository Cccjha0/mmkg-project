import torch

def prepare_true_tails_index(true_tails: dict) -> dict:
    """
    Convert filtering map to sorted CPU LongTensor values once.
    Input:
      true_tails[(h, r)] -> set/list/tensor of true tail ids
    Output:
      dict[(h, r)] -> sorted LongTensor on CPU
    """
    out = {}
    for k, v in true_tails.items():
        if isinstance(v, torch.Tensor):
            vt = v.detach().cpu().to(dtype=torch.long)
            if vt.numel() > 1:
                vt = torch.sort(vt).values
        else:
            vt = torch.tensor(sorted(v), dtype=torch.long) if len(v) > 0 else torch.empty(0, dtype=torch.long)
        out[k] = vt
    return out


@torch.inference_mode()
def filtered_ranking_eval(
        model,
        triples: torch.LongTensor,          # [N,3] on CPU
        true_tails: dict,
        true_heads: dict,
        num_entities: int,
        chunk_size: int = 10000,
        query_batch_size: int = 1,
        device: str = "cuda",
        ks=(1, 3, 10),
):
    """
    model.score(triples) -> scores (higher is better)
    We compute filtered ranks for tail prediction only: (h, r, ?).
    """

    model.eval()
    device = torch.device(device)

    mrr_sum = 0.0
    hits = {k: 0 for k in ks}
    count = 0

    # Precreate all entity ids once (on CPU), chunk later
    all_entities = torch.arange(num_entities, dtype=torch.long)
    neg_inf = float("-inf")

    # Accept preprocessed map directly to avoid repeating this cost every eval.
    if len(true_tails) > 0 and isinstance(next(iter(true_tails.values())), torch.Tensor):
        true_tails_t = true_tails
    else:
        true_tails_t = prepare_true_tails_index(true_tails)

    n = triples.size(0)
    for q_start in range(0, n, query_batch_size):
        q_end = min(n, q_start + query_batch_size)
        q = triples[q_start:q_end]  # [Bq,3] on CPU
        bq = q.size(0)

        h = q[:, 0].to(device)
        r = q[:, 1].to(device)
        t = q[:, 2].to(device)
        t_cpu = q[:, 2]  # [Bq] CPU

        # target score is just score(h,r,t) and can be computed directly once
        target_scores = model.score(torch.stack([h, r, t], dim=1))  # [Bq]
        target = target_scores.unsqueeze(1)  # [Bq,1]

        # filtered tail ids per query on CPU (sorted long tensors), excluding target
        filt_excl_list = []
        for j in range(bq):
            key = (int(q[j, 0].item()), int(q[j, 1].item()))
            filt_idx = true_tails_t.get(key, torch.empty(0, dtype=torch.long))
            if filt_idx.numel() > 0:
                filt_idx = filt_idx[filt_idx != int(t_cpu[j].item())]
            filt_excl_list.append(filt_idx)

        greater = torch.zeros(bq, device=device, dtype=torch.long)

        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            c = end - start
            cand = all_entities[start:end].to(device)  # [C]

            h_g = h.unsqueeze(1).expand(bq, c)
            r_g = r.unsqueeze(1).expand(bq, c)
            t_g = cand.unsqueeze(0).expand(bq, c)
            batch = torch.stack([h_g.reshape(-1), r_g.reshape(-1), t_g.reshape(-1)], dim=1)

            scores = model.score(batch).view(bq, c)  # [Bq,C]

            # vectorized masked assignment: build row/col indices, then assign once
            row_chunks = []
            col_chunks = []
            for j in range(bq):
                filt_idx = filt_excl_list[j]
                if filt_idx.numel() == 0:
                    continue
                l = int(torch.searchsorted(filt_idx, start, right=False).item())
                rr = int(torch.searchsorted(filt_idx, end, right=False).item())
                local = filt_idx[l:rr]
                if local.numel() == 0:
                    continue
                row_chunks.append(torch.full((local.numel(),), j, dtype=torch.long))
                col_chunks.append(local - start)

            if row_chunks:
                rows = torch.cat(row_chunks, dim=0).to(device)
                cols = torch.cat(col_chunks, dim=0).to(device)
                scores[rows, cols] = neg_inf

            greater += (scores > target).sum(dim=1)

        rank_tail = greater + 1  # [Bq]

        # ---------- accumulate metrics ----------
        mrr_sum += float((1.0 / rank_tail.float()).sum().item())
        for k in ks:
            hits[k] += int((rank_tail <= k).sum().item())
        count += bq

    mrr = mrr_sum / count
    out = {"mrr": mrr}
    for k in ks:
        out[f"hits@{k}"] = hits[k] / count
    return out
