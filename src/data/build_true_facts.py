from collections import defaultdict

def build_true_facts(all_triples):
    """
    all_triples: iterable of (h,r,t) int tuples
    returns:
      true_tails: dict[(h,r)] -> set(t)
      true_heads: dict[(r,t)] -> set(h)
    """
    true_tails = defaultdict(set)
    true_heads = defaultdict(set)
    for h, r, t in all_triples:
        true_tails[(h, r)].add(t)
        true_heads[(r, t)].add(h)
    return true_tails, true_heads