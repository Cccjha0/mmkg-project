import argparse
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.inference.runtime import load_predictor
from ml.inference.utils import to_pretty_json


def measure(fn, *, warmup: int, repeats: int):
    for _ in range(max(0, warmup)):
        fn()

    latencies = []
    last_output = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        last_output = fn()
        latencies.append((time.perf_counter() - start) * 1000.0)

    return {
        "runs": len(latencies),
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
        "sample_output": last_output,
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark inference tasks for a trained run directory.")
    ap.add_argument("--run_dir", required=True, help="Run directory containing config_merged.json and best.ckpt")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps | auto")
    ap.add_argument("--entity", default="ent_000000", help="Benchmark entity token or id")
    ap.add_argument("--relation", default="rel_0000", help="Benchmark relation token or id")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--chunk_size", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for tail batch benchmark")
    args = ap.parse_args()

    predictor = load_predictor(run_dir=args.run_dir, device=args.device)
    batch_pairs = [(args.entity, args.relation) for _ in range(max(1, args.batch_size))]

    result = {
        "run_dir": str(Path(args.run_dir)),
        "model": predictor.model_name,
        "device": predictor.device.type,
        "settings": {
            "entity": args.entity,
            "relation": args.relation,
            "topk": args.topk,
            "chunk_size": args.chunk_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "batch_size": args.batch_size,
        },
        "benchmarks": {
            "tail": measure(
                lambda: predictor.predict_tail(
                    head_id=args.entity,
                    rel_id=args.relation,
                    topk=args.topk,
                    chunk_size=args.chunk_size,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "tail_batch": measure(
                lambda: predictor.predict_tail_batch(
                    pairs=batch_pairs,
                    topk=args.topk,
                    chunk_size=args.chunk_size,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "attr": measure(
                lambda: predictor.complete_attributes(
                    entity_id=args.entity,
                    topk=min(args.topk, 3),
                    chunk_size=args.chunk_size,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "entity": measure(
                lambda: predictor.get_entity_multimodal(entity_id=args.entity),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "similar": measure(
                lambda: predictor.similar_entities(
                    entity_id=args.entity,
                    topk=args.topk,
                    chunk_size=args.chunk_size,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
        },
    }

    print(to_pretty_json(result))


if __name__ == "__main__":
    main()
