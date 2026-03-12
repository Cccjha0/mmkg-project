import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.inference.runtime import load_predictor
from ml.inference.utils import to_pretty_json


def main():
    ap = argparse.ArgumentParser(description="Run a small OpenBG-IMG inference demo.")
    ap.add_argument("--run_dir", required=True, help="Run directory containing config_merged.json and best.ckpt")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps | auto")
    ap.add_argument("--entity", default="ent_0", help="Demo entity token or id")
    ap.add_argument("--relation", default="rel_0", help="Demo relation token or id")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--chunk_size", type=int, default=None)
    args = ap.parse_args()

    predictor = load_predictor(run_dir=args.run_dir, device=args.device)

    outputs = {
        "tail_demo": predictor.predict_tail(
            head_id=args.entity,
            rel_id=args.relation,
            topk=args.topk,
            chunk_size=args.chunk_size,
        ),
        "attr_demo": predictor.complete_attributes(
            entity_id=args.entity,
            topk=min(args.topk, 3),
            chunk_size=args.chunk_size,
        ),
        "entity_demo": predictor.get_entity_multimodal(entity_id=args.entity),
        "similar_demo": predictor.similar_entities(
            entity_id=args.entity,
            topk=args.topk,
            chunk_size=args.chunk_size,
        ),
    }

    print(to_pretty_json(outputs))


if __name__ == "__main__":
    main()
