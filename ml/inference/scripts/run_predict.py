import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.inference.runtime import load_predictor
from ml.inference.utils import to_pretty_json


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run inference tasks from a trained run directory.")
    ap.add_argument("--run_dir", required=True, help="Run directory containing config_merged.json and best.ckpt")
    ap.add_argument("--task", required=True, choices=["tail", "attr", "entity", "similar"])
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps | auto")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--chunk_size", type=int, default=None)

    ap.add_argument("--head", help="Entity id or token, e.g. 12 or ent_12")
    ap.add_argument("--relation", help="Relation id or token, e.g. 4 or rel_4")
    ap.add_argument("--entity", help="Entity id or token, e.g. 12 or ent_12")
    ap.add_argument(
        "--relations",
        nargs="*",
        default=None,
        help="Optional relation ids/tokens for attribute completion; if omitted, predictor uses default frequent relations",
    )
    ap.add_argument("--space", default="fused", help="Similarity space: text | image | fused | entity_repr")

    return ap


def main():
    args = build_arg_parser().parse_args()
    predictor = load_predictor(run_dir=args.run_dir, device=args.device)

    if args.task == "tail":
        if args.head is None or args.relation is None:
            raise ValueError("--head and --relation are required for task=tail")
        result = predictor.predict_tail(
            head_id=args.head,
            rel_id=args.relation,
            topk=args.topk,
            chunk_size=args.chunk_size,
        )
    elif args.task == "attr":
        if args.entity is None:
            raise ValueError("--entity is required for task=attr")
        result = predictor.complete_attributes(
            entity_id=args.entity,
            relation_ids=args.relations,
            topk=args.topk,
            chunk_size=args.chunk_size,
        )
    elif args.task == "entity":
        if args.entity is None:
            raise ValueError("--entity is required for task=entity")
        result = predictor.get_entity_multimodal(entity_id=args.entity)
    else:
        if args.entity is None:
            raise ValueError("--entity is required for task=similar")
        result = predictor.similar_entities(
            entity_id=args.entity,
            topk=args.topk,
            space=args.space,
            chunk_size=args.chunk_size,
        )

    print(to_pretty_json(result))


if __name__ == "__main__":
    main()
