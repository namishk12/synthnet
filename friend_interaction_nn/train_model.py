from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from friendship_nn.cli_common import (
    add_data_arguments,
    add_pair_column_arguments,
    context_metadata,
    resolve_inputs,
)
from friendship_nn.features import FEATURE_NAMES, extract_pair_features
from friendship_nn.model import save_artifact, train_neural_network, write_json
from friendship_nn.pairs import load_pair_csv, prepare_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a supervised neural network to classify an unordered pair as "
            "friend-or-interacting from CDR/IPDR evidence."
        )
    )
    add_data_arguments(parser)
    parser.add_argument("--base-truth", required=True, help="CSV containing two people and a binary label.")
    add_pair_column_arguments(parser, include_label=True)
    parser.add_argument(
        "--output-dir",
        default="trained_model",
        help="Folder for model.pt, metrics, predictions, and extracted features.",
    )
    parser.add_argument("--epochs", type=int, default=250, help="Maximum neural-network epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Neural-network batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="AdamW learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cdr_schema, ipdr_schema, resolver = resolve_inputs(args)

    raw_pairs = load_pair_csv(
        args.base_truth,
        require_label=True,
        person_a_col=args.person_a_col,
        person_b_col=args.person_b_col,
        label_col=args.label_col,
    )
    pairs = prepare_pairs(raw_pairs, resolver)
    counts = pairs["label"].value_counts().to_dict()
    print(
        f"Base truth: {len(pairs):,} unique pairs "
        f"({int(counts.get(1, 0)):,} positive, {int(counts.get(0, 0)):,} negative).",
        flush=True,
    )
    if len(pairs) < 10:
        raise ValueError("Provide at least 10 unique labeled pairs.")
    if int(counts.get(1, 0)) < 3 or int(counts.get(0, 0)) < 3:
        raise ValueError("Provide at least 3 positive and 3 negative base-truth pairs.")
    if len(pairs) < 50:
        print(
            "WARNING: fewer than 50 labeled pairs; the program will run, but validation metrics may be unstable.",
            flush=True,
        )

    features, coverage = extract_pair_features(
        pairs,
        cdr_schema,
        ipdr_schema,
        resolver,
        chunk_size=args.chunk_size,
        progress=lambda message: print(message, flush=True),
    )
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature extraction produced non-finite values.")
    if float(np.abs(features.to_numpy()).sum()) == 0.0:
        raise ValueError(
            "No labeled identifier matched usable CDR/IPDR evidence. Check the pair identifier type "
            "and supply the matching subscribers.csv when mixing subscriber IDs and MSISDNs."
        )
    print(
        f"Coverage: {coverage['pairs_with_direct_cdr']:,}/{coverage['pairs']:,} pairs have direct CDR; "
        f"{coverage['pairs_with_both_ipdr_profiles']:,}/{coverage['pairs']:,} have both IPDR profiles.",
        flush=True,
    )

    artifact, metrics, probabilities, split_names, importance = train_neural_network(
        features,
        pairs["label"].to_numpy(dtype=np.int64),
        FEATURE_NAMES,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        progress=lambda message: print(message, flush=True),
    )
    identity_context = context_metadata(
        cdr_schema,
        ipdr_schema,
        resolver,
        identity_answer=getattr(args, "identity_answer", None),
        identity_validation=getattr(args, "identity_validation", None),
        identity_decision=getattr(args, "identity_decision", None),
        identity_mapping=getattr(args, "identity_mapping", None),
    )
    artifact["data_context"] = identity_context
    artifact["training_coverage"] = coverage
    metrics["data_coverage"] = coverage
    metrics["data_context"] = identity_context

    model_path = output_dir / "model.pt"
    save_artifact(artifact, model_path)
    write_json(metrics, output_dir / "metrics.json")
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    scored = pairs[["person_a", "person_b", "person_a_id", "person_b_id", "label"]].copy()
    scored["dataset_split"] = split_names
    scored["predicted_probability"] = probabilities
    scored["predicted_label"] = (probabilities >= float(artifact["threshold"])).astype(int)
    scored.to_csv(output_dir / "base_truth_predictions.csv", index=False)
    feature_export = pd.concat([scored, features], axis=1)
    feature_export.to_csv(output_dir / "engineered_pair_features.csv", index=False)

    test_metrics = metrics["test"]
    print(
        f"Training complete. Held-out test F1={test_metrics['f1']:.3f}, "
        f"accuracy={test_metrics['accuracy']:.3f}; decision threshold={metrics['threshold']:.3f}.",
        flush=True,
    )
    print(f"Saved model: {model_path}", flush=True)
    return model_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
