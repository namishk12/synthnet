from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from friendship_nn.cli_common import add_data_arguments, add_pair_column_arguments, resolve_inputs
from friendship_nn.features import extract_pair_features
from friendship_nn.model import load_artifact, predict_with_artifact
from friendship_nn.pairs import load_pair_csv, prepare_pairs


EVIDENCE_COLUMNS = [
    "cdr_has_direct_interaction",
    "cdr_events_total",
    "cdr_reciprocity_ratio",
    "cdr_total_duration_seconds",
    "cdr_active_days",
    "cdr_common_contacts",
    "cdr_contact_jaccard",
    "ipdr_both_people_present",
    "ipdr_hour_cosine_similarity",
    "ipdr_destination_jaccard",
    "ipdr_port_jaccard",
    "ipdr_time_bucket_jaccard",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one pair or a CSV of pairs using a trained relationship neural network."
    )
    parser.add_argument("--model", required=True, help="model.pt produced by train_model.py.")
    add_data_arguments(parser)
    parser.add_argument("--pairs", help="CSV containing person_a/person_b or another supported pair schema.")
    parser.add_argument("--person-a", help="First identifier for a single-pair prediction.")
    parser.add_argument("--person-b", help="Second identifier for a single-pair prediction.")
    add_pair_column_arguments(parser, include_label=False)
    parser.add_argument("--output", help="Prediction CSV path. Defaults beside model.pt.")
    parser.add_argument("--include-all-features", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _load_requested_pairs(args: argparse.Namespace) -> tuple[pd.DataFrame, bool]:
    if args.pairs:
        if args.person_a or args.person_b:
            raise ValueError("Use either --pairs or --person-a/--person-b, not both.")
        return (
            load_pair_csv(
                args.pairs,
                require_label=False,
                person_a_col=args.person_a_col,
                person_b_col=args.person_b_col,
            ),
            False,
        )
    if not args.person_a or not args.person_b:
        raise ValueError("Supply --pairs, or supply both --person-a and --person-b.")
    return (
        pd.DataFrame(
            {"person_a": [args.person_a], "person_b": [args.person_b], "source_row": [1]}
        ),
        True,
    )


def run(args: argparse.Namespace) -> Path:
    raw_pairs, single_pair = _load_requested_pairs(args)
    cdr_schema, ipdr_schema, resolver = resolve_inputs(args)
    pairs = prepare_pairs(raw_pairs, resolver)
    features, coverage = extract_pair_features(
        pairs,
        cdr_schema,
        ipdr_schema,
        resolver,
        chunk_size=args.chunk_size,
        progress=lambda message: print(message, flush=True),
    )
    artifact = load_artifact(args.model)
    probabilities, predictions = predict_with_artifact(artifact, features, args.device)
    threshold = float(artifact["threshold"])

    result = pairs[["person_a", "person_b", "person_a_id", "person_b_id"]].copy()
    result["friend_or_interacting_probability"] = probabilities
    result["predicted_label"] = predictions
    result["prediction"] = [
        "friend_or_interacting" if value else "not_friend_or_interacting" for value in predictions
    ]
    result["decision_threshold"] = threshold
    selected_features = list(features.columns) if args.include_all_features else EVIDENCE_COLUMNS
    result = pd.concat([result, features[selected_features]], axis=1)

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(args.model).expanduser().resolve().parent / "predictions.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(
        f"Scored {len(result):,} pair(s); {coverage['pairs_with_direct_cdr']:,} have direct CDR evidence. "
        f"Saved: {output}",
        flush=True,
    )
    if single_pair:
        row = result.iloc[0]
        print(
            f"Prediction: {row['prediction']} | probability={row['friend_or_interacting_probability']:.4f} "
            f"| threshold={threshold:.4f}",
            flush=True,
        )
        print(
            "This is a model estimate from communications patterns, not proof of a personal relationship.",
            flush=True,
        )
    return output


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

