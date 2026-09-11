"""CLI entry point for the same upload identity check used by the API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from identity_validation import (
    IDENTITY_ANSWERS,
    apply_identity_answer,
    load_identity_mapping,
    validate_cdr_ipdr,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate CDR subjects and IPDR subscribers across every CSV row. "
            "Identifier agreement supports matching but does not prove personal identity."
        )
    )
    parser.add_argument("--cdr", help="CDR CSV file.")
    parser.add_argument("--ipdr", help="IPDR CSV file.")
    parser.add_argument("--identity-map", help="Explicit subscriber/alias mapping CSV.")
    parser.add_argument("--identity-map-json", help="Explicit alias-to-canonical JSON mapping.")
    parser.add_argument("--identity-answer", choices=IDENTITY_ANSWERS)
    parser.add_argument("--generation-mode")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--timezone", default="Asia/Kolkata")
    parser.add_argument("--output", help="Optional JSON report path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.cdr and not args.ipdr:
        print("ERROR: supply --cdr, --ipdr, or both.", file=sys.stderr)
        return 2
    try:
        inline_mapping = None
        if args.identity_map_json:
            inline_mapping = json.loads(Path(args.identity_map_json).expanduser().read_text(encoding="utf-8"))
        mapping = load_identity_mapping(args.identity_map, inline_mapping)
        validation = validate_cdr_ipdr(
            args.cdr,
            args.ipdr,
            mapping=mapping,
            chunk_size=args.chunk_size,
            default_timezone=args.timezone,
        )
        decision = apply_identity_answer(
            validation,
            args.identity_answer,
            generation_mode=args.generation_mode,
        )
        report = {
            "identity_answer": decision.get("answer"),
            "generation_mode": decision.get("generation_mode"),
            "identity_mappings": mapping.metadata(),
            "validation": validation,
            "identity_decision": decision,
        }
        encoded = json.dumps(report, indent=2, default=str)
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded, encoding="utf-8")
        print(encoded)
        return 0 if decision["allowed"] or args.identity_answer is None else 2
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
