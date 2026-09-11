from __future__ import annotations

import argparse
from pathlib import Path

from .identities import IdentityResolver, locate_subscriber_mapping
from .schemas import public_schema, resolve_data_schema
from identity_validation import (  # noqa: E402
    IDENTITY_ANSWERS,
    apply_identity_answer,
    load_identity_mapping,
    validate_cdr_ipdr,
)


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cdr", required=True, help="CDR CSV file.")
    parser.add_argument("--ipdr", required=True, help="IPDR CSV file.")
    parser.add_argument(
        "--subscribers",
        help=(
            "Optional subscribers.csv identity map. If omitted, a sibling subscribers.csv "
            "is used automatically when present."
        ),
    )
    parser.add_argument(
        "--no-auto-subscribers",
        action="store_true",
        help="Do not auto-detect a sibling subscribers.csv file.",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000, help="CSV rows per streaming chunk.")
    parser.add_argument(
        "--identity-answer",
        choices=IDENTITY_ANSWERS,
        required=True,
        help=(
            "How the supplied CDR and IPDR subjects relate. The answer controls "
            "whether cross-source features may be combined."
        ),
    )
    parser.add_argument(
        "--identity-confirmation",
        action="store_true",
        help="Record that the reviewed identity evidence was acknowledged; it cannot override contradictions.",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Kolkata",
        help="Timezone assumed for timestamps without an offset (default: Asia/Kolkata).",
    )
    parser.add_argument("--cdr-caller-col", help="Override the detected CDR caller column.")
    parser.add_argument("--cdr-receiver-col", help="Override the detected CDR receiver column.")
    parser.add_argument("--cdr-timestamp-col", help="Override the detected CDR timestamp column.")
    parser.add_argument("--cdr-date-col", help="Override the detected CDR date column.")
    parser.add_argument("--cdr-time-col", help="Override the detected CDR time column.")
    parser.add_argument("--cdr-duration-col", help="Override the detected CDR duration column.")
    parser.add_argument("--ipdr-subscriber-col", help="Override the detected IPDR subscriber column.")
    parser.add_argument("--ipdr-timestamp-col", help="Override the detected IPDR timestamp column.")
    parser.add_argument("--ipdr-date-col", help="Override the detected IPDR date column.")
    parser.add_argument("--ipdr-time-col", help="Override the detected IPDR time column.")


def add_pair_column_arguments(parser: argparse.ArgumentParser, include_label: bool) -> None:
    parser.add_argument("--person-a-col", help="Override the first person column in the pair CSV.")
    parser.add_argument("--person-b-col", help="Override the second person column in the pair CSV.")
    if include_label:
        parser.add_argument("--label-col", help="Override the base-truth label column.")


def resolve_inputs(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object], IdentityResolver]:
    cdr_path = Path(args.cdr).expanduser().resolve()
    ipdr_path = Path(args.ipdr).expanduser().resolve()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive.")

    cdr_schema = resolve_data_schema(
        cdr_path,
        "cdr",
        {
            "caller": args.cdr_caller_col,
            "receiver": args.cdr_receiver_col,
            "timestamp": args.cdr_timestamp_col,
            "date": args.cdr_date_col,
            "time": args.cdr_time_col,
            "duration": args.cdr_duration_col,
        },
    )
    ipdr_schema = resolve_data_schema(
        ipdr_path,
        "ipdr",
        {
            "subscriber": args.ipdr_subscriber_col,
            "timestamp": args.ipdr_timestamp_col,
            "date": args.ipdr_date_col,
            "time": args.ipdr_time_col,
        },
    )
    mapping_path = locate_subscriber_mapping(
        args.subscribers,
        cdr_path,
        ipdr_path,
        auto_detect=not args.no_auto_subscribers,
    )
    if mapping_path is None:
        resolver = IdentityResolver.empty()
        print("Identity mapping: normalized identifiers directly (no subscribers.csv used).", flush=True)
    else:
        resolver = IdentityResolver.from_subscribers_csv(mapping_path)
        print(
            f"Identity mapping: {mapping_path} ({len(resolver.alias_to_canonical):,} aliases, "
            f"canonical column '{resolver.canonical_column}').",
            flush=True,
        )
        if resolver.ambiguous_alias_count:
            print(
                f"Identity mapping ignored {resolver.ambiguous_alias_count:,} ambiguous aliases.",
                flush=True,
            )
    shared_mapping = load_identity_mapping(mapping_path, country_code="91")
    requested_mode = (
        "multi_person_conditioned"
        if args.identity_answer == "multiple_person_dataset"
        else "combined"
        if args.identity_answer == "same_person"
        else "separate_sources"
    )
    identity_validation = validate_cdr_ipdr(
        cdr_path,
        ipdr_path,
        mapping=shared_mapping,
        overrides={
            "cdr": {
                "subject": cdr_schema.get("caller"),
                "contact": cdr_schema.get("receiver"),
                "timestamp": cdr_schema.get("timestamp"),
                "date": cdr_schema.get("date"),
                "time": cdr_schema.get("time"),
            },
            "ipdr": {
                "subscriber": ipdr_schema.get("subscriber"),
                "timestamp": ipdr_schema.get("timestamp"),
                "date": ipdr_schema.get("date"),
                "time": ipdr_schema.get("time"),
            },
        },
        chunk_size=args.chunk_size,
        default_timezone=args.timezone,
    )
    identity_decision = apply_identity_answer(
        identity_validation,
        args.identity_answer,
        generation_mode=requested_mode,
        confirmation=getattr(args, "identity_confirmation", False),
    )
    matching = identity_validation["matching"]
    print(
        "Identity evidence: "
        f"{matching['cdr_subject_count']:,} CDR subject(s), "
        f"{matching['ipdr_subscriber_count']:,} IPDR subscriber(s), "
        f"{matching['matched_count']:,} matched, "
        f"{matching['unmatched_cdr_count']:,} CDR-unmatched, "
        f"{matching['unmatched_ipdr_count']:,} IPDR-unmatched.",
        flush=True,
    )
    print(f"Observation periods: {identity_validation['temporal']['explanation']}", flush=True)
    if not identity_decision["allowed"]:
        raise ValueError(
            "Identity processing check blocked this operation: "
            + " ".join(identity_decision["blocking_reasons"])
        )
    if args.identity_answer in {"different_people", "not_sure"}:
        raise ValueError(
            "The relationship-training CLI requires a combined per-person mapping. "
            "Use synthnet_validate.py for separate-source evidence, or choose "
            "Same person/Multiple-person dataset after review."
        )
    args.identity_validation = identity_validation
    args.identity_decision = identity_decision
    args.identity_mapping = shared_mapping.metadata()
    cdr_timestamp_description = cdr_schema.get("timestamp") or (
        f"{cdr_schema.get('date')} + {cdr_schema.get('time')}"
    )
    ipdr_timestamp_description = ipdr_schema.get("timestamp") or (
        f"{ipdr_schema.get('date')} + {ipdr_schema.get('time')}"
    )
    print(
        "Resolved CDR: "
        f"caller='{cdr_schema['caller']}', receiver='{cdr_schema['receiver']}', "
        f"timestamp='{cdr_timestamp_description}'.",
        flush=True,
    )
    print(
        "Resolved IPDR: "
        f"subscriber='{ipdr_schema['subscriber']}', "
        f"timestamp='{ipdr_timestamp_description}'.",
        flush=True,
    )
    return cdr_schema, ipdr_schema, resolver


def context_metadata(
    cdr_schema: dict[str, object],
    ipdr_schema: dict[str, object],
    resolver: IdentityResolver,
    *,
    identity_validation: dict[str, object] | None = None,
    identity_decision: dict[str, object] | None = None,
    identity_answer: str | None = None,
    identity_mapping: dict[str, object] | None = None,
) -> dict[str, object]:
    context = {
        "cdr": public_schema(cdr_schema),
        "ipdr": public_schema(ipdr_schema),
        "identity_mapping": identity_mapping or {
            "source_path": resolver.source_path,
            "canonical_column": resolver.canonical_column,
            "alias_count": len(resolver.alias_to_canonical),
            "ambiguous_alias_count": resolver.ambiguous_alias_count,
            "aliases": dict(sorted(resolver.alias_to_canonical.items())),
        },
    }
    if identity_answer is not None:
        context["identity_answer"] = identity_answer
    if identity_validation is not None:
        context["identity_validation"] = identity_validation
    if identity_decision is not None:
        context["identity_decision"] = identity_decision
    return context
