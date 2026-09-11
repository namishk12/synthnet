"""IPDR-only behavioural correlation with a valid zero-pair result.

This is deliberately a profile comparison tool, not an identity resolver:
subscriber IDs are the only person keys. Shared destinations, IPs, ports, or
time buckets are behavioural evidence and never create a person mapping.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from identity_validation import (
    DEFAULT_TIMEZONE,
    IdentityMapping,
    iter_source_chunks,
    parse_timestamp_series,
    resolve_source_schema,
    validate_cdr_ipdr,
)


PAIR_COLUMNS = [
    "person_a",
    "person_b",
    "ipdr_correlation_score",
    "shared_destinations",
    "shared_ports",
    "shared_time_buckets",
    "evidence_type",
]


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _cosine(first: Counter[int], second: Counter[int]) -> float:
    keys = set(first) | set(second)
    if not keys:
        return 0.0
    numerator = sum(first[key] * second[key] for key in keys)
    denominator = math.sqrt(sum(value * value for value in first.values())) * math.sqrt(
        sum(value * value for value in second.values())
    )
    return numerator / denominator if denominator else 0.0


def correlate_ipdr(
    ipdr_path: str | Path,
    *,
    mapping: IdentityMapping | None = None,
    overrides: Mapping[str, object] | None = None,
    chunk_size: int = 100_000,
    default_timezone: str = DEFAULT_TIMEZONE,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Correlate every distinct IPDR subscriber profile pair."""

    progress = progress or (lambda message: None)
    resolver = mapping or IdentityMapping.empty()
    schema = resolve_source_schema(ipdr_path, "ipdr", overrides)
    profiles: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "destinations": set(),
            "ports": set(),
            "time_buckets": set(),
            "hours": Counter(),
            "sessions": 0,
        }
    )
    rows_scanned = 0
    chunks = 0
    for chunks, frame in enumerate(iter_source_chunks(schema, chunk_size), start=1):
        rows_scanned += len(frame)
        subscribers = [
            resolver.canonicalize(value, field_name=str(schema["subscriber"]))
            for value in frame[str(schema["subscriber"])].tolist()
        ]
        timestamps = parse_timestamp_series(frame, schema, default_timezone)
        destination_column = schema.get("destination")
        destinations = (
            frame[str(destination_column)].map(str).tolist()
            if destination_column
            else [""] * len(frame)
        )
        port_column = schema.get("destination_port")
        ports = (
            frame[str(port_column)].map(str).tolist()
            if port_column
            else [""] * len(frame)
        )
        for identifier, timestamp, destination, port in zip(subscribers, timestamps.tolist(), destinations, ports):
            if not identifier:
                continue
            profile = profiles[identifier]
            profile["sessions"] = int(profile["sessions"]) + 1
            destination = str(destination).strip().casefold()
            port = str(port).strip().casefold()
            if destination and destination not in {"nan", "none", "null"}:
                profile["destinations"].add(destination)
            if port and port not in {"nan", "none", "null"}:
                profile["ports"].add(port)
            if timestamp is not None and not pd.isna(timestamp):
                profile["hours"][int(timestamp.hour)] += 1
                profile["time_buckets"].add(timestamp.floor("h").isoformat())
        if chunks == 1 or chunks % 5 == 0:
            progress(f"IPDR correlation: scanned {rows_scanned:,} rows")
    progress(f"IPDR correlation complete: {rows_scanned:,} rows scanned in {chunks:,} chunk(s)")

    identifiers = sorted(profiles)
    rows: list[dict[str, object]] = []
    for first, second in itertools.combinations(identifiers, 2):
        left, right = profiles[first], profiles[second]
        destination_score = _jaccard(left["destinations"], right["destinations"])
        port_score = _jaccard(left["ports"], right["ports"])
        time_score = _jaccard(left["time_buckets"], right["time_buckets"])
        hourly_score = _cosine(left["hours"], right["hours"])
        score = 0.35 * destination_score + 0.15 * port_score + 0.25 * time_score + 0.25 * hourly_score
        rows.append(
            {
                "person_a": first,
                "person_b": second,
                "ipdr_correlation_score": round(score, 6),
                "shared_destinations": len(left["destinations"] & right["destinations"]),
                "shared_ports": len(left["ports"] & right["ports"]),
                "shared_time_buckets": len(left["time_buckets"] & right["time_buckets"]),
                "evidence_type": "observed_ipdr_behavioral_evidence",
            }
        )
    result = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    return {
        "pairs": result,
        "pair_count": len(result),
        "subscriber_count": len(identifiers),
        "subscriber_identifiers": identifiers,
        "rows_scanned": rows_scanned,
        "chunk_count": chunks,
        "evidence_type": "observed_ipdr_behavioral_evidence",
        "explanation": (
            "One or fewer IPDR subscribers produce a valid zero-pair result. "
            "Shared destinations/IPs/ports are behavioural signals, not identity mappings."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Correlate IPDR subscribers without requiring a CDR file.")
    parser.add_argument("--ipdr", required=True, help="IPDR CSV file.")
    parser.add_argument("--output", required=True, help="Output correlation CSV.")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--ipdr-subscriber-col")
    parser.add_argument("--ipdr-timestamp-col")
    parser.add_argument("--ipdr-date-col")
    parser.add_argument("--ipdr-time-col")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overrides = {
        key: value
        for key, value in {
            "subscriber": args.ipdr_subscriber_col,
            "timestamp": args.ipdr_timestamp_col,
            "date": args.ipdr_date_col,
            "time": args.ipdr_time_col,
        }.items()
        if value
    }
    result = correlate_ipdr(
        args.ipdr,
        overrides=overrides,
        chunk_size=args.chunk_size,
        default_timezone=args.timezone,
        progress=print,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result["pairs"].to_csv(output, index=False)
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
