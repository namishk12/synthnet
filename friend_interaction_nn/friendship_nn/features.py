from __future__ import annotations

import math
from collections.abc import Callable
import numpy as np
import pandas as pd

from .identities import IdentityResolver
from .pairs import canonical_pair
from .schemas import iter_csv_chunks, parse_timestamp_columns


FEATURE_NAMES = [
    "cdr_has_direct_interaction",
    "cdr_events_total",
    "cdr_direction_min_count",
    "cdr_direction_max_count",
    "cdr_reciprocity_ratio",
    "cdr_direction_balance",
    "cdr_total_duration_seconds",
    "cdr_mean_duration_seconds",
    "cdr_max_duration_seconds",
    "cdr_duration_std_seconds",
    "cdr_active_days",
    "cdr_activity_span_days",
    "cdr_events_per_active_day",
    "cdr_night_ratio",
    "cdr_weekend_ratio",
    "cdr_voice_ratio",
    "cdr_contact_share_min",
    "cdr_contact_share_max",
    "cdr_degree_min",
    "cdr_degree_max",
    "cdr_common_contacts",
    "cdr_contact_jaccard",
    "ipdr_both_people_present",
    "ipdr_session_count_min",
    "ipdr_session_count_max",
    "ipdr_session_count_similarity",
    "ipdr_total_mb_min",
    "ipdr_total_mb_max",
    "ipdr_total_mb_similarity",
    "ipdr_mean_session_seconds_similarity",
    "ipdr_active_days_min",
    "ipdr_active_days_max",
    "ipdr_hour_cosine_similarity",
    "ipdr_destination_jaccard",
    "ipdr_port_jaccard",
    "ipdr_cell_jaccard",
    "ipdr_time_bucket_jaccard",
    "ipdr_night_ratio_similarity",
    "ipdr_weekend_ratio_similarity",
]

MAX_PROFILE_SET_SIZE = 2048
MAX_CONTACT_SET_SIZE = 10000


def _new_pair_stats() -> dict[str, object]:
    return {
        "directions": [0, 0],
        "duration_sum": 0.0,
        "duration_sq_sum": 0.0,
        "duration_count": 0,
        "duration_max": 0.0,
        "active_days": set(),
        "timestamp_min": None,
        "timestamp_max": None,
        "time_count": 0,
        "night_count": 0,
        "weekend_count": 0,
        "type_count": 0,
        "voice_count": 0,
    }


def _new_cdr_person_stats() -> dict[str, object]:
    return {"events": 0, "contacts": set()}


def _new_ipdr_person_stats() -> dict[str, object]:
    return {
        "sessions": 0,
        "total_mb": 0.0,
        "duration_sum": 0.0,
        "duration_count": 0,
        "active_days": set(),
        "hours": np.zeros(24, dtype=np.float64),
        "time_count": 0,
        "night_count": 0,
        "weekend_count": 0,
        "destinations": set(),
        "ports": set(),
        "cells": set(),
        "time_buckets": set(),
    }


def _add_capped(target: set[str], value: object, cap: int) -> None:
    text = str(value).strip().casefold()
    if text and text not in {"nan", "none", "null"} and len(target) < cap:
        target.add(text)


def _safe_number(value: object) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _ratio_similarity(first: float, second: float) -> float:
    high = max(first, second)
    return min(first, second) / high if high > 0 else 0.0


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def _timestamp_values(frame: pd.DataFrame, schema: dict[str, object]) -> list[pd.Timestamp | None]:
    parsed = parse_timestamp_columns(frame, schema)
    return [None if pd.isna(value) else pd.Timestamp(value) for value in parsed]


def _extract_cdr(
    schema: dict[str, object],
    pair_keys: set[tuple[str, str]],
    people: set[str],
    resolver: IdentityResolver,
    chunk_size: int,
    progress: Callable[[str], None],
) -> tuple[dict[tuple[str, str], dict[str, object]], dict[str, dict[str, object]], int]:
    pair_stats = {key: _new_pair_stats() for key in pair_keys}
    person_stats = {person: _new_cdr_person_stats() for person in people}
    processed = 0

    for chunk_number, chunk in enumerate(iter_csv_chunks(schema, chunk_size), start=1):
        processed += len(chunk)
        callers = resolver.canonicalize_series(
            chunk[str(schema["caller"])], field_name=str(schema["caller"])
        )
        receivers = resolver.canonicalize_series(
            chunk[str(schema["receiver"])], field_name=str(schema["receiver"])
        )
        relevant = callers.isin(people) | receivers.isin(people)
        if not bool(relevant.any()):
            progress(f"CDR: scanned {processed:,} rows")
            continue
        subset = chunk.loc[relevant].copy()
        caller_values = callers.loc[relevant].tolist()
        receiver_values = receivers.loc[relevant].tolist()
        timestamps = _timestamp_values(subset, schema)

        duration_column = schema.get("duration")
        durations = (
            pd.to_numeric(subset[str(duration_column)], errors="coerce").fillna(0.0).tolist()
            if duration_column
            else [0.0] * len(subset)
        )
        event_type_column = schema.get("event_type")
        event_types = (
            subset[str(event_type_column)].astype(str).str.strip().str.casefold().tolist()
            if event_type_column
            else [""] * len(subset)
        )

        for caller, receiver, timestamp, duration, event_type in zip(
            caller_values, receiver_values, timestamps, durations, event_types
        ):
            if not caller or not receiver or caller == receiver:
                continue
            if caller in person_stats:
                stats = person_stats[caller]
                stats["events"] = int(stats["events"]) + 1
                _add_capped(stats["contacts"], receiver, MAX_CONTACT_SET_SIZE)
            if receiver in person_stats:
                stats = person_stats[receiver]
                stats["events"] = int(stats["events"]) + 1
                _add_capped(stats["contacts"], caller, MAX_CONTACT_SET_SIZE)

            key = canonical_pair(caller, receiver)
            if key not in pair_stats:
                continue
            stats = pair_stats[key]
            direction = 0 if caller == key[0] else 1
            stats["directions"][direction] += 1
            numeric_duration = max(_safe_number(duration), 0.0)
            stats["duration_sum"] = float(stats["duration_sum"]) + numeric_duration
            stats["duration_sq_sum"] = float(stats["duration_sq_sum"]) + numeric_duration**2
            stats["duration_count"] = int(stats["duration_count"]) + 1
            stats["duration_max"] = max(float(stats["duration_max"]), numeric_duration)
            if timestamp is not None:
                stats["active_days"].add(timestamp.date().isoformat())
                current_min = stats["timestamp_min"]
                current_max = stats["timestamp_max"]
                stats["timestamp_min"] = timestamp if current_min is None else min(current_min, timestamp)
                stats["timestamp_max"] = timestamp if current_max is None else max(current_max, timestamp)
                stats["time_count"] = int(stats["time_count"]) + 1
                stats["night_count"] = int(stats["night_count"]) + int(timestamp.hour < 6 or timestamp.hour >= 22)
                stats["weekend_count"] = int(stats["weekend_count"]) + int(timestamp.weekday() >= 5)
            if event_type:
                stats["type_count"] = int(stats["type_count"]) + 1
                stats["voice_count"] = int(stats["voice_count"]) + int("voice" in event_type or "call" in event_type)
        if chunk_number == 1 or chunk_number % 5 == 0:
            progress(f"CDR: scanned {processed:,} rows")
    progress(f"CDR complete: {processed:,} rows scanned")
    return pair_stats, person_stats, processed


def _extract_ipdr(
    schema: dict[str, object],
    people: set[str],
    resolver: IdentityResolver,
    chunk_size: int,
    progress: Callable[[str], None],
) -> tuple[dict[str, dict[str, object]], int]:
    person_stats = {person: _new_ipdr_person_stats() for person in people}
    processed = 0
    for chunk_number, chunk in enumerate(iter_csv_chunks(schema, chunk_size), start=1):
        processed += len(chunk)
        subscribers = resolver.canonicalize_series(
            chunk[str(schema["subscriber"])], field_name=str(schema["subscriber"])
        )
        relevant = subscribers.isin(people)
        if not bool(relevant.any()):
            progress(f"IPDR: scanned {processed:,} rows")
            continue
        subset = chunk.loc[relevant].copy()
        subscriber_values = subscribers.loc[relevant].tolist()
        timestamps = _timestamp_values(subset, schema)

        destination_column = schema.get("destination")
        destinations = (
            subset[str(destination_column)].astype(str).tolist()
            if destination_column
            else [""] * len(subset)
        )
        port_column = schema.get("destination_port")
        ports = subset[str(port_column)].astype(str).tolist() if port_column else [""] * len(subset)
        cell_column = schema.get("cell")
        cells = subset[str(cell_column)].astype(str).tolist() if cell_column else [""] * len(subset)

        megabytes_column = schema.get("megabytes")
        if megabytes_column:
            megabytes = pd.to_numeric(subset[str(megabytes_column)], errors="coerce").fillna(0.0)
        elif schema.get("upload") and schema.get("download"):
            upload = pd.to_numeric(subset[str(schema["upload"])], errors="coerce").fillna(0.0)
            download = pd.to_numeric(subset[str(schema["download"])], errors="coerce").fillna(0.0)
            megabytes = (upload + download) / 1024.0
        else:
            megabytes = pd.Series(0.0, index=subset.index)

        duration_column = schema.get("session_duration")
        durations = (
            pd.to_numeric(subset[str(duration_column)], errors="coerce").fillna(0.0).tolist()
            if duration_column
            else [0.0] * len(subset)
        )

        for subscriber, timestamp, destination, port, cell, mb, duration in zip(
            subscriber_values,
            timestamps,
            destinations,
            ports,
            cells,
            megabytes.tolist(),
            durations,
        ):
            if subscriber not in person_stats:
                continue
            stats = person_stats[subscriber]
            stats["sessions"] = int(stats["sessions"]) + 1
            stats["total_mb"] = float(stats["total_mb"]) + max(_safe_number(mb), 0.0)
            numeric_duration = max(_safe_number(duration), 0.0)
            stats["duration_sum"] = float(stats["duration_sum"]) + numeric_duration
            stats["duration_count"] = int(stats["duration_count"]) + 1
            _add_capped(stats["destinations"], destination, MAX_PROFILE_SET_SIZE)
            _add_capped(stats["ports"], port, MAX_PROFILE_SET_SIZE)
            _add_capped(stats["cells"], cell, MAX_PROFILE_SET_SIZE)
            if timestamp is not None:
                stats["active_days"].add(timestamp.date().isoformat())
                stats["hours"][timestamp.hour] += 1.0
                stats["time_count"] = int(stats["time_count"]) + 1
                stats["night_count"] = int(stats["night_count"]) + int(timestamp.hour < 6 or timestamp.hour >= 22)
                stats["weekend_count"] = int(stats["weekend_count"]) + int(timestamp.weekday() >= 5)
                _add_capped(
                    stats["time_buckets"],
                    timestamp.floor("h").isoformat(),
                    MAX_PROFILE_SET_SIZE,
                )
        if chunk_number == 1 or chunk_number % 5 == 0:
            progress(f"IPDR: scanned {processed:,} rows")
    progress(f"IPDR complete: {processed:,} rows scanned")
    return person_stats, processed


def _pair_feature_row(
    key: tuple[str, str],
    pair_stats: dict[tuple[str, str], dict[str, object]],
    cdr_people: dict[str, dict[str, object]],
    ipdr_people: dict[str, dict[str, object]],
) -> dict[str, float]:
    cdr = pair_stats[key]
    first_cdr = cdr_people[key[0]]
    second_cdr = cdr_people[key[1]]
    directions = [int(value) for value in cdr["directions"]]
    total_events = sum(directions)
    direction_min = min(directions)
    direction_max = max(directions)
    duration_count = int(cdr["duration_count"])
    duration_sum = float(cdr["duration_sum"])
    duration_mean = duration_sum / duration_count if duration_count else 0.0
    duration_variance = (
        max(float(cdr["duration_sq_sum"]) / duration_count - duration_mean**2, 0.0)
        if duration_count
        else 0.0
    )
    active_days = len(cdr["active_days"])
    timestamp_min = cdr["timestamp_min"]
    timestamp_max = cdr["timestamp_max"]
    activity_span = (
        max((timestamp_max - timestamp_min).total_seconds() / 86400.0, 0.0)
        if timestamp_min is not None and timestamp_max is not None
        else 0.0
    )
    time_count = int(cdr["time_count"])
    type_count = int(cdr["type_count"])
    first_events = int(first_cdr["events"])
    second_events = int(second_cdr["events"])
    shares = [
        total_events / first_events if first_events else 0.0,
        total_events / second_events if second_events else 0.0,
    ]
    first_contacts = first_cdr["contacts"]
    second_contacts = second_cdr["contacts"]

    first_ipdr = ipdr_people[key[0]]
    second_ipdr = ipdr_people[key[1]]
    sessions = [int(first_ipdr["sessions"]), int(second_ipdr["sessions"])]
    total_mb = [float(first_ipdr["total_mb"]), float(second_ipdr["total_mb"])]
    mean_durations = [
        float(stats["duration_sum"]) / int(stats["duration_count"])
        if int(stats["duration_count"])
        else 0.0
        for stats in (first_ipdr, second_ipdr)
    ]
    ipdr_active_days = [len(first_ipdr["active_days"]), len(second_ipdr["active_days"])]
    night_ratios = [
        int(stats["night_count"]) / int(stats["time_count"]) if int(stats["time_count"]) else 0.0
        for stats in (first_ipdr, second_ipdr)
    ]
    weekend_ratios = [
        int(stats["weekend_count"]) / int(stats["time_count"]) if int(stats["time_count"]) else 0.0
        for stats in (first_ipdr, second_ipdr)
    ]

    return {
        "cdr_has_direct_interaction": float(total_events > 0),
        "cdr_events_total": float(total_events),
        "cdr_direction_min_count": float(direction_min),
        "cdr_direction_max_count": float(direction_max),
        "cdr_reciprocity_ratio": (2.0 * direction_min / total_events) if total_events else 0.0,
        "cdr_direction_balance": (abs(directions[0] - directions[1]) / total_events) if total_events else 0.0,
        "cdr_total_duration_seconds": duration_sum,
        "cdr_mean_duration_seconds": duration_mean,
        "cdr_max_duration_seconds": float(cdr["duration_max"]),
        "cdr_duration_std_seconds": math.sqrt(duration_variance),
        "cdr_active_days": float(active_days),
        "cdr_activity_span_days": activity_span,
        "cdr_events_per_active_day": total_events / active_days if active_days else 0.0,
        "cdr_night_ratio": int(cdr["night_count"]) / time_count if time_count else 0.0,
        "cdr_weekend_ratio": int(cdr["weekend_count"]) / time_count if time_count else 0.0,
        "cdr_voice_ratio": int(cdr["voice_count"]) / type_count if type_count else 0.0,
        "cdr_contact_share_min": float(min(shares)),
        "cdr_contact_share_max": float(max(shares)),
        "cdr_degree_min": float(min(len(first_contacts), len(second_contacts))),
        "cdr_degree_max": float(max(len(first_contacts), len(second_contacts))),
        "cdr_common_contacts": float(len(first_contacts & second_contacts)),
        "cdr_contact_jaccard": _jaccard(first_contacts, second_contacts),
        "ipdr_both_people_present": float(sessions[0] > 0 and sessions[1] > 0),
        "ipdr_session_count_min": float(min(sessions)),
        "ipdr_session_count_max": float(max(sessions)),
        "ipdr_session_count_similarity": _ratio_similarity(*sessions),
        "ipdr_total_mb_min": float(min(total_mb)),
        "ipdr_total_mb_max": float(max(total_mb)),
        "ipdr_total_mb_similarity": _ratio_similarity(*total_mb),
        "ipdr_mean_session_seconds_similarity": _ratio_similarity(*mean_durations),
        "ipdr_active_days_min": float(min(ipdr_active_days)),
        "ipdr_active_days_max": float(max(ipdr_active_days)),
        "ipdr_hour_cosine_similarity": _cosine(first_ipdr["hours"], second_ipdr["hours"]),
        "ipdr_destination_jaccard": _jaccard(first_ipdr["destinations"], second_ipdr["destinations"]),
        "ipdr_port_jaccard": _jaccard(first_ipdr["ports"], second_ipdr["ports"]),
        "ipdr_cell_jaccard": _jaccard(first_ipdr["cells"], second_ipdr["cells"]),
        "ipdr_time_bucket_jaccard": _jaccard(first_ipdr["time_buckets"], second_ipdr["time_buckets"]),
        "ipdr_night_ratio_similarity": (
            1.0 - abs(night_ratios[0] - night_ratios[1])
            if int(first_ipdr["time_count"]) and int(second_ipdr["time_count"])
            else 0.0
        ),
        "ipdr_weekend_ratio_similarity": (
            1.0 - abs(weekend_ratios[0] - weekend_ratios[1])
            if int(first_ipdr["time_count"]) and int(second_ipdr["time_count"])
            else 0.0
        ),
    }


def extract_pair_features(
    pairs: pd.DataFrame,
    cdr_schema: dict[str, object],
    ipdr_schema: dict[str, object],
    resolver: IdentityResolver,
    chunk_size: int = 100_000,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    progress = progress or (lambda message: None)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    pair_keys = set(zip(pairs["pair_lo"], pairs["pair_hi"]))
    people = {person for key in pair_keys for person in key}
    if not pair_keys:
        raise ValueError("No person pairs were supplied.")

    pair_stats, cdr_people, cdr_rows = _extract_cdr(
        cdr_schema, pair_keys, people, resolver, chunk_size, progress
    )
    ipdr_people, ipdr_rows = _extract_ipdr(
        ipdr_schema, people, resolver, chunk_size, progress
    )
    rows = [
        _pair_feature_row(key, pair_stats, cdr_people, ipdr_people)
        for key in zip(pairs["pair_lo"], pairs["pair_hi"])
    ]
    feature_frame = pd.DataFrame(rows, columns=FEATURE_NAMES).astype(float)
    coverage = {
        "cdr_rows_scanned": cdr_rows,
        "ipdr_rows_scanned": ipdr_rows,
        "pairs": len(feature_frame),
        "pairs_with_direct_cdr": int(feature_frame["cdr_has_direct_interaction"].sum()),
        "pairs_with_both_ipdr_profiles": int(feature_frame["ipdr_both_people_present"].sum()),
    }
    return feature_frame, coverage
