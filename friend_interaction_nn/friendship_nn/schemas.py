from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from identity_validation import parse_timestamp_series as _shared_parse_timestamp_series  # noqa: E402


def normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


CDR_ALIASES: dict[str, tuple[str, ...]] = {
    "caller": (
        "Mobile_No",
        "caller_id",
        "caller",
        "caller_msisdn",
        "subscriber_id",
        "calling_party",
        "a_party",
        "A_Party_No",
    ),
    "receiver": (
        "Other_Party_No",
        "receiver_msisdn",
        "receiver_id",
        "receiver",
        "callee_id",
        "called_party",
        "b_party",
        "B_Party_No",
    ),
    "timestamp": ("timestamp", "call_timestamp", "datetime", "event_time"),
    "date": ("Call_Date", "call_date", "date"),
    "time": (
        "Call_Initiation_Time(CIT)",
        "call_time",
        "start_time",
        "time",
    ),
    "duration": ("Call_Duration", "duration_seconds", "call_duration", "duration"),
    "event_type": ("Call_Type", "Service_Type", "event_type", "type"),
    "tower": (
        "First_Cell_id",
        "First CELL ID",
        "routing_tower",
        "cell_id",
        "tower_id",
    ),
}


IPDR_ALIASES: dict[str, tuple[str, ...]] = {
    "subscriber": (
        "Landline/MSISDN/MDN/Leased Circuit ID for Internet Access",
        "subscriber_id",
        "msisdn",
        "Mobile_No",
        "user_id",
        "User Id for internet Access based on authentication",
    ),
    "timestamp": (
        "TIME1 (dd/MM/yyyy HH:mm:ss)",
        "timestamp",
        "session_timestamp",
        "datetime",
        "start_datetime",
    ),
    "date": (
        "Start Date of Public IP Address allocation (dd/mm/yyyy)",
        "start_date",
        "date",
    ),
    "time": (
        "IST Start Time of Public IP address allocation (hh:mm:ss)",
        "start_time",
        "time",
    ),
    "destination": (
        "Destination IP Address",
        "destination_ip",
        "destination",
        "remote_ip",
    ),
    "destination_port": (
        "Destination Port",
        "destination_port",
        "Destination_Port",
        "DestinationPort",
        "destinationPort",
    ),
    "cell": ("First CELL ID", "First_Cell_id", "cell_id", "tower_id"),
    "session_duration": (
        "Session Duration (Seconds)",
        "duration_seconds",
        "session_duration",
    ),
    "megabytes": ("megabytes_transferred", "megabytes", "data_mb"),
    "upload": ("Data Volume Up Link", "upload_kb", "uplink_kb"),
    "download": ("Data Volume Down Link", "download_kb", "downlink_kb"),
}


ALIASES_BY_KIND = {"cdr": CDR_ALIASES, "ipdr": IPDR_ALIASES}


def resolve_column(
    columns: Iterable[str], aliases: Iterable[str], required: bool = False
) -> str | None:
    column_list = [str(column).strip() for column in columns]
    exact = {column: column for column in column_list}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
    normalized = {normalize_column_name(column): column for column in column_list}
    for alias in aliases:
        match = normalized.get(normalize_column_name(alias))
        if match is not None:
            return match
    if required:
        raise ValueError(
            f"Missing required column. Looked for {list(aliases)}; available columns are {column_list}."
        )
    return None


def _row_looks_like_header(row: list[str], kind: str) -> bool:
    aliases = ALIASES_BY_KIND[kind]
    try:
        if kind == "cdr":
            resolve_column(row, aliases["caller"], required=True)
            resolve_column(row, aliases["receiver"], required=True)
        else:
            resolve_column(row, aliases["subscriber"], required=True)
        timestamp = resolve_column(row, aliases["timestamp"])
        date = resolve_column(row, aliases["date"])
        time = resolve_column(row, aliases["time"])
        return timestamp is not None or (date is not None and time is not None)
    except ValueError:
        return False


def inspect_csv_header(path: str | Path, kind: str) -> tuple[int, list[str]]:
    source = Path(path).expanduser().resolve()
    if kind not in ALIASES_BY_KIND:
        raise ValueError(f"Unsupported CSV kind: {kind}")
    if not source.is_file():
        raise FileNotFoundError(f"{kind.upper()} CSV not found: {source}")
    with source.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader):
            columns = [column.strip() for column in row]
            if _row_looks_like_header(columns, kind):
                return row_index, columns
            if row_index >= 100:
                break
    columns = list(
        pd.read_csv(source, nrows=0, encoding="utf-8-sig", encoding_errors="replace").columns
    )
    return 0, [str(column).strip() for column in columns]


def _resolve_override(
    columns: list[str], override: str | None, aliases: tuple[str, ...], required: bool = False
) -> str | None:
    if override:
        return resolve_column(columns, (override,), required=True)
    return resolve_column(columns, aliases, required=required)


def resolve_data_schema(
    path: str | Path,
    kind: str,
    overrides: dict[str, str | None] | None = None,
) -> dict[str, object]:
    overrides = overrides or {}
    header_row, columns = inspect_csv_header(path, kind)
    aliases = ALIASES_BY_KIND[kind]
    schema: dict[str, object] = {
        "kind": kind,
        "path": str(Path(path).expanduser().resolve()),
        "header_row": header_row,
        "all_columns": columns,
    }
    required_fields = ("caller", "receiver") if kind == "cdr" else ("subscriber",)
    for field, field_aliases in aliases.items():
        schema[field] = _resolve_override(
            columns,
            overrides.get(field),
            field_aliases,
            required=field in required_fields,
        )

    # Timestamps improve temporal features but are not identity keys.  Keep a
    # missing temporal window inspectable; the shared parser will return NaT
    # and downstream features will record unavailable temporal evidence.
    return schema


def schema_use_columns(schema: dict[str, object]) -> list[str]:
    ignored = {"kind", "path", "header_row", "all_columns"}
    result: list[str] = []
    for key, value in schema.items():
        if key not in ignored and isinstance(value, str) and value not in result:
            result.append(value)
    return result


def iter_csv_chunks(
    schema: dict[str, object], chunk_size: int
) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        str(schema["path"]),
        skiprows=int(schema["header_row"]),
        usecols=schema_use_columns(schema),
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )


def parse_timestamp_columns(frame: pd.DataFrame, schema: dict[str, object]) -> pd.Series:
    return _shared_parse_timestamp_series(frame, schema)


def public_schema(schema: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in schema.items() if key != "all_columns"}

