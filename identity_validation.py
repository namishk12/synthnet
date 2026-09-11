"""Shared identity, schema, and observation-window checks for SynthNet.

The upload UI, the local HTTP API, the generator, and the relationship tools
all need to make the same distinction: a value in a caller/subscriber column
is an identifier, not proof that two records describe the same human.  This
module keeps that distinction explicit and provides a streaming validator so
large CSVs are never reduced to their preview rows for identity decisions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import timedelta, timezone as dt_timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


IDENTITY_QUESTION = "Do these CDR and IPDR files belong to the same person?"
IDENTITY_ANSWERS = (
    "same_person",
    "different_people",
    "multiple_person_dataset",
    "not_sure",
)
IDENTITY_ANSWER_LABELS = {
    "same_person": "Same person",
    "different_people": "Different people",
    "multiple_person_dataset": "Multiple-person dataset",
    "not_sure": "Not sure",
}
IDENTITY_ANSWER_ALIASES = {
    "same": "same_person",
    "same person": "same_person",
    "same_person": "same_person",
    "different": "different_people",
    "different people": "different_people",
    "different_people": "different_people",
    "multiple": "multiple_person_dataset",
    "multiple people": "multiple_person_dataset",
    "multiple-person dataset": "multiple_person_dataset",
    "multiple person dataset": "multiple_person_dataset",
    "multiple_person_dataset": "multiple_person_dataset",
    "not sure": "not_sure",
    "not_sure": "not_sure",
}

DEFAULT_COUNTRY_CODE = "91"
DEFAULT_TIMEZONE = "Asia/Kolkata"
MISSING_TEXT = {"", "nan", "none", "null", "nat", "<na>"}

CDR_SUBJECT_ALIASES = (
    "Mobile_No",
    "caller_id",
    "caller",
    "caller_msisdn",
    "subscriber_id",
    "calling_party",
    "a_party",
    "A_Party_No",
)
CDR_CONTACT_ALIASES = (
    "Other_Party_No",
    "receiver_msisdn",
    "receiver_id",
    "receiver",
    "callee_id",
    "called_party",
    "b_party",
    "B_Party_No",
)
IPDR_SUBSCRIBER_ALIASES = (
    "Landline/MSISDN/MDN/Leased Circuit ID for Internet Access",
    "subscriber_id",
    "msisdn",
    "Mobile_No",
    "user_id",
    "User Id for internet Access based on authentication",
)
TIMESTAMP_ALIASES = (
    "timestamp",
    "call_timestamp",
    "session_timestamp",
    "datetime",
    "event_time",
    "start_datetime",
    "TIME1 (dd/MM/yyyy HH:mm:ss)",
)
CDR_DATE_ALIASES = ("Call_Date", "call_date", "date")
CDR_TIME_ALIASES = (
    "Call_Initiation_Time(CIT)",
    "call_time",
    "start_time",
    "time",
)
IPDR_DATE_ALIASES = (
    "Start Date of Public IP Address allocation (dd/mm/yyyy)",
    "start_date",
    "date",
)
IPDR_TIME_ALIASES = (
    "IST Start Time of Public IP address allocation (hh:mm:ss)",
    "start_time",
    "time",
)
TIMEZONE_ALIASES = ("timezone", "time_zone", "tz", "utc_offset", "UTC Offset")

_EXCEL_WRAPPED = re.compile(r'^="((?:""|[^"])*)"$')
_PHONE_PUNCTUATION = re.compile(r"^[+()\-\s.0-9]+$")
_PHONE_COLUMN_WORDS = re.compile(
    r"(?:phone|mobile|msisdn|contact|party|a_party|b_party|landline|"
    r"caller|callee|calling|called|receiver|sender)",
    re.IGNORECASE,
)
_EXPLICIT_TZ = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$", re.IGNORECASE)


def normalize_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def is_phone_field(field_name: object | None) -> bool:
    """Return whether country-code normalization is appropriate for a field.

    Generic fields such as ``subscriber_id``, ``source_identifier``, IMEI,
    IMSI, names, IP addresses, and destinations are intentionally excluded.
    Callers can pass ``phone=True`` to override this when a custom column is
    known to contain phone numbers.
    """

    if field_name is None:
        return False
    text = str(field_name)
    return bool(_PHONE_COLUMN_WORDS.search(text))


def _text(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFKC", str(value)).replace("\u200b", "")
    # Excel commonly exports text identifiers as ="001234".  Decode doubled
    # quotes as csv/Excel would, and allow a second wrapper after whitespace.
    for _ in range(2):
        match = _EXCEL_WRAPPED.fullmatch(text.strip())
        if not match:
            break
        text = match.group(1).replace('""', '"').strip()
    return text.strip()


def _looks_phone(value: str) -> bool:
    return bool(value and _PHONE_PUNCTUATION.fullmatch(value) and sum(ch.isdigit() for ch in value) >= 7)


def normalize_identifier(
    value: object,
    *,
    field_name: object | None = None,
    phone: bool | None = None,
    country_code: str | None = DEFAULT_COUNTRY_CODE,
) -> str:
    """Normalize one identifier while keeping it a string.

    Whitespace and Excel wrappers are removed.  Phone punctuation and local
    versus international forms are canonicalized only for phone fields (or
    when ``phone=True``).  No name, filename, IP, destination, or arbitrary
    numeric identifier is used to infer a person.
    """

    text = _text(value)
    if not text or text.casefold() in MISSING_TEXT:
        return ""
    if re.fullmatch(r"[0-9]+\.0+", text):
        text = text.split(".", 1)[0]

    use_phone = is_phone_field(field_name) if phone is None else bool(phone)
    if use_phone and _looks_phone(text):
        digits = re.sub(r"\D", "", text)
        if digits.startswith("00"):
            digits = digits[2:]
        code = re.sub(r"\D", "", str(country_code or ""))
        if code and len(digits) == 10:
            digits = code + digits
        elif code and len(digits) == 11 and digits.startswith("0"):
            digits = code + digits[1:]
        return digits

    # Preserve meaningful characters in non-phone identifiers.  Collapsing
    # internal whitespace handles pasted values without turning names/IPs or
    # destinations into identity keys.
    return " ".join(text.split()).casefold()


def identifier_variants(
    value: object,
    *,
    field_name: object | None = None,
    country_code: str | None = DEFAULT_COUNTRY_CODE,
) -> tuple[str, ...]:
    """Return safe lookup variants in deterministic priority order."""

    generic = normalize_identifier(value, field_name=field_name, phone=False, country_code=country_code)
    contextual = normalize_identifier(value, field_name=field_name, phone=None, country_code=country_code)
    phone_variant = (
        normalize_identifier(value, field_name=field_name, phone=True, country_code=country_code)
        if is_phone_field(field_name)
        else ""
    )
    result: list[str] = []
    for candidate in (contextual, generic, phone_variant):
        if candidate and candidate not in result:
            result.append(candidate)
    return tuple(result)


def normalize_answer(value: object) -> str | None:
    if value is None:
        return None
    key = " ".join(_text(value).casefold().replace("_", " ").split())
    return IDENTITY_ANSWER_ALIASES.get(key)


def normalize_generation_mode(value: object | None) -> str | None:
    if value is None or not _text(value):
        return None
    key = _text(value).casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "combined": "combined",
        "combined_same_person": "combined",
        "same_person": "combined",
        "cdr_only": "cdr_only",
        "separate_sources": "separate_sources",
        "different_people": "separate_sources",
        "multi_person": "multi_person_conditioned",
        "multiple_person_dataset": "multi_person_conditioned",
        "multi_person_conditioned": "multi_person_conditioned",
        "single_person_template": "single_person_template",
        "template_expansion": "single_person_template",
        "ipdr_only": "ipdr_only_correlation",
        "ipdr_only_correlation": "ipdr_only_correlation",
        "review_required": "review_required",
    }
    return aliases.get(key, key)


def _resolve_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    values = [str(column).strip() for column in columns]
    exact = {value: value for value in values}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
    normalized = {normalize_column_name(value): value for value in values}
    for alias in aliases:
        match = normalized.get(normalize_column_name(alias))
        if match is not None:
            return match
    return None


def _header_matches(
    columns: list[str],
    kind: str,
    overrides: Mapping[str, object] | None = None,
) -> bool:
    overrides = overrides or {}
    if kind == "cdr":
        requested_subject = overrides.get("subject") or overrides.get("caller")
        requested_contact = overrides.get("contact") or overrides.get("receiver")
        if requested_subject and requested_contact:
            if _resolve_column(columns, (str(requested_subject),)) and _resolve_column(columns, (str(requested_contact),)):
                return True
    elif kind == "ipdr":
        requested_subscriber = overrides.get("subscriber")
        if requested_subscriber and _resolve_column(columns, (str(requested_subscriber),)):
            return True
    if kind == "cdr":
        return bool(_resolve_column(columns, CDR_SUBJECT_ALIASES) and _resolve_column(columns, CDR_CONTACT_ALIASES))
    if kind == "ipdr":
        return bool(_resolve_column(columns, IPDR_SUBSCRIBER_ALIASES))
    raise ValueError(f"Unsupported source kind: {kind}")


def inspect_csv_header(
    path: str | Path,
    kind: str,
    overrides: Mapping[str, object] | None = None,
) -> tuple[int, list[str]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{kind.upper()} CSV not found: {source}")
    with source.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader):
            columns = [str(value).strip() for value in row]
            if _header_matches(columns, kind, overrides):
                return row_index, columns
            if row_index >= 100:
                break
    try:
        columns = list(pd.read_csv(source, nrows=0, dtype=str, encoding="utf-8-sig", encoding_errors="replace").columns)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{kind.upper()} CSV is empty: {source}") from error
    return 0, [str(column).strip() for column in columns]


def resolve_source_schema(
    path: str | Path,
    kind: str,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve subject/contact and time columns without looking at filenames."""

    overrides = dict(overrides or {})
    source = Path(path).expanduser().resolve()
    header_row, columns = inspect_csv_header(source, kind, overrides)
    if kind == "cdr":
        subject_aliases = CDR_SUBJECT_ALIASES
        contact_aliases = CDR_CONTACT_ALIASES
        date_aliases, time_aliases = CDR_DATE_ALIASES, CDR_TIME_ALIASES
    elif kind == "ipdr":
        subject_aliases = IPDR_SUBSCRIBER_ALIASES
        contact_aliases = ()
        date_aliases, time_aliases = IPDR_DATE_ALIASES, IPDR_TIME_ALIASES
    else:
        raise ValueError(f"Unsupported source kind: {kind}")

    def pick(key: str, aliases: Iterable[str], required: bool = False) -> str | None:
        requested = overrides.get(key)
        result = _resolve_column(columns, (str(requested),)) if requested else _resolve_column(columns, aliases)
        if required and result is None:
            raise ValueError(
                f"{kind.upper()} CSV is missing the {key} column. Available columns: {columns}"
            )
        return result

    subject_key = "subject" if kind == "cdr" else "subscriber"
    subject = pick(subject_key, subject_aliases, required=True)
    contact = pick("contact", contact_aliases, required=kind == "cdr") if kind == "cdr" else None
    timestamp = pick("timestamp", TIMESTAMP_ALIASES)
    date = pick("date", date_aliases)
    time = pick("time", time_aliases)
    if timestamp is None and date is None:
        # Time is evidence, not an identity key.  Keep an unavailable time
        # window in the report rather than inventing a relationship or failing
        # an otherwise inspectable file.  A date-only column remains useful
        # for an observation period and is parsed at midnight in the supplied
        # timezone.
        time = None
    timezone_column = pick("timezone", TIMEZONE_ALIASES)
    schema: dict[str, object] = {
        "kind": kind,
        "path": str(source),
        "header_row": header_row,
        "all_columns": columns,
        "subject": subject,
        "timestamp": timestamp,
        "date": date,
        "time": time,
        "timezone": timezone_column,
    }
    if kind == "cdr":
        schema.update({"caller": subject, "contact": contact, "receiver": contact})
    else:
        schema["subscriber"] = subject
    # Common fields are optional but useful to downstream generators and
    # correlation tools.  They are resolved here only when present.
    optional_aliases = {
        "duration": ("duration_seconds", "Call_Duration", "call_duration", "duration"),
        "destination": ("destination_ip", "Destination IP Address", "destination", "remote_ip"),
        "destination_port": ("destination_port", "Destination Port", "Destination_Port", "DestinationPort"),
        "cell": ("cell_id", "First_Cell_id", "First CELL ID", "tower_id"),
        "megabytes": ("megabytes_transferred", "megabytes", "data_mb"),
        "upload": ("Data Volume Up Link", "upload_kb", "uplink_kb"),
        "download": ("Data Volume Down Link", "download_kb", "downlink_kb"),
        "session_duration": ("Session Duration (Seconds)", "duration_seconds", "session_duration"),
        "event_type": ("Call_Type", "Service_Type", "event_type", "type"),
    }
    for key, aliases in optional_aliases.items():
        if key not in schema:
            schema[key] = pick(key, aliases)
    return schema


def schema_use_columns(schema: Mapping[str, object]) -> list[str]:
    ignored = {"kind", "path", "header_row", "all_columns"}
    columns: list[str] = []
    for key, value in schema.items():
        if key not in ignored and isinstance(value, str) and value and value not in columns:
            columns.append(value)
    return columns


def iter_source_chunks(schema: Mapping[str, object], chunk_size: int = 100_000) -> Iterator[pd.DataFrame]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    columns = schema_use_columns(schema)
    kwargs: dict[str, object] = {
        "skiprows": int(schema.get("header_row", 0)),
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "chunksize": chunk_size,
        "encoding": "utf-8-sig",
        "encoding_errors": "replace",
    }
    if columns:
        kwargs["usecols"] = columns
    try:
        yield from pd.read_csv(str(schema["path"]), **kwargs)
    except pd.errors.EmptyDataError:
        return


def _parse_timestamp_values(
    frame: pd.DataFrame,
    schema: Mapping[str, object],
    default_timezone: str = DEFAULT_TIMEZONE,
) -> tuple[pd.Series, dict[str, object]]:
    timestamp_column = schema.get("timestamp")
    if timestamp_column:
        raw = frame[str(timestamp_column)].map(_text)
    elif schema.get("date"):
        raw = frame[str(schema["date"])].map(_text)
        if schema.get("time"):
            raw = raw + " " + frame[str(schema["time"])].map(_text)
    else:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"), {
            "available": False,
            "timezone_status": "unavailable",
            "assumed_timezone": None,
            "total_values": len(frame),
            "valid_values": 0,
            "invalid_or_missing_values": len(frame),
        }

    nonempty = raw[raw != ""]
    if nonempty.empty:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"), {
            "available": False,
            "timezone_status": "unavailable",
            "assumed_timezone": default_timezone,
            "timezone_column": schema.get("timezone"),
            "invalid_timezone_rows": 0,
            "total_values": len(frame),
            "valid_values": 0,
            "invalid_or_missing_values": len(frame),
        }

    year_first = bool(nonempty.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() > 0.5)
    explicit_mask = raw.str.contains(_EXPLICIT_TZ, na=False)
    explicit_mask &= raw != ""
    has_explicit = bool(explicit_mask.any())
    naive_mask = (raw != "") & ~explicit_mask
    has_naive = bool(naive_mask.any())
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    invalid_timezone_rows = 0

    if has_explicit:
        explicit_values = pd.to_datetime(
            raw.loc[explicit_mask],
            errors="coerce",
            dayfirst=not year_first,
            yearfirst=year_first,
            format="mixed",
            utc=True,
        )
        parsed.loc[explicit_mask] = explicit_values

    timezone_column = schema.get("timezone")
    if timezone_column and str(timezone_column) in frame.columns:
        timezone_values = frame[str(timezone_column)].map(_text)
    else:
        timezone_values = pd.Series(default_timezone, index=raw.index, dtype=object)

    def timezone_for(value: object):
        text = _text(value)
        if not text:
            try:
                return ZoneInfo(default_timezone), True
            except ZoneInfoNotFoundError as error:
                raise ValueError(f"Unknown timezone {default_timezone!r}") from error
        upper = text.upper()
        if upper in {"Z", "UTC", "GMT"}:
            return dt_timezone.utc, False
        offset = re.fullmatch(r"(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?", upper)
        if offset:
            hours = int(offset.group(2))
            minutes = int(offset.group(3) or 0)
            if hours <= 23 and minutes <= 59:
                seconds = (hours * 60 + minutes) * 60
                if offset.group(1) == "-":
                    seconds *= -1
                return dt_timezone(timedelta(seconds=seconds)), False
        try:
            return ZoneInfo(text), False
        except ZoneInfoNotFoundError:
            return None, False

    if has_naive:
        naive_values = pd.to_datetime(
            raw.loc[naive_mask],
            errors="coerce",
            dayfirst=not year_first,
            yearfirst=year_first,
            format="mixed",
        )
        # Localize per timezone group so a provided row-level timezone is
        # honored.  Invalid zone names are counted as invalid temporal
        # evidence rather than silently interpreted as UTC.
        for timezone_text, indices in timezone_values.loc[naive_mask].groupby(timezone_values.loc[naive_mask]).groups.items():
            tzinfo, used_default = timezone_for(timezone_text)
            if tzinfo is None:
                invalid_timezone_rows += len(indices)
                continue
            try:
                localized = naive_values.loc[indices].dt.tz_localize(
                    tzinfo,
                    ambiguous="NaT",
                    nonexistent="NaT",
                ).dt.tz_convert("UTC")
            except (TypeError, ValueError):
                localized = pd.Series(pd.NaT, index=indices, dtype="datetime64[ns, UTC]")
            parsed.loc[indices] = localized

    valid_count = int(parsed.notna().sum())
    if invalid_timezone_rows:
        timezone_status = "invalid_timezone_values"
    elif timezone_column and has_naive:
        timezone_status = "mixed_explicit_and_per_row" if has_explicit else "per_row_timezone"
    elif has_explicit and has_naive:
        timezone_status = "mixed_explicit_and_assumed"
    elif has_explicit:
        timezone_status = "explicit"
    else:
        timezone_status = "naive_assumed"
    return parsed, {
        "available": valid_count > 0,
        "timezone_status": timezone_status,
        "assumed_timezone": default_timezone if has_naive and not timezone_column else None,
        "timezone_column": timezone_column,
        "invalid_timezone_rows": invalid_timezone_rows,
        "total_values": len(frame),
        "valid_values": valid_count,
        "invalid_or_missing_values": len(frame) - valid_count,
    }


def parse_timestamp_series(
    frame: pd.DataFrame,
    schema: Mapping[str, object],
    default_timezone: str = DEFAULT_TIMEZONE,
) -> pd.Series:
    """Public timestamp parser shared by generation and correlation."""

    return _parse_timestamp_values(frame, schema, default_timezone)[0]


@dataclass
class IdentityMapping:
    """Explicit subscriber/alias mapping with ambiguous aliases quarantined."""

    alias_to_canonical: dict[str, str] = field(default_factory=dict)
    ambiguous_aliases: set[str] = field(default_factory=set)
    source_path: str | None = None
    canonical_column: str | None = None
    country_code: str = DEFAULT_COUNTRY_CODE

    @classmethod
    def empty(cls, country_code: str = DEFAULT_COUNTRY_CODE) -> "IdentityMapping":
        return cls(country_code=country_code)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[object, object] | Iterable[Mapping[str, object]] | None,
        *,
        country_code: str = DEFAULT_COUNTRY_CODE,
    ) -> "IdentityMapping":
        result = cls.empty(country_code)
        if mapping is None:
            return result
        if isinstance(mapping, Mapping):
            items = mapping.items()
            for left, right in items:
                if isinstance(right, (list, tuple, set)):
                    for alias in right:
                        result._register(alias, left)
                elif isinstance(right, Mapping):
                    canonical = right.get("canonical") or right.get("subscriber") or left
                    aliases = right.get("aliases") or right.get("alias") or []
                    if not isinstance(aliases, (list, tuple, set)):
                        aliases = [aliases]
                    for alias in aliases:
                        result._register(alias, canonical)
                else:
                    # The API contract uses alias -> canonical for a simple
                    # JSON object.  Wide subscriber maps are handled below.
                    result._register(left, right)
        else:
            for record in mapping:
                canonical = record.get("canonical") or record.get("canonical_id") or record.get("subscriber_id") or record.get("subscriber")
                alias = record.get("alias") or record.get("alias_id") or record.get("source_identifier") or record.get("source")
                if canonical is not None and alias is not None:
                    result._register(alias, canonical)
        result._finalize()
        return result

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        country_code: str = DEFAULT_COUNTRY_CODE,
    ) -> "IdentityMapping":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Identity mapping CSV not found: {source}")
        frame = pd.read_csv(
            source,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
            encoding_errors="replace",
        )
        columns = [str(column).strip() for column in frame.columns]
        canonical_aliases = (
            "canonical_id",
            "canonical",
            "person_id",
            "subscriber_id",
            "subscriber",
            "target_id",
            "target",
        )
        alias_aliases = (
            "alias",
            "alias_id",
            "source_identifier",
            "source_id",
            "msisdn",
            "mobile_no",
            "phone",
            "alternate_msisdn",
            "from",
        )
        canonical_column = _resolve_column(columns, canonical_aliases)
        alias_column = _resolve_column(columns, alias_aliases)
        if canonical_column is None and len(columns) >= 2:
            canonical_column, alias_column = columns[0], columns[1]
        if canonical_column is None:
            raise ValueError(
                "Identity mapping needs a canonical/subscriber column and an alias column. "
                f"Available columns: {columns}"
            )
        identity_columns = [canonical_column]
        for column in columns:
            if column not in identity_columns and (
                column == alias_column
                or normalize_column_name(column) in {
                    normalize_column_name(value)
                    for value in (
                        "msisdn",
                        "Mobile_No",
                        "source_identifier",
                        "alternate_msisdn",
                        "imei",
                        "imsi",
                        "alias",
                        "alias_id",
                    )
                }
            ):
                identity_columns.append(column)
        result = cls(
            source_path=str(source),
            canonical_column=canonical_column,
            country_code=country_code,
        )
        for record in frame[identity_columns].itertuples(index=False, name=None):
            canonical_raw = record[0]
            canonical = normalize_identifier(
                canonical_raw,
                field_name=canonical_column,
                country_code=country_code,
            )
            if not canonical:
                canonical = next(
                    (
                        normalize_identifier(value, field_name=column, country_code=country_code)
                        for column, value in zip(identity_columns[1:], record[1:])
                        if normalize_identifier(value, field_name=column, country_code=country_code)
                    ),
                    "",
                )
            if not canonical:
                continue
            for column, value in zip(identity_columns, record):
                if normalize_identifier(value, field_name=column, country_code=country_code):
                    result._register(value, canonical, field_name=column)
        result._finalize()
        return result

    def _register(self, alias: object, canonical: object, *, field_name: object | None = None) -> None:
        canonical_key = normalize_identifier(
            canonical,
            field_name=field_name,
            country_code=self.country_code,
        )
        if not canonical_key:
            return
        candidates = identifier_variants(alias, field_name=field_name, country_code=self.country_code)
        # Keep candidates in a private set until finalization so a collision
        # cannot silently select the first row.
        registry = getattr(self, "_candidate_map", None)
        if registry is None:
            registry = defaultdict(set)
            setattr(self, "_candidate_map", registry)
        for candidate in candidates:
            registry[candidate].add(canonical_key)

    def _finalize(self) -> None:
        registry = getattr(self, "_candidate_map", {})
        self.alias_to_canonical = {
            alias: next(iter(canonicals))
            for alias, canonicals in registry.items()
            if len(canonicals) == 1
        }
        self.ambiguous_aliases = {
            alias for alias, canonicals in registry.items() if len(canonicals) > 1
        }
        if hasattr(self, "_candidate_map"):
            delattr(self, "_candidate_map")

    @property
    def ambiguous_alias_count(self) -> int:
        return len(self.ambiguous_aliases)

    def canonicalize_with_reason(
        self,
        value: object,
        *,
        field_name: object | None = None,
    ) -> tuple[str, str]:
        variants = identifier_variants(value, field_name=field_name, country_code=self.country_code)
        for candidate in variants:
            if candidate in self.ambiguous_aliases:
                return "", "ambiguous_mapping"
            mapped = self.alias_to_canonical.get(candidate)
            if mapped:
                return mapped, "explicit_mapping"
        return (variants[0], "normalized_identifier") if variants else ("", "missing")

    def canonicalize(self, value: object, *, field_name: object | None = None) -> str:
        return self.canonicalize_with_reason(value, field_name=field_name)[0]

    def canonicalize_series(
        self,
        series: pd.Series,
        *,
        field_name: object | None = None,
    ) -> pd.Series:
        return series.map(lambda value: self.canonicalize(value, field_name=field_name))

    def public(self) -> dict[str, object]:
        mapping_fingerprint = hashlib.sha256(
            _stable_json(
                {
                    "aliases": sorted(self.alias_to_canonical.items()),
                    "ambiguous": sorted(self.ambiguous_aliases),
                    "country_code": self.country_code,
                }
            ).encode("utf-8")
        ).hexdigest()
        return {
            "source_path": self.source_path,
            "canonical_column": self.canonical_column,
            "alias_count": len(self.alias_to_canonical),
            "ambiguous_alias_count": self.ambiguous_alias_count,
            "ambiguous_aliases": sorted(self.ambiguous_aliases),
            "mapping_fingerprint": mapping_fingerprint,
        }

    def metadata(self) -> dict[str, object]:
        """Return auditable mapping metadata, including explicit alias pairs."""

        result = self.public()
        result["aliases"] = dict(sorted(self.alias_to_canonical.items()))
        return result


def load_identity_mapping(
    path: str | Path | None = None,
    inline: Mapping[object, object] | Iterable[Mapping[str, object]] | None = None,
    *,
    country_code: str = DEFAULT_COUNTRY_CODE,
) -> IdentityMapping:
    if path:
        result = IdentityMapping.from_csv(path, country_code=country_code)
        if inline:
            extra = IdentityMapping.from_mapping(inline, country_code=country_code)
            combined = IdentityMapping.empty(country_code)
            for alias, canonical in result.alias_to_canonical.items():
                combined._register(alias, canonical)
            for alias, canonical in extra.alias_to_canonical.items():
                combined._register(alias, canonical)
            combined.source_path = result.source_path
            combined.canonical_column = result.canonical_column
            combined._finalize()
            return combined
        return result
    return IdentityMapping.from_mapping(inline, country_code=country_code)


def _empty_source(kind: str, present: bool = False) -> dict[str, object]:
    return {
        "kind": kind,
        "present": present,
        "row_count": 0,
        "chunk_count": 0,
        "subject_column": None,
        "subscriber_column": None,
        "contact_column": None,
        "subject_count": 0,
        "subject_identifiers": [],
        "subject_counts": {},
        "subject_date_ranges": {},
        "contact_count": 0,
        "contact_identifiers": [],
        "contact_counts": {},
        "contact_only_identifiers": [],
        "missing_identifier_rows": 0,
        "ambiguous_identifier_rows": 0,
        "invalid_timestamp_rows": 0,
        "invalid_timezone_rows": 0,
        "timestamp": {
            "available": False,
            "timezone_status": "unavailable",
            "assumed_timezone": None,
            "total_values": 0,
            "valid_values": 0,
            "invalid_or_missing_values": 0,
        },
        "date_range": {"start": None, "end": None},
        "schema": None,
    }


def _format_timestamp(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _update_range(target: dict[str, object], timestamp: object) -> None:
    if timestamp is None or pd.isna(timestamp):
        return
    current_start = target.get("start")
    current_end = target.get("end")
    value = _format_timestamp(timestamp)
    if value is None:
        return
    if current_start is None or value < str(current_start):
        target["start"] = value
    if current_end is None or value > str(current_end):
        target["end"] = value


def _scan_source(
    path: str | Path,
    kind: str,
    mapping: IdentityMapping,
    *,
    overrides: Mapping[str, object] | None = None,
    chunk_size: int = 100_000,
    default_timezone: str = DEFAULT_TIMEZONE,
) -> dict[str, object]:
    schema = resolve_source_schema(path, kind, overrides)
    result = _empty_source(kind, True)
    result["schema"] = {
        key: value for key, value in schema.items() if key != "all_columns"
    }
    result["subject_column"] = schema.get("subject")
    if kind == "ipdr":
        result["subscriber_column"] = schema.get("subject")
    result["contact_column"] = schema.get("contact")
    subject_counts: Counter[str] = Counter()
    contact_counts: Counter[str] = Counter()
    date_ranges: dict[str, dict[str, object]] = defaultdict(lambda: {"start": None, "end": None})
    overall_range = {"start": None, "end": None}
    all_contacts: set[str] = set()
    subject_ids: set[str] = set()
    timestamp_stats: list[dict[str, object]] = []
    for chunk_number, chunk in enumerate(iter_source_chunks(schema, chunk_size), start=1):
        result["chunk_count"] = chunk_number
        result["row_count"] = int(result["row_count"]) + len(chunk)
        subject_column = str(schema["subject"])
        subjects: list[str] = []
        ambiguous_subjects = 0
        for value in chunk[subject_column].tolist():
            canonical, reason = mapping.canonicalize_with_reason(value, field_name=subject_column)
            subjects.append(canonical)
            if reason == "ambiguous_mapping":
                ambiguous_subjects += 1
        result["ambiguous_identifier_rows"] = int(result["ambiguous_identifier_rows"]) + ambiguous_subjects
        result["missing_identifier_rows"] = int(result["missing_identifier_rows"]) + sum(not value for value in subjects)
        timestamps, time_stats = _parse_timestamp_values(chunk, schema, default_timezone)
        timestamp_stats.append(time_stats)
        result["invalid_timestamp_rows"] = int(result["invalid_timestamp_rows"]) + int(time_stats["invalid_or_missing_values"])
        result["invalid_timezone_rows"] = int(result["invalid_timezone_rows"]) + int(time_stats.get("invalid_timezone_rows", 0))
        for identifier, timestamp in zip(subjects, timestamps.tolist()):
            if not identifier:
                continue
            subject_ids.add(identifier)
            subject_counts[identifier] += 1
            _update_range(date_ranges[identifier], timestamp)
            _update_range(overall_range, timestamp)

        if kind == "cdr" and schema.get("contact"):
            contact_column = str(schema["contact"])
            for value in chunk[contact_column].tolist():
                canonical, reason = mapping.canonicalize_with_reason(value, field_name=contact_column)
                if reason == "ambiguous_mapping":
                    result["ambiguous_identifier_rows"] = int(result["ambiguous_identifier_rows"]) + 1
                if not canonical:
                    continue
                all_contacts.add(canonical)
                contact_counts[canonical] += 1

    result["subject_identifiers"] = sorted(subject_ids)
    result["subject_count"] = len(subject_ids)
    result["subject_counts"] = dict(sorted(subject_counts.items()))
    result["subject_date_ranges"] = dict(sorted(date_ranges.items()))
    result["contact_identifiers"] = sorted(all_contacts)
    result["contact_count"] = len(all_contacts)
    result["contact_counts"] = dict(sorted(contact_counts.items()))
    result["contact_only_identifiers"] = sorted(all_contacts - subject_ids)
    result["date_range"] = overall_range
    if timestamp_stats:
        total_values = sum(int(stat["total_values"]) for stat in timestamp_stats)
        valid_values = sum(int(stat["valid_values"]) for stat in timestamp_stats)
        invalid_values = sum(int(stat["invalid_or_missing_values"]) for stat in timestamp_stats)
        invalid_timezone_values = sum(int(stat.get("invalid_timezone_rows", 0)) for stat in timestamp_stats)
        statuses = {str(stat["timezone_status"]) for stat in timestamp_stats}
        result["timestamp"] = {
            "available": valid_values > 0,
            "timezone_status": next(iter(statuses)) if len(statuses) == 1 else "mixed",
            "assumed_timezone": default_timezone if any(stat["assumed_timezone"] for stat in timestamp_stats) else None,
            "total_values": total_values,
            "valid_values": valid_values,
            "invalid_or_missing_values": invalid_values,
            "invalid_timezone_rows": invalid_timezone_values,
        }
    return result


def _file_fingerprint(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _temporal_evidence(cdr: Mapping[str, object], ipdr: Mapping[str, object]) -> dict[str, object]:
    cdr_range = cdr.get("date_range") or {}
    ipdr_range = ipdr.get("date_range") or {}
    cdr_start, cdr_end = cdr_range.get("start"), cdr_range.get("end")
    ipdr_start, ipdr_end = ipdr_range.get("start"), ipdr_range.get("end")
    if not cdr_start or not cdr_end or not ipdr_start or not ipdr_end:
        return {
            "available": False,
            "overlaps": None,
            "status": "unavailable",
            "cdr_range": cdr_range,
            "ipdr_range": ipdr_range,
            "overlap_range": {"start": None, "end": None},
            "explanation": "An unavailable or unparseable observation period is missing temporal evidence, not evidence of no relationship.",
        }
    overlap_start = max(str(cdr_start), str(ipdr_start))
    overlap_end = min(str(cdr_end), str(ipdr_end))
    overlaps = overlap_start <= overlap_end
    return {
        "available": True,
        "overlaps": overlaps,
        "status": "overlap" if overlaps else "non_overlapping",
        "cdr_range": cdr_range,
        "ipdr_range": ipdr_range,
        "overlap_range": {
            "start": overlap_start if overlaps else None,
            "end": overlap_end if overlaps else None,
        },
        "explanation": (
            "Observation periods overlap. This is temporal context, not identity proof."
            if overlaps
            else "Observation periods do not overlap; the absence of overlap is not evidence that the files describe different people."
        ),
    }


def validate_cdr_ipdr(
    cdr_path: str | Path | None = None,
    ipdr_path: str | Path | None = None,
    *,
    mapping: IdentityMapping | Mapping[object, object] | Iterable[Mapping[str, object]] | None = None,
    mapping_path: str | Path | None = None,
    overrides: Mapping[str, object] | None = None,
    chunk_size: int = 100_000,
    default_timezone: str = DEFAULT_TIMEZONE,
    country_code: str = DEFAULT_COUNTRY_CODE,
) -> dict[str, object]:
    """Scan every CDR/IPDR row and return auditable identity evidence."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if isinstance(mapping, IdentityMapping):
        resolver = mapping
    else:
        resolver = load_identity_mapping(mapping_path, mapping, country_code=country_code)
    overrides = dict(overrides or {})
    cdr_overrides = overrides.get("cdr", {}) if isinstance(overrides.get("cdr", {}), Mapping) else {}
    ipdr_overrides = overrides.get("ipdr", {}) if isinstance(overrides.get("ipdr", {}), Mapping) else {}
    cdr = _empty_source("cdr", cdr_path is not None)
    ipdr = _empty_source("ipdr", ipdr_path is not None)
    fingerprints: dict[str, object] = {}
    if cdr_path is not None:
        cdr = _scan_source(
            cdr_path,
            "cdr",
            resolver,
            overrides=cdr_overrides,
            chunk_size=chunk_size,
            default_timezone=default_timezone,
        )
        fingerprints["cdr"] = _file_fingerprint(cdr_path)
    if ipdr_path is not None:
        ipdr = _scan_source(
            ipdr_path,
            "ipdr",
            resolver,
            overrides=ipdr_overrides,
            chunk_size=chunk_size,
            default_timezone=default_timezone,
        )
        fingerprints["ipdr"] = _file_fingerprint(ipdr_path)

    cdr_subjects = set(cdr["subject_identifiers"])
    ipdr_subjects = set(ipdr["subject_identifiers"])
    matched = sorted(cdr_subjects & ipdr_subjects)
    unmatched_cdr = sorted(cdr_subjects - ipdr_subjects)
    unmatched_ipdr = sorted(ipdr_subjects - cdr_subjects)
    temporal = _temporal_evidence(cdr, ipdr) if cdr_path is not None and ipdr_path is not None else {
        "available": False,
        "overlaps": None,
        "status": "not_applicable",
        "cdr_range": cdr.get("date_range"),
        "ipdr_range": ipdr.get("date_range"),
        "overlap_range": {"start": None, "end": None},
        "explanation": "A single supplied source has no cross-source temporal comparison.",
    }
    matched_details = []
    for identifier in matched:
        matched_details.append(
            {
                "identifier": identifier,
                "cdr_rows": cdr["subject_counts"].get(identifier, 0),
                "ipdr_rows": ipdr["subject_counts"].get(identifier, 0),
                "cdr_date_range": cdr["subject_date_ranges"].get(identifier),
                "ipdr_date_range": ipdr["subject_date_ranges"].get(identifier),
            }
        )
    both_supplied = cdr_path is not None and ipdr_path is not None
    validation_payload: dict[str, object] = {
        "validation_version": 1,
        "checked_at": pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z"),
        "status": "checked",
        "chunk_size": chunk_size,
        "all_records_scanned": True,
        "sources": {"cdr": cdr, "ipdr": ipdr},
        "mapping": resolver.public(),
        "matching": {
            "cdr_subject_count": len(cdr_subjects),
            "ipdr_subscriber_count": len(ipdr_subjects),
            "matched_count": len(matched),
            "unmatched_cdr_count": len(unmatched_cdr),
            "unmatched_ipdr_count": len(unmatched_ipdr),
            "matched_identifiers": matched,
            "unmatched_cdr_subjects": unmatched_cdr,
            "unmatched_ipdr_subscribers": unmatched_ipdr,
            "matched_details": matched_details,
            "cdr_subject_coverage": (len(matched) / len(cdr_subjects)) if cdr_subjects else None,
            "ipdr_subscriber_coverage": (len(matched) / len(ipdr_subjects)) if ipdr_subjects else None,
        },
        "temporal": temporal,
        "identity_evidence": {
            "identifier_agreement": bool(matched),
            "identifier_agreement_supports_matching": True,
            "identifier_agreement_proves_personal_identity": False,
            "statement": "Identifier agreement supports matching; it does not prove personal identity. Names, filenames, shared IPs, and destinations are not identity evidence.",
        },
        "fingerprints": fingerprints,
    }
    validation_payload["validation_fingerprint"] = hashlib.sha256(
        _stable_json(
            {
                "fingerprints": fingerprints,
                "default_timezone": default_timezone,
                "country_code": country_code,
                "schemas": {
                    "cdr": cdr.get("schema"),
                    "ipdr": ipdr.get("schema"),
                },
                "mapping": resolver.public(),
            }
        ).encode("utf-8")
    ).hexdigest()
    if not both_supplied:
        validation_payload["decision_status"] = "single_source"
    else:
        validation_payload["decision_status"] = "awaiting_identity_answer"
    return validation_payload


def apply_identity_answer(
    validation: Mapping[str, object],
    answer: object | None,
    *,
    generation_mode: object | None = None,
    confirmation: bool = False,
) -> dict[str, object]:
    """Turn the user's answer into a server-enforceable processing decision."""

    normalized = normalize_answer(answer)
    mode = normalize_generation_mode(generation_mode)
    matching = validation.get("matching", {}) if isinstance(validation.get("matching"), Mapping) else {}
    sources = validation.get("sources", {}) if isinstance(validation.get("sources"), Mapping) else {}
    cdr = sources.get("cdr", {}) if isinstance(sources.get("cdr"), Mapping) else {}
    ipdr = sources.get("ipdr", {}) if isinstance(sources.get("ipdr"), Mapping) else {}
    both_supplied = bool(cdr.get("present") and ipdr.get("present"))
    blockers: list[str] = []
    warnings: list[str] = []
    operations: list[str] = []

    if both_supplied and normalized is None:
        return {
            "allowed": False,
            "answer": None,
            "generation_mode": mode,
            "decision_status": "awaiting_identity_answer",
            "blocking_reasons": ["Choose one identity answer before processing both sources."],
            "warnings": [],
            "available_operations": ["cdr_only_generation", "ipdr_only_correlation", "provide_mapping_or_correct_files"],
            "confirmation_recorded": bool(confirmation),
            "confirmation_is_not_an_override": True,
        }
    if answer is not None and normalized is None:
        blockers.append("Identity answer must be Same person, Different people, Multiple-person dataset, or Not sure.")

    if not both_supplied:
        if cdr.get("present"):
            if int(cdr.get("subject_count", 0)) < 1:
                blockers.append("The CDR contains no usable subject identifiers; CDR-only generation cannot start.")
            operations.append("cdr_only_generation")
            selected_mode = mode or "cdr_only"
            if selected_mode not in {"cdr_only", "single_person_template"}:
                blockers.append("A CDR-only input can only use CDR-only generation or explicit single-person template mode.")
        elif ipdr.get("present"):
            operations.append("ipdr_only_correlation")
            selected_mode = mode or "ipdr_only_correlation"
            if selected_mode != "ipdr_only_correlation":
                blockers.append("An IPDR-only input can only use IPDR-only correlation.")
        else:
            selected_mode = mode
            blockers.append("Supply a CDR or IPDR file.")
        return {
            "allowed": not blockers,
            "answer": normalized,
            "generation_mode": selected_mode,
            "decision_status": "allowed" if not blockers else "blocked",
            "blocking_reasons": blockers,
            "warnings": warnings,
            "available_operations": operations,
            "confirmation_recorded": bool(confirmation),
            "confirmation_is_not_an_override": True,
        }

    cdr_count = int(matching.get("cdr_subject_count", 0))
    ipdr_count = int(matching.get("ipdr_subscriber_count", 0))
    matched_count = int(matching.get("matched_count", 0))
    unmatched_cdr_count = int(matching.get("unmatched_cdr_count", 0))
    unmatched_ipdr_count = int(matching.get("unmatched_ipdr_count", 0))
    ambiguous_count = int((validation.get("mapping") or {}).get("ambiguous_alias_count", 0)) if isinstance(validation.get("mapping"), Mapping) else 0
    if cdr_count < 1:
        blockers.append("The CDR contains no usable subject identifiers; no CDR-backed generation operation is available.")

    if normalized == "same_person":
        selected_mode = mode or "combined"
        if selected_mode not in {"combined", "single_person_template"}:
            blockers.append("Same person requires combined generation or the explicit single-person template mode.")
        if cdr_count != 1 or ipdr_count != 1 or matched_count != 1 or unmatched_cdr_count or unmatched_ipdr_count:
            blockers.append(
                "Same person requires one CDR subject and one IPDR subscriber with a consistent identifier mapping. "
                "The supplied evidence conflicts; correct the files or provide an explicit alias mapping before combining."
            )
        if ambiguous_count:
            blockers.append("An ambiguous alias mapping must be corrected before sources can be combined.")
        operations.append("combined_cdr_ipdr_generation")
    elif normalized == "different_people":
        selected_mode = mode or "separate_sources"
        if selected_mode in {"combined", "multi_person_conditioned"}:
            blockers.append("Different people cannot be combined into one identity-conditioned generation run.")
        operations.extend(["cdr_only_generation", "ipdr_only_correlation", "separate_source_reports"])
        warnings.append("CDR and IPDR identities remain separate; IPDR behavior is not assigned to CDR subjects.")
    elif normalized == "multiple_person_dataset":
        selected_mode = mode or "multi_person_conditioned"
        if selected_mode not in {"multi_person_conditioned", "separate_sources", "cdr_only"}:
            blockers.append("Multiple-person data requires per-subject matching or separate-source processing.")
        if matched_count < cdr_count or matched_count < ipdr_count:
            warnings.append(
                f"Partial coverage: {matched_count} matched subject(s), {unmatched_cdr_count} CDR-only, and {unmatched_ipdr_count} IPDR-only. "
                "Unmatched profiles will not inherit another person's IPDR behavior."
            )
        operations.extend(["per_subject_matching", "cdr_only_generation", "ipdr_only_correlation"])
    elif normalized == "not_sure":
        selected_mode = mode or "review_required"
        if selected_mode not in {"cdr_only", "ipdr_only_correlation", "separate_sources", "review_required"}:
            blockers.append("Not sure cannot enable combined generation. Provide a mapping/correction or choose a separate operation.")
        operations.extend(["show_detected_evidence", "provide_mapping_or_correct_files", "cdr_only_generation", "ipdr_only_correlation"])
        warnings.append("The evidence is displayed for review; confirmation alone cannot override contradictory identity data.")
    else:
        selected_mode = mode
        blockers.append("An explicit identity answer is required when both CDR and IPDR are supplied.")

    return {
        "allowed": not blockers,
        "answer": normalized,
        "answer_label": IDENTITY_ANSWER_LABELS.get(normalized or ""),
        "generation_mode": selected_mode,
        "decision_status": "allowed" if not blockers else "blocked",
        "blocking_reasons": blockers,
        "warnings": warnings,
        "available_operations": operations,
        "confirmation_recorded": bool(confirmation),
        "confirmation_is_not_an_override": True,
        "evidence_summary": {
            "cdr_subject_count": cdr_count,
            "ipdr_subscriber_count": ipdr_count,
            "matched_count": matched_count,
            "unmatched_cdr_count": unmatched_cdr_count,
            "unmatched_ipdr_count": unmatched_ipdr_count,
        },
    }


def plan_source_profiles(
    source_ids: Iterable[object],
    target_count: int,
    *,
    source_counts: Mapping[object, int] | None = None,
    generation_mode: object | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Create a reproducible, lineage-preserving source-profile plan.

    If there are fewer source callers than requested agents, every source
    caller is retained.  Extra synthetic identities round-robin over source
    profiles using a seed-derived offset.  A one-source expansion is allowed
    only in explicit ``single_person_template`` mode.
    """

    if int(target_count) < 1:
        raise ValueError("target_count must be positive")
    mode = normalize_generation_mode(generation_mode)
    unique: list[str] = []
    seen: set[str] = set()
    for value in source_ids:
        identifier = normalize_identifier(value)
        if identifier and identifier not in seen:
            unique.append(identifier)
            seen.add(identifier)
    if source_counts:
        unique.sort(key=lambda value: (-int(source_counts.get(value, source_counts.get(str(value), 0))), value))
    if not unique:
        raise ValueError("No non-empty source subject identifiers were found.")
    target = int(target_count)
    if target <= len(unique):
        selected = unique[:target]
        strategy = "source_profiles_only"
    else:
        if len(unique) == 1 and mode != "single_person_template":
            raise ValueError(
                "A one-person source requires explicit --generation-mode single_person_template; "
                "the generator will not silently propagate one profile."
            )
        selected = list(unique)
        for index in range(target - len(unique)):
            candidate = f"SYNTHETIC_AGENT_{index + 1:07d}"
            while candidate in seen:
                index += 1
                candidate = f"SYNTHETIC_AGENT_{index + 1:07d}"
            selected.append(candidate)
            seen.add(candidate)
        strategy = "explicit_single_person_template_round_robin" if len(unique) == 1 else "round_robin_source_profile_expansion"
    offset = int(seed) % len(unique)
    lineage: dict[str, str] = {}
    for index, agent_id in enumerate(selected):
        lineage[agent_id] = agent_id if index < len(unique) else unique[(index - len(unique) + offset) % len(unique)]
    return {
        "agent_ids": selected,
        "source_profile_for_agent": lineage,
        "source_profile_count": len(unique),
        "target_count": target,
        "strategy": strategy,
        "seed": int(seed),
        "all_source_profiles_preserved": all(identifier in selected for identifier in unique),
        "distinct_synthetic_identities": len(selected) == len(set(selected)),
    }


__all__ = [
    "DEFAULT_COUNTRY_CODE",
    "DEFAULT_TIMEZONE",
    "IDENTITY_ANSWER_LABELS",
    "IDENTITY_ANSWERS",
    "IDENTITY_QUESTION",
    "IdentityMapping",
    "apply_identity_answer",
    "identifier_variants",
    "inspect_csv_header",
    "is_phone_field",
    "iter_source_chunks",
    "load_identity_mapping",
    "normalize_answer",
    "normalize_generation_mode",
    "normalize_identifier",
    "normalize_column_name",
    "parse_timestamp_series",
    "plan_source_profiles",
    "resolve_source_schema",
    "schema_use_columns",
    "validate_cdr_ipdr",
]
