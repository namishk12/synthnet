from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from identity_validation import (  # noqa: E402
    DEFAULT_COUNTRY_CODE,
    identifier_variants,
    normalize_identifier as _shared_normalize_identifier,
)


def normalize_identifier(
    value: object,
    *,
    field_name: object | None = None,
    phone: bool | None = None,
) -> str:
    """Use SynthNet's shared string-preserving identity normalization."""

    return _shared_normalize_identifier(
        value,
        field_name=field_name,
        phone=phone,
        country_code=DEFAULT_COUNTRY_CODE,
    )


def _normalized_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


SUBSCRIBER_IDENTIFIER_ALIASES = (
    "subscriber_id",
    "msisdn",
    "Mobile_No",
    "Landline/MSISDN/MDN/Leased Circuit ID for Internet Access",
    "source_identifier",
    "imei",
    "imsi",
    "User Id for internet Access based on authentication",
)


def _find_columns(columns: Iterable[str], aliases: Iterable[str]) -> list[str]:
    normalized = {_normalized_column_name(column): column for column in columns}
    found: list[str] = []
    for alias in aliases:
        column = normalized.get(_normalized_column_name(alias))
        if column is not None and column not in found:
            found.append(column)
    return found


@dataclass
class IdentityResolver:
    alias_to_canonical: dict[str, str]
    source_path: str | None = None
    canonical_column: str | None = None
    ambiguous_alias_count: int = 0

    @classmethod
    def empty(cls) -> "IdentityResolver":
        return cls(alias_to_canonical={})

    @classmethod
    def from_subscribers_csv(cls, path: str | Path) -> "IdentityResolver":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Subscriber mapping CSV not found: {source}")

        frame = pd.read_csv(
            source,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            encoding_errors="replace",
        )
        identifier_columns = _find_columns(frame.columns, SUBSCRIBER_IDENTIFIER_ALIASES)
        if not identifier_columns:
            raise ValueError(
                "The subscriber mapping has no recognized identity column. Expected one of: "
                + ", ".join(SUBSCRIBER_IDENTIFIER_ALIASES)
            )
        subscriber_columns = _find_columns(frame.columns, ("subscriber_id",))
        msisdn_columns = _find_columns(frame.columns, ("msisdn", "Mobile_No"))
        canonical_column = (
            subscriber_columns[0]
            if subscriber_columns
            else msisdn_columns[0]
            if msisdn_columns
            else identifier_columns[0]
        )

        aliases: dict[str, str] = {}
        ambiguous: set[str] = set()
        for record in frame[identifier_columns].itertuples(index=False, name=None):
            values = {
                column: normalize_identifier(value, field_name=column)
                for column, value in zip(identifier_columns, record)
            }
            canonical = values.get(canonical_column, "")
            if not canonical:
                canonical = next((value for value in values.values() if value), "")
            if not canonical:
                continue
            for alias in values.values():
                if not alias:
                    continue
                previous = aliases.get(alias)
                if previous is not None and previous != canonical:
                    ambiguous.add(alias)
                else:
                    aliases[alias] = canonical

        for alias in ambiguous:
            aliases.pop(alias, None)
        return cls(
            alias_to_canonical=aliases,
            source_path=str(source),
            canonical_column=canonical_column,
            ambiguous_alias_count=len(ambiguous),
        )

    def canonicalize(self, value: object, *, field_name: object | None = None) -> str:
        variants = list(identifier_variants(value, field_name=field_name, country_code=DEFAULT_COUNTRY_CODE))
        if field_name is None:
            # Pair files contain an identifier without a source-column
            # semantic.  Treat an explicitly phone-shaped pair value as a
            # phone identifier for lookup, while source columns still obey
            # the field-only country-code rule in the shared normalizer.
            phone_variant = normalize_identifier(value, phone=True)
            if phone_variant and phone_variant not in variants:
                variants.append(phone_variant)
        for variant in variants:
            mapped = self.alias_to_canonical.get(variant)
            if mapped is not None:
                return mapped
        return variants[0] if variants else ""

    def canonicalize_series(
        self,
        series: pd.Series,
        *,
        field_name: object | None = None,
    ) -> pd.Series:
        return series.map(lambda value: self.canonicalize(value, field_name=field_name))


def locate_subscriber_mapping(
    explicit_path: str | None,
    cdr_path: str | Path,
    ipdr_path: str | Path | None = None,
    auto_detect: bool = True,
) -> Path | None:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    if not auto_detect:
        return None
    candidates = [Path(cdr_path).expanduser().resolve().parent / "subscribers.csv"]
    if ipdr_path:
        candidates.append(Path(ipdr_path).expanduser().resolve().parent / "subscribers.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None

