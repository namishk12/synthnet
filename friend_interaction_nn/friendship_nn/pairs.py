from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .identities import IdentityResolver
from .schemas import resolve_column


PAIR_COLUMN_OPTIONS = (
    ("person_a", "person_b"),
    ("msisdn_u", "msisdn_v"),
    ("subscriber_id_u", "subscriber_id_v"),
    ("source_identifier_u", "source_identifier_v"),
    ("person_1", "person_2"),
    ("user_a", "user_b"),
    ("user_1", "user_2"),
    ("caller", "receiver"),
    ("caller_id", "receiver_id"),
)
LABEL_ALIASES = (
    "label",
    "is_friend_or_interacting",
    "friend_or_interacting",
    "is_friend",
    "is_interacting",
    "target",
    "relationship",
    "base_truth",
    "ground_truth",
    "truth_label",
)
POSITIVE_LABELS = {
    "1",
    "1.0",
    "true",
    "yes",
    "y",
    "friend",
    "friends",
    "interacting",
    "interaction",
    "friend_or_interacting",
    "connected",
    "positive",
}
NEGATIVE_LABELS = {
    "0",
    "0.0",
    "false",
    "no",
    "n",
    "stranger",
    "strangers",
    "not_friend",
    "not friend",
    "not_interacting",
    "not interacting",
    "not_friend_or_interacting",
    "unconnected",
    "negative",
    "none",
}


def parse_label(value: object) -> int:
    text = str(value).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0
    raise ValueError(
        f"Unrecognized base-truth label {value!r}. Use 1/0, yes/no, "
        "friend/interacting, or stranger/not_interacting."
    )


def _resolve_pair_columns(
    columns: list[str], person_a_col: str | None, person_b_col: str | None
) -> tuple[str, str]:
    if bool(person_a_col) != bool(person_b_col):
        raise ValueError("Specify both pair identifier columns, not only one.")
    if person_a_col and person_b_col:
        return (
            str(resolve_column(columns, (person_a_col,), required=True)),
            str(resolve_column(columns, (person_b_col,), required=True)),
        )
    for left_alias, right_alias in PAIR_COLUMN_OPTIONS:
        left = resolve_column(columns, (left_alias,))
        right = resolve_column(columns, (right_alias,))
        if left is not None and right is not None:
            return left, right
    expected = [f"{left}/{right}" for left, right in PAIR_COLUMN_OPTIONS]
    raise ValueError(
        f"Could not identify the two person columns. Expected one of {expected}; "
        f"available columns are {columns}."
    )


def load_pair_csv(
    path: str | Path,
    require_label: bool,
    person_a_col: str | None = None,
    person_b_col: str | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Pair CSV not found: {source}")
    frame = pd.read_csv(
        source,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )
    columns = [str(column) for column in frame.columns]
    left, right = _resolve_pair_columns(columns, person_a_col, person_b_col)
    result = pd.DataFrame(
        {
            "person_a": frame[left].astype(str),
            "person_b": frame[right].astype(str),
            "source_row": frame.index + 2,
        }
    )
    if require_label:
        resolved_label = (
            resolve_column(columns, (label_col,), required=True)
            if label_col
            else resolve_column(columns, LABEL_ALIASES, required=True)
        )
        result["label"] = frame[str(resolved_label)].map(parse_label).astype(int)
    return result


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def prepare_pairs(frame: pd.DataFrame, resolver: IdentityResolver) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["person_a_id"] = prepared["person_a"].map(resolver.canonicalize)
    prepared["person_b_id"] = prepared["person_b"].map(resolver.canonicalize)

    missing = prepared[(prepared["person_a_id"] == "") | (prepared["person_b_id"] == "")]
    if not missing.empty:
        rows = missing["source_row"].astype(str).head(10).tolist()
        raise ValueError(f"Blank person identifier at CSV row(s): {', '.join(rows)}")
    self_pairs = prepared[prepared["person_a_id"] == prepared["person_b_id"]]
    if not self_pairs.empty:
        rows = self_pairs["source_row"].astype(str).head(10).tolist()
        raise ValueError(f"A person cannot be paired with themselves (CSV row(s): {', '.join(rows)}).")

    keys = [
        canonical_pair(left, right)
        for left, right in zip(prepared["person_a_id"], prepared["person_b_id"])
    ]
    prepared["pair_lo"] = [key[0] for key in keys]
    prepared["pair_hi"] = [key[1] for key in keys]

    if "label" in prepared.columns:
        conflicts = (
            prepared.groupby(["pair_lo", "pair_hi"])["label"].nunique().reset_index(name="labels")
        )
        conflicts = conflicts[conflicts["labels"] > 1]
        if not conflicts.empty:
            examples = [f"{row.pair_lo}/{row.pair_hi}" for row in conflicts.head(5).itertuples()]
            raise ValueError(
                "The same unordered pair has conflicting base-truth labels: " + ", ".join(examples)
            )
        prepared = prepared.drop_duplicates(["pair_lo", "pair_hi"], keep="first")
    return prepared.reset_index(drop=True)
