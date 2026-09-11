#!/usr/bin/env python3
"""Filter OTP/spam rows from a CDR CSV and write a cleaned CSV.

By default this removes rows where common party-number columns contain letters
and rows where any cell contains OTP/spam keywords.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = "call_logs_cdr_schema.csv"
DEFAULT_LETTER_COLUMNS = (
    "Mobile_No",
    "Other_Party_No",
    "SMSC_No",
    "LRN_B_Party_No",
    "Original_Originated_Party",
)
DEFAULT_KEYWORDS = ("otp", "spam")


def split_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _cell_text(value: object) -> str:
    """Return a safe text representation for regular and malformed CSV cells."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(part) for part in value if part is not None)
    return str(value)


def row_has_keyword(row: dict[str, str], keywords: tuple[str, ...]) -> bool:
    if not keywords:
        return False

    pattern = re.compile("|".join(re.escape(word) for word in keywords), re.IGNORECASE)
    return any(pattern.search(_cell_text(value)) for value in row.values())


def row_has_letters(row: dict[str, str], columns: tuple[str, ...]) -> bool:
    letter_pattern = re.compile(r"[A-Za-z]")
    return any(letter_pattern.search(_cell_text(row.get(column, ""))) for column in columns)


def _row_values_match(row: dict[str, str], pattern: re.Pattern[str]) -> bool:
    return any(pattern.search(_cell_text(value)) for value in row.values())


def _row_columns_match(
    row: dict[str, str], columns: tuple[str, ...], pattern: re.Pattern[str]
) -> bool:
    return any(pattern.search(_cell_text(row.get(column, ""))) for column in columns)


def filter_cdr_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    header_row: int = 0,
    letter_columns: tuple[str, ...] | str | None = DEFAULT_LETTER_COLUMNS,
    keywords: tuple[str, ...] = DEFAULT_KEYWORDS,
    strict_columns: bool = True,
) -> dict[str, object]:
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different to avoid overwriting source data.")

    total_rows = 0
    kept_rows = 0
    removed_reasons: Counter[str] = Counter()

    with input_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as source:
        for _ in range(header_row):
            next(source, None)

        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header row: {input_path}")

        if letter_columns == "all":
            resolved_letter_columns = tuple(reader.fieldnames)
        else:
            requested_columns = tuple(letter_columns or ())
            missing_columns = [
                column for column in requested_columns if column not in reader.fieldnames
            ]
            if missing_columns and strict_columns:
                raise ValueError(
                    "These letter columns were not found in the CSV: "
                    + ", ".join(missing_columns)
                )
            resolved_letter_columns = tuple(
                column for column in requested_columns if column in reader.fieldnames
            )

        keyword_pattern = (
            re.compile("|".join(re.escape(word) for word in keywords), re.IGNORECASE)
            if keywords
            else None
        )
        letter_pattern = re.compile(r"[A-Za-z]") if resolved_letter_columns else None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as target:
                temp_name = target.name
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()

                for row in reader:
                    total_rows += 1
                    if None in row:
                        source_row = header_row + total_rows + 1
                        raise ValueError(
                            f"Malformed CSV row {source_row}: it has more fields than the header."
                        )
                    has_letters = bool(
                        letter_pattern
                        and _row_columns_match(row, resolved_letter_columns, letter_pattern)
                    )
                    has_keyword = bool(
                        keyword_pattern and _row_values_match(row, keyword_pattern)
                    )

                    if has_letters or has_keyword:
                        if has_letters:
                            removed_reasons["letters_in_checked_columns"] += 1
                        if has_keyword:
                            removed_reasons["keyword_match"] += 1
                        continue

                    writer.writerow(row)
                    kept_rows += 1
            os.replace(temp_name, output_path)
        except Exception:
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)
            raise

    return {
        "input": input_path,
        "output": output_path,
        "rows_read": total_rows,
        "rows_kept": kept_rows,
        "rows_removed": total_rows - kept_rows,
        "removed_reasons": removed_reasons,
        "letter_columns": resolved_letter_columns,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a new CDR CSV without OTP/spam rows or sender IDs containing letters."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input CDR CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Cleaned output CSV path. Default: <input-name>_clean.csv beside the input.",
    )
    parser.add_argument(
        "--letter-columns",
        default=",".join(DEFAULT_LETTER_COLUMNS),
        help=(
            "Comma-separated columns checked for letters, or 'all' to check every column. "
            f"Default: {','.join(DEFAULT_LETTER_COLUMNS)}"
        ),
    )
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated text keywords to remove anywhere in a row. Use '' to disable.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    keywords = tuple(split_csv_arg(args.keywords))
    letter_columns = (
        "all" if args.letter_columns.strip().lower() == "all"
        else tuple(split_csv_arg(args.letter_columns))
    )
    result = filter_cdr_file(
        args.input,
        args.output,
        letter_columns=letter_columns,
        keywords=keywords,
    )

    print(f"Input: {result['input']}")
    print(f"Output: {result['output']}")
    print(f"Rows read: {result['rows_read']}")
    print(f"Rows kept: {result['rows_kept']}")
    print(f"Rows removed: {result['rows_removed']}")
    if result["removed_reasons"]:
        print("Removal reasons:")
        for reason, count in result["removed_reasons"].items():
            print(f"  {reason}: {count}")
    else:
        print("Removal reasons: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
