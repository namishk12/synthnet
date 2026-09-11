import os
import argparse
import csv
import hashlib
import json
from copy import deepcopy
from filter_cdr import DEFAULT_KEYWORDS, filter_cdr_file, split_csv_arg
from identity_validation import (
    IDENTITY_ANSWERS,
    IdentityMapping,
    apply_identity_answer,
    load_identity_mapping,
    normalize_generation_mode,
    normalize_identifier,
    parse_timestamp_series as shared_parse_timestamp_series,
    plan_source_profiles,
    validate_cdr_ipdr,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic telecom subscriber/CDR/IPDR data. Optionally "
            "condition the graph GAN and hourly activity patterns on real CSV files."
        )
    )
    parser.add_argument(
        "--cdr-input",
        default="call_logs_cdr_schema_clean.csv",
        help=(
            "Path to a CDR CSV. When supplied, the caller/receiver graph in this "
            "file is used as the per-agent behavior source. A one-caller CSV is "
            "expanded only when --generation-mode single_person_template is explicit. Default: "
            "call_logs_cdr_schema_clean.csv."
        ),
    )
    parser.add_argument(
        "--ipdr-input",
        default=None,
        help=(
            "Optional path to an IPDR CSV. Used to include extra subscriber IDs "
            "and learn the hourly internet-session pattern."
        ),
    )
    parser.add_argument("--cdr-caller-col", default="caller_id", help="CDR caller/subscriber column name.")
    parser.add_argument("--cdr-receiver-col", default="receiver_id", help="CDR receiver/subscriber column name.")
    parser.add_argument("--cdr-timestamp-col", default="timestamp", help="CDR timestamp column name.")
    parser.add_argument("--cdr-date-col", default=None, help="Optional CDR date column for split date/time schemas.")
    parser.add_argument("--cdr-time-col", default=None, help="Optional CDR time column for split date/time schemas.")
    parser.add_argument(
        "--cdr-clean-output",
        default=None,
        help=(
            "Optional path for the filtered CDR input. Defaults to "
            "<cdr-input-name>_clean.csv beside the input file."
        ),
    )
    parser.add_argument(
        "--cdr-filter-keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated keywords to remove from CDR input rows. Use '' to disable.",
    )
    parser.add_argument(
        "--skip-cdr-filter",
        action="store_true",
        help="Use --cdr-input directly without removing OTP/spam rows first.",
    )
    parser.add_argument("--ipdr-subscriber-col", default="subscriber_id", help="IPDR subscriber column name.")
    parser.add_argument("--ipdr-timestamp-col", default="timestamp", help="IPDR timestamp column name.")
    parser.add_argument(
        "--ipdr-destination-port-col",
        default="destination_port",
        help=(
            "Optional IPDR destination-port column. If absent, common service-port "
            "behavior is synthesized."
        ),
    )
    parser.add_argument("--ipdr-date-col", default=None, help="Optional IPDR date column for split date/time schemas.")
    parser.add_argument("--ipdr-time-col", default=None, help="Optional IPDR time column for split date/time schemas.")
    parser.add_argument(
        "--identity-answer",
        choices=IDENTITY_ANSWERS,
        default=None,
        help=(
            "How supplied CDR and IPDR subjects relate. Required when both files are supplied; "
            "identifier agreement is supporting evidence, not proof of identity."
        ),
    )
    parser.add_argument(
        "--generation-mode",
        choices=(
            "auto",
            "cdr_only",
            "combined",
            "multi_person_conditioned",
            "separate_sources",
            "single_person_template",
            "ipdr_only_correlation",
        ),
        default="auto",
        help=(
            "Processing mode. Single-person template expansion must be explicit; "
            "auto never silently copies one source profile."
        ),
    )
    parser.add_argument(
        "--identity-map",
        default=None,
        help="Optional explicit subscriber/alias mapping CSV used by every identity check.",
    )
    parser.add_argument(
        "--identity-map-json",
        default=None,
        help="Optional explicit alias-to-canonical JSON mapping produced by the upload API.",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Kolkata",
        help="Timezone assumed for timestamps without an offset (default: Asia/Kolkata).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk for validation and source inspection.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. Use --device cuda to require your NVIDIA GPU.",
    )
    parser.add_argument("--synthetic-users", type=int, default=1_000,
                        help="Target agent count, including when CDR input is supplied (default: 1,000).")
    parser.add_argument("--core-users", type=int, default=1000,
                        help="Core-user count; must equal --synthetic-users in core-only mode (default: 1,000).")
    parser.add_argument("--gan-steps", type=int, default=3000,
                        help="Conditional GAN training steps (default: 3000).")
    parser.add_argument("--gan-batch-size", type=int, default=4096,
                        help="Conditional GAN minibatch size (default: 4096).")
    parser.add_argument("--candidate-sample-size", type=int, default=2_000_000,
                        help="Candidate node pairs scored for inferred edges (default: 2,000,000).")
    parser.add_argument("--gan-inferred-top-fraction", type=float, default=0.001,
                        help="Fraction of scored candidate pairs retained (default: 0.001).")
    parser.add_argument(
        "--negative-truth-ratio", type=float, default=1.0,
        help=(
            "Number of sampled non-friend truth pairs per positive pair in "
            "friendship_truth_pairs.csv (default: 1.0)."
        ),
    )
    parser.add_argument("--simulation-days", type=int, default=7,
                        help="Number of days of CDR/IPDR activity to simulate (default: 7).")
    parser.add_argument("--interactions-per-user", type=int, default=1_000,
                        help="Exact number of generated CDR interactions per user (default: 1,000).")
    parser.add_argument("--sessions-per-user", type=int, default=1_000,
                        help="Exact number of generated IPDR sessions per user (default: 1,000).")
    parser.add_argument(
        "--output-mode",
        choices=["normal", "clean", "both"],
        default="both",
        help="Write normal outputs, clean outputs, or both (default: both).",
    )
    parser.add_argument(
        "--skip-schema-exports",
        action="store_true",
        help="Skip the large nodal-office schema exports; useful for the local GUI.",
    )
    args = parser.parse_args()
    if args.synthetic_users < 4:
        parser.error("--synthetic-users must be at least 4.")
    if args.core_users < 0:
        parser.error("--core-users cannot be negative.")
    if args.core_users != args.synthetic_users:
        parser.error("Core-only mode requires --core-users to equal --synthetic-users.")
    if args.gan_steps < 1 or args.gan_batch_size < 1 or args.candidate_sample_size < 1:
        parser.error("GAN steps, batch size, and candidate sample size must all be positive.")
    if not 0.0 <= args.gan_inferred_top_fraction <= 1.0:
        parser.error("--gan-inferred-top-fraction must be between 0 and 1.")
    if args.negative_truth_ratio < 0.0:
        parser.error("--negative-truth-ratio cannot be negative.")
    if args.simulation_days < 1:
        parser.error("--simulation-days must be positive.")
    if args.interactions_per_user < 1:
        parser.error("--interactions-per-user must be positive.")
    if args.sessions_per_user < 1:
        parser.error("--sessions-per-user must be positive.")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive.")
    return args


ARGS = parse_args()

import torch
import torch.nn as nn
import networkx as nx
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from sdv.metadata import MultiTableMetadata


def write_csv_safely(df, path, **kwargs):
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        stem, ext = os.path.splitext(path)
        fallback_path = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        print(f"  Could not overwrite {path}; it may be open. Writing {fallback_path} instead.")
        df.to_csv(fallback_path, **kwargs)
        return fallback_path


def select_device(device_arg):
    cuda_available = torch.cuda.is_available()
    if device_arg == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested with --device cuda, but this Python environment "
            "cannot see a CUDA-enabled PyTorch install. Install a CUDA build of "
            "PyTorch for your RTX 4060, then rerun the script."
        )
    if device_arg == "cpu":
        return torch.device("cpu")
    if cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        return torch.device("cuda")
    return torch.device("cpu")


# Set deterministic seeds for identical behavior across runs
SEED = 101
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
fake = Faker('en_IN')
fake.seed_instance(SEED)

# Common application/service destination ports used only when an IPDR source
# does not provide a usable destination-port column. If the source does have
# ports, its empirical distribution takes precedence.
COMMON_DESTINATION_PORTS = np.array(
    [443, 80, 53, 8080, 8443, 123, 22, 3478, 5222, 5223, 5228, 5242, 993, 995, 25, 5060],
    dtype=np.int64,
)
COMMON_DESTINATION_PORT_WEIGHTS = np.array(
    [0.522, 0.12, 0.09, 0.035, 0.025, 0.025, 0.02, 0.02, 0.04, 0.02, 0.02, 0.02, 0.015, 0.008, 0.005, 0.015],
    dtype=float,
)
COMMON_DESTINATION_PORT_WEIGHTS /= COMMON_DESTINATION_PORT_WEIGHTS.sum()

# ======================================================================
# REAL-DATA INPUT CONFIG
# ======================================================================
# Point these at your own files to ground the synthetic pipeline in real
# calling/session behavior instead of a fully invented graph. Leave a
# path as None to fall back to the original synthetic-only behavior for
# that piece.
#
# CDR_INPUT_PATH: a real call-detail-record file. Used to (a) discover
#   the real set of subscribers, (b) build the real who-calls-whom graph
#   that the GAN is conditioned on, and (c) learn the real hour-of-day
#   calling-rate pattern.
# IPDR_INPUT_PATH: a real internet-session file. Used to learn the real
#   hour-of-day data-session-rate pattern. Optional even if CDR is given.
#
# Expected minimal columns (rename via *_COLUMNS dicts below to match
# your file, no need to rename the actual CSV):
#   CDR file:  one column identifying the caller, one identifying the
#              receiver (any consistent subscriber identifier, e.g.
#              subscriber_id or msisdn), one timestamp column.
#   IPDR file: one column identifying the subscriber, one timestamp
#              column, and optionally a destination-port column.
#
# You can pass these from the command line instead of editing the file:
#   python Synthetic_data_cgan.py --cdr-input real_cdr.csv --ipdr-input real_ipdr.csv
CDR_INPUT_PATH = ARGS.cdr_input
IPDR_INPUT_PATH = ARGS.ipdr_input

CDR_COLUMNS = {
    'caller': ARGS.cdr_caller_col,
    'receiver': ARGS.cdr_receiver_col,
    'timestamp': ARGS.cdr_timestamp_col,
}
CDR_TIMESTAMP_PARTS = {
    'date': ARGS.cdr_date_col,
    'time': ARGS.cdr_time_col,
}
IPDR_COLUMNS = {
    'subscriber': ARGS.ipdr_subscriber_col,
    'timestamp': ARGS.ipdr_timestamp_col,
    'destination_port': ARGS.ipdr_destination_port_col,
}
IPDR_TIMESTAMP_PARTS = {
    'date': ARGS.ipdr_date_col,
    'time': ARGS.ipdr_time_col,
}

FRIEND_EDGE_NUMBER_COLUMNS = (
    'msisdn_u',
    'msisdn_v',
    'source_identifier_u',
    'source_identifier_v',
)
FINAL_CDR_NUMBER_COLUMNS = (
    'Mobile_No',
    'Other_Party_No',
    'Original_Originated_Party',
    'SMSC_No',
    'LRN_B_Party_No',
)
FINAL_IPDR_NUMBER_COLUMNS = (
    'Contact No.',
    'Alternate Contact No.',
    'Landline/MSISDN/MDN/Leased Circuit ID for Internet Access',
    'Source MAC-ID Address/Other device Identification number',
    'IMSI',
)

CSV_HEADER_ROWS = {
    'CDR': 0,
    'IPDR': 0,
}

USING_REAL_GRAPH = CDR_INPUT_PATH is not None
TEMPLATE_AGENT_MODE = False
IDENTITY_MAPPING = IdentityMapping.empty()
IDENTITY_VALIDATION = None
IDENTITY_DECISION = None
EFFECTIVE_GENERATION_MODE = None
USE_IPDR_FOR_GENERATION = False
SOURCE_PROFILE_FOR_NODE = {}
SOURCE_PROFILE_IDS = []
SOURCE_PROFILE_PLAN = None
# In one-person template mode, each synthetic core user is the center of an
# ego/star network. A small, reproducible random subset of center pairs is
# marked as a known friendship; those pairs receive a higher call probability.
TEMPLATE_KNOWN_FRIEND_PAIR_FRACTION = 0.02
TEMPLATE_FRIEND_CALL_SHARE = 0.22
TEMPLATE_DUMMY_MAX_INTERACTIONS = 2

DEVICE = select_device(ARGS.device)
if DEVICE.type == "cuda":
    print(f"Using GPU for GAN training: {torch.cuda.get_device_name(DEVICE)}")
else:
    print("Using CPU for GAN training. Pass --device cuda to require GPU training.")


# ======================================================================
# REAL-DATA LOADERS
# ======================================================================
def find_csv_header_row(path, label):
    """
    Finds the real header row even when an export file starts with report
    metadata lines before the CSV table. Returns (zero_based_row, columns).
    """
    if path is None:
        return 0, []
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} CSV not found: {path}")

    marker_sets = {
        'CDR': [
            {'caller_id', 'receiver_id'},
            {'caller', 'receiver'},
            {'caller_msisdn', 'receiver_msisdn'},
            {'Mobile_No', 'Other_Party_No', 'Call_Date', 'Call_Initiation_Time(CIT)'},
        ],
        'IPDR': [
            {'subscriber_id', 'timestamp'},
            {
                'Landline/MSISDN/MDN/Leased Circuit ID for Internet Access',
                'TIME1 (dd/MM/yyyy HH:mm:ss)',
            },
            {
                'Landline/MSISDN/MDN/Leased Circuit ID for Internet Access',
                'Start Date of Public IP Address allocation (dd/mm/yyyy)',
                'IST Start Time of Public IP address allocation (hh:mm:ss)',
            },
        ],
    }

    with open(path, newline='', encoding='utf-8-sig', errors='replace') as handle:
        reader = csv.reader(handle)
        for row_idx, row in enumerate(reader):
            columns = [col.strip() for col in row]
            column_set = set(columns)
            if any(markers.issubset(column_set) for markers in marker_sets[label]):
                return row_idx, columns
            if row_idx >= 100:
                break

    columns = list(pd.read_csv(path, nrows=0).columns)
    return 0, columns


def read_input_csv(path, label):
    return pd.read_csv(
        path,
        skiprows=CSV_HEADER_ROWS[label],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding='utf-8-sig',
        encoding_errors='replace',
    )


def has_default_arg(current_value, default_value):
    return current_value == default_value


def resolve_input_column(columns, requested_column, fallback_columns, logical_name, label, used_default):
    if requested_column in columns:
        return requested_column
    if used_default:
        for fallback in fallback_columns:
            if fallback in columns:
                return fallback
    raise ValueError(
        f"{label} CSV is missing the {logical_name} column. Looked for "
        f"{[requested_column] + fallback_columns}. Available columns: {columns}"
    )


def resolve_timestamp_columns(columns, timestamp_col, timestamp_parts, fallback_timestamp_cols,
                              fallback_date_time_pairs, label, used_default):
    if timestamp_parts['date'] is not None or timestamp_parts['time'] is not None:
        if timestamp_parts['date'] is None:
            raise ValueError(f"{label} split timestamp input needs a date column.")
        requested_columns = [timestamp_parts['date']]
        if timestamp_parts['time'] is not None:
            requested_columns.append(timestamp_parts['time'])
        missing = [col for col in requested_columns if col not in columns]
        if missing:
            raise ValueError(
                f"{label} CSV is missing split timestamp column(s): {missing}. "
                f"Available columns: {columns}"
            )
        return None, timestamp_parts

    if timestamp_col in columns:
        return timestamp_col, {'date': None, 'time': None}

    if used_default:
        for fallback in fallback_timestamp_cols:
            if fallback in columns:
                return fallback, {'date': None, 'time': None}
        for date_col, time_col in fallback_date_time_pairs:
            if date_col in columns and time_col in columns:
                return None, {'date': date_col, 'time': time_col}

        for date_col, _ in fallback_date_time_pairs:
            if date_col in columns:
                return None, {'date': date_col, 'time': None}

    print(
        f"WARNING: {label} CSV has no usable timestamp/date column; temporal evidence will be recorded as unavailable.",
        flush=True,
    )
    return None, {'date': None, 'time': None}


def clean_csv_text_series(series, field_name=None):
    """Normalize source text with the shared identity rules when appropriate."""
    return series.map(lambda value: normalize_identifier(value, field_name=field_name))


def normalize_identifier_series(df, column):
    values = df[column].map(
        lambda value: IDENTITY_MAPPING.canonicalize(value, field_name=column)
    )
    return values[values != '']


def canonical_identifier_series(series, field_name):
    return series.map(
        lambda value: IDENTITY_MAPPING.canonicalize(value, field_name=field_name)
    )


def validate_real_data_inputs():
    global CDR_INPUT_PATH, IPDR_INPUT_PATH, IDENTITY_MAPPING
    global IDENTITY_VALIDATION, IDENTITY_DECISION
    global EFFECTIVE_GENERATION_MODE, USE_IPDR_FOR_GENERATION

    inline_mapping = None
    if ARGS.identity_map_json:
        with open(ARGS.identity_map_json, 'r', encoding='utf-8') as mapping_handle:
            inline_mapping = json.load(mapping_handle)
    IDENTITY_MAPPING = load_identity_mapping(
        ARGS.identity_map,
        inline_mapping,
        country_code='91',
    )
    if IDENTITY_MAPPING.source_path:
        print(
            f"Resolved explicit identity mapping: {IDENTITY_MAPPING.source_path} "
            f"({len(IDENTITY_MAPPING.alias_to_canonical):,} aliases)."
        )
    if IDENTITY_MAPPING.ambiguous_alias_count:
        print(
            f"WARNING: quarantined {IDENTITY_MAPPING.ambiguous_alias_count:,} ambiguous identity aliases; "
            "they cannot be used to combine sources."
        )

    if IPDR_INPUT_PATH is not None and CDR_INPUT_PATH is None:
        raise ValueError(
            "--ipdr-input can only be used together with --cdr-input because "
            "the CDR caller/receiver graph is the graph GAN training backbone."
        )

    if CDR_INPUT_PATH is not None:
        header_row, columns = find_csv_header_row(CDR_INPUT_PATH, 'CDR')
        CSV_HEADER_ROWS['CDR'] = header_row

        CDR_COLUMNS['caller'] = resolve_input_column(
            columns,
            CDR_COLUMNS['caller'],
            ['Mobile_No', 'caller', 'caller_msisdn', 'calling_party', 'a_party', 'A_Party_No'],
            'caller/subscriber',
            'CDR',
            has_default_arg(ARGS.cdr_caller_col, 'caller_id'),
        )
        CDR_COLUMNS['receiver'] = resolve_input_column(
            columns,
            CDR_COLUMNS['receiver'],
            ['Other_Party_No', 'receiver', 'receiver_msisdn', 'called_party', 'b_party', 'B_Party_No'],
            'receiver/other-party',
            'CDR',
            has_default_arg(ARGS.cdr_receiver_col, 'receiver_id'),
        )
        CDR_COLUMNS['timestamp'], resolved_parts = resolve_timestamp_columns(
            columns,
            CDR_COLUMNS['timestamp'],
            CDR_TIMESTAMP_PARTS,
            [],
            [('Call_Date', 'Call_Initiation_Time(CIT)')],
            'CDR',
            has_default_arg(ARGS.cdr_timestamp_col, 'timestamp'),
        )
        CDR_TIMESTAMP_PARTS.update(resolved_parts)
        print(
            f"Resolved CDR input schema: header row {header_row + 1}, "
            f"caller='{CDR_COLUMNS['caller']}', receiver='{CDR_COLUMNS['receiver']}'."
        )

        input_stem = os.path.splitext(os.path.basename(CDR_INPUT_PATH))[0].lower()
        input_is_already_clean = input_stem.endswith('_clean')
        if not ARGS.skip_cdr_filter and not input_is_already_clean:
            cdr_filter_keywords = tuple(split_csv_arg(ARGS.cdr_filter_keywords))
            cdr_filter_letter_columns = tuple(dict.fromkeys([
                CDR_COLUMNS['receiver'],
                'Other_Party_No',
                'Original_Originated_Party',
                'SMSC_No',
                'LRN_B_Party_No',
            ]))
            filter_result = filter_cdr_file(
                CDR_INPUT_PATH,
                ARGS.cdr_clean_output,
                header_row=header_row,
                letter_columns=cdr_filter_letter_columns,
                keywords=cdr_filter_keywords,
                strict_columns=False,
            )
            CDR_INPUT_PATH = str(filter_result['output'])
            CSV_HEADER_ROWS['CDR'] = 0
            print(
                "Filtered CDR input for OTP/spam before training: "
                f"{filter_result['rows_removed']} removed, "
                f"{filter_result['rows_kept']} kept -> {filter_result['output']}"
            )
        elif ARGS.skip_cdr_filter:
            print("Skipping CDR OTP/spam input filter.")
        else:
            print("CDR input filename ends in '_clean'; using it without filtering again.")

    if IPDR_INPUT_PATH is not None:
        header_row, columns = find_csv_header_row(IPDR_INPUT_PATH, 'IPDR')
        CSV_HEADER_ROWS['IPDR'] = header_row

        IPDR_COLUMNS['subscriber'] = resolve_input_column(
            columns,
            IPDR_COLUMNS['subscriber'],
            [
                'Landline/MSISDN/MDN/Leased Circuit ID for Internet Access',
                'msisdn',
                'Mobile_No',
                'user_id',
                'User Id for internet Access based on authentication',
            ],
            'subscriber',
            'IPDR',
            has_default_arg(ARGS.ipdr_subscriber_col, 'subscriber_id'),
        )
        IPDR_COLUMNS['timestamp'], resolved_parts = resolve_timestamp_columns(
            columns,
            IPDR_COLUMNS['timestamp'],
            IPDR_TIMESTAMP_PARTS,
            ['TIME1 (dd/MM/yyyy HH:mm:ss)'],
            [(
                'Start Date of Public IP Address allocation (dd/mm/yyyy)',
                'IST Start Time of Public IP address allocation (hh:mm:ss)',
            )],
            'IPDR',
            has_default_arg(ARGS.ipdr_timestamp_col, 'timestamp'),
        )
        IPDR_TIMESTAMP_PARTS.update(resolved_parts)
        requested_port_column = IPDR_COLUMNS['destination_port']
        if requested_port_column in columns:
            IPDR_COLUMNS['destination_port'] = requested_port_column
        elif has_default_arg(ARGS.ipdr_destination_port_col, 'destination_port'):
            IPDR_COLUMNS['destination_port'] = next(
                (
                    fallback for fallback in (
                        'Destination Port',
                        'Destination_Port',
                        'DestinationPort',
                        'destinationPort',
                    ) if fallback in columns
                ),
                None,
            )
        else:
            raise ValueError(
                f"IPDR CSV is missing the requested destination-port column "
                f"'{requested_port_column}'. Available columns: {columns}"
            )
        print(
            f"Resolved IPDR input schema: header row {header_row + 1}, "
            f"subscriber='{IPDR_COLUMNS['subscriber']}', "
            f"destination_port='{IPDR_COLUMNS['destination_port'] or 'synthetic common-service distribution'}'."
        )
        print("IPDR input is used for session modeling but is not filtered; only the CDR CSV is filtered.")

    # Re-scan the final source files independently of the modeling preview.
    # This is deliberately after the optional CDR filter so the exact files
    # consumed by generation are the files represented in run metadata.
    IDENTITY_VALIDATION = validate_cdr_ipdr(
        CDR_INPUT_PATH,
        IPDR_INPUT_PATH,
        mapping=IDENTITY_MAPPING,
        overrides={
            'cdr': {
                'subject': CDR_COLUMNS.get('caller'),
                'contact': CDR_COLUMNS.get('receiver'),
                'timestamp': CDR_COLUMNS.get('timestamp'),
                'date': CDR_TIMESTAMP_PARTS.get('date'),
                'time': CDR_TIMESTAMP_PARTS.get('time'),
            },
            'ipdr': {
                'subscriber': IPDR_COLUMNS.get('subscriber'),
                'timestamp': IPDR_COLUMNS.get('timestamp'),
                'date': IPDR_TIMESTAMP_PARTS.get('date'),
                'time': IPDR_TIMESTAMP_PARTS.get('time'),
            },
        },
        chunk_size=ARGS.chunk_size,
        default_timezone=ARGS.timezone,
    )
    IDENTITY_DECISION = apply_identity_answer(
        IDENTITY_VALIDATION,
        ARGS.identity_answer,
        generation_mode=None if ARGS.generation_mode == 'auto' else ARGS.generation_mode,
    )
    EFFECTIVE_GENERATION_MODE = IDENTITY_DECISION.get('generation_mode') or 'cdr_only'
    USE_IPDR_FOR_GENERATION = EFFECTIVE_GENERATION_MODE in {
        'combined',
        'multi_person_conditioned',
        'single_person_template',
    }
    matching = IDENTITY_VALIDATION['matching']
    print(
        "Identity evidence: "
        f"{matching['cdr_subject_count']:,} CDR subject(s), "
        f"{matching['ipdr_subscriber_count']:,} IPDR subscriber(s), "
        f"{matching['matched_count']:,} matched, "
        f"{matching['unmatched_cdr_count']:,} CDR-unmatched, "
        f"{matching['unmatched_ipdr_count']:,} IPDR-unmatched. "
        "Agreement supports matching; it does not prove personal identity."
    )
    temporal = IDENTITY_VALIDATION['temporal']
    print(f"Observation periods: {temporal['explanation']}")
    if not IDENTITY_DECISION['allowed']:
        raise ValueError("Identity processing check blocked generation: " + " ".join(IDENTITY_DECISION['blocking_reasons']))


def build_subscriber_id_map(cdr_path, cdr_cols, target_users):
    """
    Select source caller profiles without silently dropping identities.

    When the requested population is larger than the source population, every
    source caller is retained and additional synthetic IDs use a deterministic
    round-robin source-profile plan. A one-caller expansion is only permitted
    when the caller explicitly selected ``single_person_template``.
    """
    cdr_df = read_input_csv(cdr_path, 'CDR')
    callers = normalize_identifier_series(cdr_df, cdr_cols['caller'])
    caller_counts = callers.value_counts().rename_axis('subscriber').reset_index(name='calls')
    caller_counts['subscriber'] = caller_counts['subscriber'].astype(str)
    caller_counts = caller_counts.sort_values(
        ['calls', 'subscriber'], ascending=[False, True], kind='stable'
    )
    global TEMPLATE_AGENT_MODE, SOURCE_PROFILE_FOR_NODE, SOURCE_PROFILE_IDS, SOURCE_PROFILE_PLAN
    source_ids = caller_counts['subscriber'].tolist()
    source_counts = dict(zip(caller_counts['subscriber'], caller_counts['calls']))
    requested_mode = None if ARGS.generation_mode == 'auto' else ARGS.generation_mode
    SOURCE_PROFILE_PLAN = plan_source_profiles(
        source_ids,
        target_users,
        source_counts=source_counts,
        generation_mode=requested_mode,
        seed=SEED,
    )
    selected_ids = list(SOURCE_PROFILE_PLAN['agent_ids'])
    SOURCE_PROFILE_IDS = list(source_ids)
    SOURCE_PROFILE_FOR_NODE = {
        index: selected_ids.index(source_id)
        for index, agent_id in enumerate(selected_ids)
        for source_id in [SOURCE_PROFILE_PLAN['source_profile_for_agent'][agent_id]]
    }
    TEMPLATE_AGENT_MODE = SOURCE_PROFILE_PLAN['strategy'].startswith('explicit_single_person_template')
    if len(selected_ids) > len(source_ids):
        print(
            f"  Preserved all {len(source_ids)} CDR source profile(s) and expanded to "
            f"{target_users} agents with {SOURCE_PROFILE_PLAN['strategy']} (seed={SEED})."
        )
    else:
        print(f"  Selected {len(selected_ids)} CDR source profile(s) without contact-only promotion.")
    return {sid: idx for idx, sid in enumerate(selected_ids)}, cdr_df


def learn_agent_call_patterns(cdr_df, cdr_cols, id_to_node):
    """Learn each selected agent's hours, durations, and modeled contacts."""
    num_agents = len(id_to_node)
    caller_ids = canonical_identifier_series(cdr_df[cdr_cols['caller']], cdr_cols['caller'])
    receiver_ids = canonical_identifier_series(cdr_df[cdr_cols['receiver']], cdr_cols['receiver'])
    caller_nodes = caller_ids.map(id_to_node)
    receiver_nodes = receiver_ids.map(id_to_node)
    timestamps = parse_timestamp_series(
        cdr_df, cdr_cols['timestamp'], CDR_TIMESTAMP_PARTS
    )

    duration_column = next(
        (column for column in ('duration_seconds', 'Call_Duration') if column in cdr_df.columns),
        None,
    )
    if duration_column is None:
        duration_values = pd.Series(60.0, index=cdr_df.index)
    else:
        duration_values = pd.to_numeric(cdr_df[duration_column], errors='coerce')

    patterns = []
    global_hours = timestamps.dt.hour.to_numpy(dtype=np.int64)
    global_hours = global_hours if len(global_hours) else np.arange(24, dtype=np.int64)
    valid_global_durations = duration_values[(duration_values >= 0) & duration_values.notna()]
    global_durations = valid_global_durations.to_numpy(dtype=np.int64)
    if not len(global_durations):
        global_durations = np.array([60], dtype=np.int64)

    for node in range(num_agents):
        agent_rows = caller_nodes[caller_nodes == node].index
        agent_timestamps = timestamps.reindex(agent_rows).dropna()
        hours = agent_timestamps.dt.hour.to_numpy(dtype=np.int64)
        if not len(hours):
            hours = global_hours

        durations = duration_values.reindex(agent_rows)
        durations = durations[(durations >= 0) & durations.notna()].to_numpy(dtype=np.int64)
        if not len(durations):
            durations = global_durations

        contacts = receiver_nodes.reindex(agent_rows).dropna().to_numpy(dtype=np.int64)
        contacts = contacts[contacts != node]
        crosslink_share = len(contacts) / max(len(agent_rows), 1)
        patterns.append({
            'hours': hours,
            'durations': durations,
            'contacts': contacts,
            'crosslink_share': crosslink_share,
        })
    # Expanded synthetic agents inherit only the selected source profile's
    # behavior. Their generated subscriber IDs remain distinct and their
    # lineage is recorded separately; no unmatched agent receives another
    # person's IPDR profile by accident.
    for node, source_node in SOURCE_PROFILE_FOR_NODE.items():
        if node != source_node and 0 <= source_node < len(patterns):
            patterns[node] = deepcopy(patterns[source_node])
    return patterns


def learn_agent_session_patterns(ipdr_df, ipdr_cols, id_to_node):
    """Learn per-agent IPDR hours, destinations, volumes, and destination ports."""
    num_agents = len(id_to_node)
    subscriber_ids = canonical_identifier_series(ipdr_df[ipdr_cols['subscriber']], ipdr_cols['subscriber'])
    subscriber_nodes = subscriber_ids.map(id_to_node)
    timestamps = parse_timestamp_series(
        ipdr_df, ipdr_cols['timestamp'], IPDR_TIMESTAMP_PARTS
    )

    destination_column = next(
        (column for column in ('destination_ip', 'Destination IP Address') if column in ipdr_df.columns),
        None,
    )
    if destination_column is None:
        destination_values = pd.Series('', index=ipdr_df.index, dtype=object)
    else:
        destination_values = clean_csv_text_series(ipdr_df[destination_column])

    if 'megabytes_transferred' in ipdr_df.columns:
        volume_values = pd.to_numeric(ipdr_df['megabytes_transferred'], errors='coerce')
    elif 'Data Volume Up Link' in ipdr_df.columns and 'Data Volume Down Link' in ipdr_df.columns:
        up_kb = pd.to_numeric(ipdr_df['Data Volume Up Link'], errors='coerce')
        down_kb = pd.to_numeric(ipdr_df['Data Volume Down Link'], errors='coerce')
        volume_values = (up_kb + down_kb) / 1024.0
    else:
        volume_values = pd.Series(np.nan, index=ipdr_df.index, dtype=float)

    destination_port_column = ipdr_cols.get('destination_port')
    if destination_port_column and destination_port_column in ipdr_df.columns:
        destination_port_values = pd.to_numeric(
            ipdr_df[destination_port_column], errors='coerce'
        )
        destination_port_values = destination_port_values.where(
            destination_port_values.between(1, 65535)
        )
    else:
        destination_port_values = pd.Series(np.nan, index=ipdr_df.index, dtype=float)

    # These values are intentionally neutral fallbacks.  A source subject
    # with no matching IPDR rows must not inherit a different person's
    # destination/hour/volume profile, even when the uploaded IPDR contains
    # rich observations for other subjects.
    neutral_hours = np.arange(24, dtype=np.int64)
    neutral_destinations = np.array(
        ["142.250.190.46", "157.240.22.35", "104.244.42.1", "31.13.71.36"],
        dtype=object,
    )
    neutral_volumes = np.array([5.0], dtype=float)

    valid_global_ports = destination_port_values.dropna().astype(np.int64)
    if len(valid_global_ports):
        global_port_counts = valid_global_ports.value_counts().sort_index()
        observed_destination_ports = global_port_counts.index.to_numpy(dtype=np.int64)
        observed_destination_port_weights = (
            global_port_counts.to_numpy(dtype=float) / global_port_counts.sum()
        )
        print(
            f"  Learned {len(observed_destination_ports)} IPDR destination-port values "
            f"from the uploaded data."
        )
    else:
        observed_destination_ports = COMMON_DESTINATION_PORTS
        observed_destination_port_weights = COMMON_DESTINATION_PORT_WEIGHTS
        print(
            "  IPDR has no usable destination-port column; using common web, DNS, "
            "and messaging-service ports (including WhatsApp-style 5222/5223/5228/5242)."
        )

    patterns = []
    for node in range(num_agents):
        agent_rows = subscriber_nodes[subscriber_nodes == node].index
        agent_timestamps = timestamps.reindex(agent_rows).dropna()
        hours = agent_timestamps.dt.hour.to_numpy(dtype=np.int64)
        if not len(hours):
            hours = neutral_hours

        destinations = destination_values.reindex(agent_rows)
        destinations = destinations[destinations != ''].to_numpy(dtype=object)
        if not len(destinations):
            destinations = neutral_destinations

        volumes = volume_values.reindex(agent_rows)
        volumes = volumes[(volumes > 0) & volumes.notna()].to_numpy(dtype=float)
        if not len(volumes):
            volumes = neutral_volumes

        destination_ports = destination_port_values.reindex(agent_rows).dropna().astype(np.int64)
        if len(destination_ports):
            port_counts = destination_ports.value_counts().sort_index()
            destination_ports = port_counts.index.to_numpy(dtype=np.int64)
            destination_port_weights = (
                port_counts.to_numpy(dtype=float) / port_counts.sum()
            )
        else:
            destination_ports = COMMON_DESTINATION_PORTS
            destination_port_weights = COMMON_DESTINATION_PORT_WEIGHTS
        patterns.append({
            'hours': hours,
            'destinations': destinations,
            'volumes': volumes,
            'destination_ports': destination_ports,
            'destination_port_weights': destination_port_weights,
        })
    for node, source_node in SOURCE_PROFILE_FOR_NODE.items():
        if node != source_node and 0 <= source_node < len(patterns):
            patterns[node] = deepcopy(patterns[source_node])
    return patterns


def build_real_edges(cdr_df, cdr_cols, id_to_node):
    """
    Builds the real subscriber-to-subscriber graph as a sparse (E, 2)
    edge list directly from caller/receiver pairs in the real CDR file
    (an edge exists if two subscribers called each other at least once).

    Deliberately sparse/edge-list based rather than a dense N x N matrix:
    a dense matrix is fine at a few thousand real subscribers but becomes
    impossible at real-world scale (e.g. 500,000 real subscribers would
    need a quarter-trillion-cell matrix). This scales the same way the
    rest of this pipeline's GAN does -- via edge lists, never a full
    pairwise matrix -- so it works regardless of how large your real
    input file is.
    """
    caller_ids = canonical_identifier_series(cdr_df[cdr_cols['caller']], cdr_cols['caller'])
    receiver_ids = canonical_identifier_series(cdr_df[cdr_cols['receiver']], cdr_cols['receiver'])
    a_idx = caller_ids.map(id_to_node)
    b_idx = receiver_ids.map(id_to_node)
    valid = a_idx.notna() & b_idx.notna()
    a_idx = a_idx[valid].to_numpy(dtype=np.int64)
    b_idx = b_idx[valid].to_numpy(dtype=np.int64)

    not_self = a_idx != b_idx
    a_idx, b_idx = a_idx[not_self], b_idx[not_self]
    if len(a_idx) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # Canonicalize each pair as (min, max) then drop duplicates so a
    # caller<->receiver pair that appears in both directions (or many
    # times) becomes a single edge.
    lo = np.minimum(a_idx, b_idx)
    hi = np.maximum(a_idx, b_idx)
    pairs = np.stack([lo, hi], axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    return unique_pairs


def build_template_known_friend_edges(num_nodes, pair_fraction, seed):
    """Create sparse known friendships between synthetic star centers.

    The uploaded one-person CDR establishes the shape of each center's ego
    network, but it cannot identify which *other* synthetic centers know one
    another. We therefore reveal a small random fraction of center pairs as
    known friendships and use those as the base truth for the graph GAN and
    interaction generator.
    """
    if num_nodes < 2 or pair_fraction <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    total_pairs = num_nodes * (num_nodes - 1) // 2
    pair_count = min(total_pairs, max(1, int(round(total_pairs * pair_fraction))))
    upper_u, upper_v = np.triu_indices(num_nodes, k=1)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total_pairs, size=pair_count, replace=False)
    return np.stack([upper_u[chosen], upper_v[chosen]], axis=1).astype(np.int64)


def apply_template_friend_contacts(patterns, friend_edges, call_share):
    """Attach known center-to-center friends to every template-mode agent."""
    neighbors = [[] for _ in range(len(patterns))]
    for u, v in np.asarray(friend_edges, dtype=np.int64):
        neighbors[int(u)].append(int(v))
        neighbors[int(v)].append(int(u))
    for node, pattern in enumerate(patterns):
        pattern['contacts'] = np.asarray(neighbors[node], dtype=np.int64)
        pattern['crosslink_share'] = call_share if neighbors[node] else 0.0
    return patterns


def parse_timestamp_series(df, timestamp_col=None, timestamp_parts=None):
    if timestamp_parts is None:
        timestamp_parts = {'date': None, 'time': None}
    schema = {
        'timestamp': timestamp_col,
        'date': timestamp_parts.get('date'),
        'time': timestamp_parts.get('time'),
    }
    return shared_parse_timestamp_series(df, schema, ARGS.timezone).dropna()


def learn_hourly_rate_pattern(df, timestamp_col, num_edges, timestamp_parts=None):
    """
    Learns an empirical hour-of-day activity pattern from real
    timestamps: a base per-edge-per-hour event probability plus a 24-slot
    multiplier curve capturing the real diurnal shape (e.g. daytime call
    spikes, overnight lull). Returns (base_rate, hourly_multiplier) or
    (None, None) if the timestamp column can't be parsed / is empty.
    """
    try:
        ts = parse_timestamp_series(df, timestamp_col, timestamp_parts)
    except (KeyError, ValueError, TypeError):
        return None, None
    if ts.empty or num_edges == 0:
        return None, None

    span_days = max((ts.max() - ts.min()).total_seconds() / 86400.0, 1.0)
    counts_by_hour = ts.dt.hour.value_counts().reindex(range(24), fill_value=0).astype(float)
    total_events = counts_by_hour.sum()
    if total_events == 0:
        return None, None

    avg_per_hour = counts_by_hour.mean()
    hourly_multiplier = (counts_by_hour / avg_per_hour).to_dict()
    base_rate = total_events / (num_edges * span_days * 24.0)
    return base_rate, hourly_multiplier


REAL_ID_TO_NODE = None
REAL_EDGES = None
REAL_AGENT_CALL_PATTERNS = None
REAL_AGENT_SESSION_PATTERNS = None
REAL_CALL_BASE_RATE, REAL_CALL_HOURLY_MULT = None, None
REAL_SESSION_BASE_RATE, REAL_SESSION_HOURLY_MULT = None, None
TEMPLATE_KNOWN_FRIEND_EDGES = None

validate_real_data_inputs()

if USING_REAL_GRAPH:
    print(f"Loading real subscriber graph from {CDR_INPUT_PATH} ...")
    REAL_ID_TO_NODE, _cdr_df_for_ids = build_subscriber_id_map(
        CDR_INPUT_PATH, CDR_COLUMNS, ARGS.synthetic_users
    )
    REAL_EDGES = build_real_edges(_cdr_df_for_ids, CDR_COLUMNS, REAL_ID_TO_NODE)
    REAL_AGENT_CALL_PATTERNS = learn_agent_call_patterns(
        _cdr_df_for_ids, CDR_COLUMNS, REAL_ID_TO_NODE
    )
    real_edge_count = len(REAL_EDGES)
    if real_edge_count == 0:
        if TEMPLATE_AGENT_MODE:
            TEMPLATE_KNOWN_FRIEND_EDGES = build_template_known_friend_edges(
                len(REAL_ID_TO_NODE),
                TEMPLATE_KNOWN_FRIEND_PAIR_FRACTION,
                SEED + 19,
            )
            REAL_EDGES = TEMPLATE_KNOWN_FRIEND_EDGES
            REAL_AGENT_CALL_PATTERNS = apply_template_friend_contacts(
                REAL_AGENT_CALL_PATTERNS,
                TEMPLATE_KNOWN_FRIEND_EDGES,
                TEMPLATE_FRIEND_CALL_SHARE,
            )
            real_edge_count = len(REAL_EDGES)
            print(
                f"  Template mode created {real_edge_count} known center-to-center "
                f"friendships ({TEMPLATE_KNOWN_FRIEND_PAIR_FRACTION:.1%} of possible pairs)."
            )
            print(
                f"  Each center gets a star network; known friends receive a "
                f"{TEMPLATE_FRIEND_CALL_SHARE:.0%} interaction share and remaining contacts are dummy entities."
            )
        else:
            raise ValueError(
                "The CDR CSV loaded successfully, but no usable non-self caller/receiver "
                "edges were found. Check that the caller and receiver columns contain "
                "matching subscriber identifiers and are not all identical."
            )
    print(
        f"  Selected {len(REAL_ID_TO_NODE)} CSV-modeled agents and "
        f"{real_edge_count} base friendship edges."
    )

    REAL_CALL_BASE_RATE, REAL_CALL_HOURLY_MULT = learn_hourly_rate_pattern(
        _cdr_df_for_ids, CDR_COLUMNS['timestamp'], max(real_edge_count, 1), CDR_TIMESTAMP_PARTS
    )
    if REAL_CALL_BASE_RATE is not None:
        print(f"  Learned real hourly calling-rate pattern (base rate={REAL_CALL_BASE_RATE:.5f}/edge/hour).")
    else:
        print("  Could not parse real call timestamps; falling back to default day/night call pattern.")

    if IPDR_INPUT_PATH is not None and USE_IPDR_FOR_GENERATION:
        _ipdr_df_for_pattern = read_input_csv(IPDR_INPUT_PATH, 'IPDR')
        REAL_AGENT_SESSION_PATTERNS = learn_agent_session_patterns(
            _ipdr_df_for_pattern, IPDR_COLUMNS, REAL_ID_TO_NODE
        )
        REAL_SESSION_BASE_RATE, REAL_SESSION_HOURLY_MULT = learn_hourly_rate_pattern(
            _ipdr_df_for_pattern, IPDR_COLUMNS['timestamp'], max(real_edge_count, 1), IPDR_TIMESTAMP_PARTS
        )
        if REAL_SESSION_BASE_RATE is not None:
            print(f"  Learned real hourly session-rate pattern (base rate={REAL_SESSION_BASE_RATE:.5f}/edge/hour).")
        else:
            print("  Could not parse real session timestamps; falling back to default session rate.")


# Core-only mode: the target population is 1,000 agents by default. With
# multi-agent CDR input, these are the most active caller IDs; with a single
# caller, that caller becomes the behavior template for all synthetic agents.
# In that template mode each agent is the center of a star: low-frequency
# contacts are dummy spokes, while a small random set of center pairs is
# marked as known friendship and receives more interactions.
TOTAL_USERS = len(REAL_ID_TO_NODE) if USING_REAL_GRAPH else ARGS.synthetic_users
NUM_CORE_USERS = TOTAL_USERS

# Graph GAN hyperparameters
NOISE_DIM = 32          # per-node latent noise dimensionality fed to the Generator
EMBED_DIM = 32          # low-rank node embedding size used for the inner-product decoder
COND_EMBED_DIM = 16     # size of the per-node condition embedding derived from the known-friendship matrix
GAN_LR = 2e-3
TARGET_EDGES_PER_NODE = 3   # mirrors the old BA m=3 parameter, used to pin output sparsity when no real graph is given
# Conditioning hyperparameters
CONDITION_KNOWN_FRACTION = 0.05  # fraction of all node pairs whose true friend/stranger label is "already known"



# ======================================================================
# STEP 0: CONDITIONAL GRAPH GAN — learns to emit a realistic social-network
# skeleton that is consistent with a supplied "base-level truth" of known
# friendships/non-friendships, at full TOTAL_USERS scale.
# ======================================================================
# NOTE ON SCALE: the original version of this GAN represented the graph
# as a dense N x N adjacency matrix. That's fine at N=1,000-5,000 but is
# mathematically impossible at N=1,000,000 (a single N x N float32 matrix
# would need ~4 TB). So this version uses the standard technique real
# large-graph GANs/link-predictors use (e.g. GraphGAN, NetGAN): learn a
# per-node embedding via minibatches of sampled positive edges and random
# negative pairs, and never materialize a full pairwise matrix. The
# conditioning concept is unchanged -- a small known_fraction of real
# edges is revealed as "already known truth," and the trained model is
# used afterward to infer additional plausible connections -- it's just
# computed via sampling instead of dense matrix ops.

def build_structural_backbone(num_nodes, m, seed, real_edges=None):
    """
    Returns an (E, 2) int64 array of positive edges: the observed/base
    social structure the GAN is trained to be consistent with.

    Uses the real graph if supplied (as a sparse edge list, built by
    build_real_edges above), otherwise a scale-free Barabasi-Albert
    graph. Sparse/edge-list based throughout -- never a dense matrix --
    so it scales to millions of nodes regardless of source (benchmarked:
    ~19s for 1,000,000 synthetic nodes, ~3,000,000 edges).
    """
    if real_edges is not None:
        return real_edges
    g = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=seed)
    return np.array(g.edges(), dtype=np.int64)


def make_scalar_condition(num_nodes, edges, known_fraction, seed):
    """
    Scalable stand-in for the old dense N x N condition matrix: reveals a
    known_fraction of edges as "already known truth," then summarizes
    each node's condition as a single scalar (the fraction of ITS edges
    that were revealed), rather than a full row of per-neighbor labels.
    This scalar is what gets fed into the Generator as the conditioning
    signal for that node.
    """
    rng = np.random.RandomState(seed)
    reveal_mask = rng.rand(len(edges)) < known_fraction
    revealed = edges[reveal_mask]

    known_count = np.zeros(num_nodes, dtype=np.float32)
    degree_count = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(known_count, revealed[:, 0], 1)
    np.add.at(known_count, revealed[:, 1], 1)
    np.add.at(degree_count, edges[:, 0], 1)
    np.add.at(degree_count, edges[:, 1], 1)

    ratio = np.divide(known_count, np.maximum(degree_count, 1), dtype=np.float32)
    return ratio


def canonicalize_unique_edges(edges):
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    return np.unique(np.stack([lo, hi], axis=1), axis=0)


class ScalableGraphGenerator(nn.Module):
    """
    Produces an embedding for a batch of node indices as a function of a
    learned per-node identity embedding (nn.Embedding -- O(N) memory,
    completely fine even at N=1,000,000: ~128MB at embed_dim=32), fresh
    noise, and that node's scalar known-friendship condition. Trained via
    minibatches, never a dense matrix.
    """
    def __init__(self, num_nodes, noise_dim, embed_dim, cond_embed_dim):
        super().__init__()
        self.identity = nn.Embedding(num_nodes, embed_dim)
        self.noise_proj = nn.Linear(noise_dim, embed_dim)
        self.cond_encoder = nn.Sequential(nn.Linear(1, cond_embed_dim), nn.LeakyReLU(0.2))
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 2 + cond_embed_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, embed_dim),
        )

    def forward(self, node_idx, noise, cond_scalar):
        base = self.identity(node_idx)
        noise_emb = self.noise_proj(noise)
        cond_emb = self.cond_encoder(cond_scalar.unsqueeze(-1))
        return self.combine(torch.cat([base, noise_emb, cond_emb], dim=-1))


class ScalableGraphDiscriminator(nn.Module):
    """Scores one embedding pair (node_u, node_v) as a real vs. fake edge."""
    def __init__(self, embed_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb_u, emb_v):
        return self.net(torch.cat([emb_u, emb_v], dim=-1)).squeeze(-1)


def train_scalable_graph_gan(num_nodes, positive_edges, condition_ratio,
                              noise_dim, embed_dim, cond_embed_dim,
                              steps, batch_size, lr):
    print("Step 0: Training the scalable Conditional Graph GAN "
          f"(minibatch/negative-sampling, {num_nodes} nodes)...")

    generator = ScalableGraphGenerator(num_nodes, noise_dim, embed_dim, cond_embed_dim).to(DEVICE)
    discriminator = ScalableGraphDiscriminator(embed_dim).to(DEVICE)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    cond_t = torch.tensor(condition_ratio, device=DEVICE)
    n_edges = positive_edges.shape[0]

    for step in range(steps):
        pos_idx = np.random.randint(0, n_edges, batch_size)
        pu = torch.tensor(positive_edges[pos_idx, 0], device=DEVICE)
        pv = torch.tensor(positive_edges[pos_idx, 1], device=DEVICE)
        nu = torch.randint(0, num_nodes, (batch_size,), device=DEVICE)
        nv = torch.randint(0, num_nodes, (batch_size,), device=DEVICE)

        # ---- Discriminator step: real sampled edges vs. random pairs ----
        with torch.no_grad():
            emb_pu = generator(pu, torch.randn(batch_size, noise_dim, device=DEVICE), cond_t[pu])
            emb_pv = generator(pv, torch.randn(batch_size, noise_dim, device=DEVICE), cond_t[pv])
            emb_nu = generator(nu, torch.randn(batch_size, noise_dim, device=DEVICE), cond_t[nu])
            emb_nv = generator(nv, torch.randn(batch_size, noise_dim, device=DEVICE), cond_t[nv])

        d_real = discriminator(emb_pu, emb_pv)
        d_fake = discriminator(emb_nu, emb_nv)
        d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # ---- Generator step: make a fresh random pair's embeddings look like a real edge ----
        gu = torch.randint(0, num_nodes, (batch_size,), device=DEVICE)
        gv = torch.randint(0, num_nodes, (batch_size,), device=DEVICE)
        emb_gu = generator(gu, torch.randn(batch_size, noise_dim, device=DEVICE), cond_t[gu])
        emb_gv = generator(gv, torch.randn(batch_size, noise_dim, device=DEVICE), cond_t[gv])
        g_score = discriminator(emb_gu, emb_gv)
        g_loss = bce(g_score, torch.ones_like(g_score))
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"  step {step:5d}/{steps} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

    generator.eval()
    return generator, discriminator


def infer_additional_edges(generator, discriminator, num_nodes, condition_ratio,
                            n_candidates, top_fraction, noise_dim):
    """
    Scalable analogue of the old dense top-K-over-N^2 edge selection:
    scores a large random SAMPLE of candidate pairs (never every possible
    pair, which is infeasible at this N) with the trained embeddings, and
    keeps the top-scoring fraction as additional GAN-inferred connections
    on top of the structural backbone.
    """
    generator.eval()
    with torch.no_grad():
        cu = torch.randint(0, num_nodes, (n_candidates,), device=DEVICE)
        cv = torch.randint(0, num_nodes, (n_candidates,), device=DEVICE)
        cond_t = torch.tensor(condition_ratio, device=DEVICE)
        emb_u = generator(cu, torch.randn(n_candidates, noise_dim, device=DEVICE), cond_t[cu])
        emb_v = generator(cv, torch.randn(n_candidates, noise_dim, device=DEVICE), cond_t[cv])
        scores = torch.sigmoid(discriminator(emb_u, emb_v)).cpu().numpy()
    cu_np, cv_np = cu.cpu().numpy(), cv.cpu().numpy()

    not_self = cu_np != cv_np
    cu_np, cv_np, scores = cu_np[not_self], cv_np[not_self], scores[not_self]
    keep_n = int(n_candidates * top_fraction)
    if keep_n <= 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)
    keep_n = min(keep_n, len(scores))
    top_idx = np.argpartition(-scores, keep_n - 1)[:keep_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    edges = np.stack([cu_np[top_idx], cv_np[top_idx]], axis=1)
    edges = np.stack([np.minimum(edges[:, 0], edges[:, 1]), np.maximum(edges[:, 0], edges[:, 1])], axis=1)
    inferred_df = pd.DataFrame({
        'node_u': edges[:, 0],
        'node_v': edges[:, 1],
        'gan_friend_score': scores[top_idx],
    })
    inferred_df = (
        inferred_df
        .groupby(['node_u', 'node_v'], as_index=False)['gan_friend_score']
        .max()
        .sort_values('gan_friend_score', ascending=False)
    )
    return (
        inferred_df[['node_u', 'node_v']].to_numpy(dtype=np.int64),
        inferred_df['gan_friend_score'].to_numpy(dtype=np.float32),
    )


GAN_STEPS = ARGS.gan_steps
GAN_BATCH_SIZE = ARGS.gan_batch_size
CANDIDATE_SAMPLE_SIZE = ARGS.candidate_sample_size
GAN_INFERRED_TOP_FRACTION = ARGS.gan_inferred_top_fraction

positive_edges = canonicalize_unique_edges(
    build_structural_backbone(TOTAL_USERS, TARGET_EDGES_PER_NODE, SEED, real_edges=REAL_EDGES)
)
condition_ratio = make_scalar_condition(TOTAL_USERS, positive_edges, CONDITION_KNOWN_FRACTION, SEED)

generator, discriminator = train_scalable_graph_gan(
    TOTAL_USERS, positive_edges, condition_ratio, NOISE_DIM, EMBED_DIM, COND_EMBED_DIM,
    steps=GAN_STEPS, batch_size=GAN_BATCH_SIZE, lr=GAN_LR
)

inferred_edges, inferred_edge_scores = infer_additional_edges(
    generator, discriminator, TOTAL_USERS, condition_ratio,
    n_candidates=CANDIDATE_SAMPLE_SIZE, top_fraction=GAN_INFERRED_TOP_FRACTION, noise_dim=NOISE_DIM
)

all_edges = canonicalize_unique_edges(np.concatenate([positive_edges, inferred_edges], axis=0))
G = nx.Graph()
G.add_nodes_from(range(TOTAL_USERS))
G.add_edges_from(map(tuple, all_edges.tolist()))

print(f"Graph GAN produced a crosslink graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges "
      f"({len(positive_edges)} structural backbone + {len(inferred_edges)} GAN-inferred).")


# ======================================================================
# Everything below is unchanged from the original pipeline: it consumes
# the NetworkX graph `G` (now Conditional-GAN-generated instead of
# nx.barabasi_albert_graph) exactly as before.
# ======================================================================

print("Step 1: Assigning subscriber attributes onto the Invisible Structural Core...")

# Geopolitical circle distributions in India with real-world weights
INDIAN_CIRCLES = {
    'MH': {'name': 'Maharashtra & Mumbai', 'weight': 0.20, 'is_tier_1': True},
    'DL': {'name': 'Delhi NCR', 'weight': 0.15, 'is_tier_1': True},
    'KA': {'name': 'Karnataka', 'weight': 0.15, 'is_tier_1': True},
    'UP': {'name': 'Uttar Pradesh', 'weight': 0.30, 'is_tier_1': False},
    'WB': {'name': 'West Bengal', 'weight': 0.20, 'is_tier_1': False}
}

circle_codes = list(INDIAN_CIRCLES.keys())
circle_weights = [info['weight'] for info in INDIAN_CIRCLES.values()]

## --- TABLE 1: USERS (The Parent Entity) ---
# When real data was supplied, keep a lookup back from node index to the
# original real identifier so subscriber rows can be traced to source.
node_to_real_id = None
if USING_REAL_GRAPH:
    node_to_real_id = {
        idx: SOURCE_PROFILE_PLAN['source_profile_for_agent'].get(agent_id, agent_id)
        for agent_id, idx in REAL_ID_TO_NODE.items()
    }

print(
    f"  Generating {TOTAL_USERS} core subscriber profiles with full "
    "IMEI/IMSI/name/address detail + IPDR activity..."
)

# Vectorized bulk generation for the FULL population -- this has to stay
# vectorized where practical; rich Faker attributes are generated once per
# selected agent below.
gen_rng = np.random.default_rng(SEED + 7)
all_idx = np.arange(TOTAL_USERS)
circle_arr = gen_rng.choice(circle_codes, size=TOTAL_USERS, p=circle_weights)
tier_1_circles = [c for c in circle_codes if INDIAN_CIRCLES[c]['is_tier_1']]
is_tier_1_arr = np.isin(circle_arr, tier_1_circles)

msisdn_numbers = gen_rng.integers(6000000000, 9999999999, size=TOTAL_USERS)
tower_numbers = gen_rng.integers(1000, 1500, size=TOTAL_USERS)
device_5g4g = gen_rng.choice(['5G', '4G'], size=TOTAL_USERS)
device_4g2g = gen_rng.choice(['4G', '2G'], size=TOTAL_USERS)
premium_roll = gen_rng.random(TOTAL_USERS)

df_subscribers = pd.DataFrame({
    'subscriber_id': [f"SUB_{i:07d}" for i in all_idx],
    'msisdn': [f"+91{n}" for n in msisdn_numbers],
    'telecom_circle': circle_arr,
    'home_tower_id': [f"TWR_{c}_{t}" for c, t in zip(circle_arr, tower_numbers)],
    'device_capability': np.where(is_tier_1_arr, device_5g4g, device_4g2g),
    'base_data_tier': np.where(is_tier_1_arr & (premium_roll > 0.4), 'Premium', 'Standard'),
    'is_core_user': False,
    'fake_full_name': None,
    'fake_address': None,
    'imei': None,
    'imsi': None,
    'connection_type': None,
    'apn': None,
})

if node_to_real_id is not None:
    df_subscribers['source_identifier'] = [node_to_real_id.get(i, '') for i in all_idx]
    df_subscribers['source_profile_id'] = [node_to_real_id.get(i, '') for i in all_idx]
else:
    df_subscribers['source_identifier'] = ''
    df_subscribers['source_profile_id'] = ''
df_subscribers['generation_mode'] = EFFECTIVE_GENERATION_MODE or normalize_generation_mode(ARGS.generation_mode) or 'cdr_only'
df_subscribers['identity_answer'] = ARGS.identity_answer or ''
df_subscribers['identity_evidence_status'] = 'synthetic_lineage_only'

# Rich per-row enrichment for every agent in core-only mode.
core_slice = df_subscribers.index[:NUM_CORE_USERS]
apn_choices = ['jionet', 'airtelgprs.com', 'vodafone3g', 'ideanetsetter', 'bsnlnet']
df_subscribers.loc[core_slice, 'is_core_user'] = True
df_subscribers.loc[core_slice, 'fake_full_name'] = [fake.name() for _ in core_slice]
df_subscribers.loc[core_slice, 'fake_address'] = [fake.address().replace("\n", ", ") for _ in core_slice]
df_subscribers.loc[core_slice, 'imei'] = [f"{random.randint(10**14, 10**15 - 1)}" for _ in core_slice]
df_subscribers.loc[core_slice, 'imsi'] = [f"40{random.randint(10**11, 10**12 - 1)}" for _ in core_slice]
df_subscribers.loc[core_slice, 'connection_type'] = [random.choice(['PREPAID', 'POSTPAID']) for _ in core_slice]
df_subscribers.loc[core_slice, 'apn'] = [random.choice(apn_choices) for _ in core_slice]


def sample_non_friend_pairs(num_nodes, excluded_edges, requested_count, seed):
    """Sample unordered pairs that are absent from the generated friend graph.

    This uses rejection sampling so it remains practical when the population is
    large and the graph is sparse. For small populations, a deterministic
    complement fallback guarantees that an available pair is returned whenever
    one exists.
    """
    requested_count = max(int(requested_count), 0)
    if num_nodes < 2 or requested_count == 0:
        return np.zeros((0, 2), dtype=np.int64)

    excluded = canonicalize_unique_edges(np.asarray(excluded_edges, dtype=np.int64))
    excluded_keys = {
        int(u) * int(num_nodes) + int(v)
        for u, v in excluded
        if 0 <= int(u) < num_nodes and 0 <= int(v) < num_nodes and int(u) != int(v)
    }
    total_pairs = int(num_nodes) * (int(num_nodes) - 1) // 2
    available_pairs = max(total_pairs - len(excluded_keys), 0)
    target_count = min(requested_count, available_pairs)
    if target_count == 0:
        return np.zeros((0, 2), dtype=np.int64)

    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    max_attempts = max(10_000, target_count * 100)
    attempts = 0
    while len(chosen) < target_count and attempts < max_attempts:
        batch_size = min(max(1024, (target_count - len(chosen)) * 8), 65_536)
        first = rng.integers(0, num_nodes, size=batch_size, dtype=np.int64)
        second = rng.integers(0, num_nodes, size=batch_size, dtype=np.int64)
        low = np.minimum(first, second)
        high = np.maximum(first, second)
        valid = low != high
        for u, v in zip(low[valid], high[valid]):
            key = int(u) * int(num_nodes) + int(v)
            if key not in excluded_keys:
                chosen.add(key)
                if len(chosen) >= target_count:
                    break
        attempts += batch_size

    # With a dense graph, rejection can be inefficient. A bounded complement
    # enumeration handles the small-population case without risking a dense
    # matrix at the large scales this generator supports.
    if len(chosen) < target_count and total_pairs <= 2_000_000:
        upper_u, upper_v = np.triu_indices(num_nodes, k=1)
        upper_keys = upper_u.astype(np.int64) * np.int64(num_nodes) + upper_v.astype(np.int64)
        unavailable_keys = np.fromiter(
            excluded_keys | chosen,
            dtype=np.int64,
        )
        available_mask = ~np.isin(upper_keys, unavailable_keys)
        available_indices = np.flatnonzero(available_mask)
        remaining = target_count - len(chosen)
        if remaining > 0 and len(available_indices):
            remaining = min(remaining, len(available_indices))
            selected = rng.choice(available_indices, size=remaining, replace=False)
            chosen.update(int(upper_keys[index]) for index in selected)

    pairs = np.asarray(
        [[key // int(num_nodes), key % int(num_nodes)] for key in sorted(chosen)],
        dtype=np.int64,
    )
    return pairs[:target_count]


def export_conditional_gan_friend_edges(positive_edges, inferred_edges, inferred_scores, subscribers):
    """
    Exports the graph pairs that this pipeline treats as "friends"/contact
    links. Observed edges come from the training backbone; inferred edges
    are the top-scoring pairs selected by the Conditional GAN.
    """
    if TEMPLATE_AGENT_MODE:
        observed_source = 'template_known_friendship'
    elif USING_REAL_GRAPH:
        observed_source = 'observed_cdr_training_edge'
    else:
        observed_source = 'synthetic_structural_training_edge'
    subscriber_ids = subscribers['subscriber_id'].to_numpy()
    msisdns = subscribers['msisdn'].to_numpy()
    if 'source_identifier' in subscribers.columns:
        source_ids = subscribers['source_identifier'].fillna('').astype(str).to_numpy()
    else:
        source_ids = np.full(len(subscribers), '', dtype=object)

    observed_df = pd.DataFrame({
        'node_u': positive_edges[:, 0],
        'node_v': positive_edges[:, 1],
        'edge_source': observed_source,
        'is_observed_training_edge': True,
        'is_gan_inferred_friend': False,
        'gan_friend_score': np.nan,
    })
    inferred_df = pd.DataFrame({
        'node_u': inferred_edges[:, 0],
        'node_v': inferred_edges[:, 1],
        'edge_source': 'conditional_gan_inferred_friend',
        'is_observed_training_edge': False,
        'is_gan_inferred_friend': True,
        'gan_friend_score': inferred_scores,
    })

    observed_key = (
        observed_df['node_u'].to_numpy(dtype=np.int64) * np.int64(TOTAL_USERS)
        + observed_df['node_v'].to_numpy(dtype=np.int64)
    )
    inferred_key = (
        inferred_df['node_u'].to_numpy(dtype=np.int64) * np.int64(TOTAL_USERS)
        + inferred_df['node_v'].to_numpy(dtype=np.int64)
    )
    inferred_overlaps_observed = np.isin(inferred_key, observed_key)

    overlap_df = inferred_df[inferred_overlaps_observed].copy()
    if not overlap_df.empty:
        overlap_df['edge_source'] = observed_source + '+conditional_gan_inferred_friend'
        overlap_df['is_observed_training_edge'] = True

    observed_without_overlap = observed_df[~np.isin(observed_key, inferred_key)]
    inferred_only = inferred_df[~inferred_overlaps_observed]
    edge_parts = [part for part in (observed_without_overlap, overlap_df, inferred_only) if not part.empty]
    for part in edge_parts:
        part['gan_friend_score'] = pd.to_numeric(
            part['gan_friend_score'], errors='coerce'
        ).astype('float64')
    friend_edges = pd.concat(edge_parts, ignore_index=True)
    friend_edges = friend_edges.sort_values(
        ['is_gan_inferred_friend', 'gan_friend_score', 'node_u', 'node_v'],
        ascending=[False, False, True, True],
        na_position='last',
    ).reset_index(drop=True)

    u = friend_edges['node_u'].to_numpy(dtype=np.int64)
    v = friend_edges['node_v'].to_numpy(dtype=np.int64)
    friend_edges.insert(0, 'friend_edge_id', [f"FRIEND_{i:07d}" for i in range(len(friend_edges))])
    friend_edges['subscriber_id_u'] = subscriber_ids[u]
    friend_edges['subscriber_id_v'] = subscriber_ids[v]
    friend_edges['msisdn_u'] = msisdns[u]
    friend_edges['msisdn_v'] = msisdns[v]
    friend_edges['source_identifier_u'] = source_ids[u]
    friend_edges['source_identifier_v'] = source_ids[v]
    # Keep the existing positive edge exports backward-compatible, while
    # making their positive truth explicit for downstream classifiers.
    friend_edges['label'] = 1
    friend_edges['truth_label'] = 1
    friend_edges['relationship_truth'] = 'friend_or_interacting'
    friend_edges['truth_source'] = friend_edges['edge_source']

    all_friend_edges = canonicalize_unique_edges(
        np.concatenate([positive_edges, inferred_edges], axis=0)
    )
    requested_negative_count = int(round(len(friend_edges) * ARGS.negative_truth_ratio))
    negative_edges = sample_non_friend_pairs(
        TOTAL_USERS,
        all_friend_edges,
        requested_negative_count,
        SEED + 131,
    )
    negative_df = pd.DataFrame({
        'node_u': negative_edges[:, 0],
        'node_v': negative_edges[:, 1],
        'edge_source': 'sampled_non_friend',
        'is_observed_training_edge': False,
        'is_gan_inferred_friend': False,
        'gan_friend_score': np.nan,
        'label': 0,
        'truth_label': 0,
        'relationship_truth': 'not_friend_or_interacting',
        'truth_source': 'sampled_non_friend',
    })
    negative_df.insert(
        0,
        'friend_edge_id',
        [f"NONFRIEND_{index:07d}" for index in range(len(negative_df))],
    )
    negative_u = negative_df['node_u'].to_numpy(dtype=np.int64)
    negative_v = negative_df['node_v'].to_numpy(dtype=np.int64)
    negative_df['subscriber_id_u'] = subscriber_ids[negative_u]
    negative_df['subscriber_id_v'] = subscriber_ids[negative_v]
    negative_df['msisdn_u'] = msisdns[negative_u]
    negative_df['msisdn_v'] = msisdns[negative_v]
    negative_df['source_identifier_u'] = source_ids[negative_u]
    negative_df['source_identifier_v'] = source_ids[negative_v]

    truth_pairs = pd.concat([friend_edges, negative_df], ignore_index=True)
    truth_pairs.insert(
        0,
        'truth_pair_id',
        [f"TRUTH_{index:07d}" for index in range(len(truth_pairs))],
    )
    # These aliases make the file directly consumable by the supervised
    # friend-interaction classifier without a column-renaming step.
    truth_pairs['person_a'] = truth_pairs['msisdn_u']
    truth_pairs['person_b'] = truth_pairs['msisdn_v']

    conditional_friend_edges_path = write_csv_safely(
        friend_edges,
        "conditional_gan_friend_edges.csv",
        index=False,
    )
    gan_friend_edges_path = write_csv_safely(
        friend_edges[friend_edges['is_gan_inferred_friend']],
        "gan_inferred_friend_edges.csv",
        index=False,
    )
    friendship_truth_path = write_csv_safely(
        truth_pairs,
        "friendship_truth_pairs.csv",
        index=False,
    )

    if ARGS.output_mode in ("clean", "both"):
        clean_edge_outputs = [
            (conditional_friend_edges_path, "conditional_gan_friend_edges_clean.csv"),
            (gan_friend_edges_path, "gan_inferred_friend_edges_clean.csv"),
            (friendship_truth_path, "friendship_truth_pairs_clean.csv"),
        ]
        for source_path, clean_path in clean_edge_outputs:
            filter_result = filter_cdr_file(
                source_path,
                clean_path,
                letter_columns=FRIEND_EDGE_NUMBER_COLUMNS,
                keywords=tuple(split_csv_arg(ARGS.cdr_filter_keywords)),
                strict_columns=False,
            )
            print(
                f"  Cleaned {source_path}: {filter_result['rows_removed']} removed, "
                f"{filter_result['rows_kept']} kept -> {filter_result['output']}"
            )
    if ARGS.output_mode == "clean":
        os.remove(conditional_friend_edges_path)
        os.remove(gan_friend_edges_path)
        os.remove(friendship_truth_path)

    print(
        f"  Exported {len(friend_edges)} Conditional-GAN friend/contact edges "
        f"({friend_edges['is_gan_inferred_friend'].sum()} GAN-inferred) and "
        f"{len(negative_df)} sampled non-friend truth pairs to friendship_truth_pairs.csv "
        f"using output mode '{ARGS.output_mode}'."
    )


export_conditional_gan_friend_edges(positive_edges, inferred_edges, inferred_edge_scores, df_subscribers)

# NOTE: subscriber attributes are looked up from df_subscribers arrays
# directly during CDR/IPDR generation below (fast, vectorized) rather
# than being attached to G's node dicts one at a time -- attaching a
# dict per node in a Python loop would be far too slow at TOTAL_USERS=
# 1,000,000 and isn't needed by anything downstream.

print("Step 2: Simulating CSV-conditioned behavioral footprints for every core agent...")
# Set simulation timeline window within June 2026
START_WINDOW = datetime(2026, 6, 15, 0, 0, 0)

# Interactions are placed across this many simulated calendar days.
SIMULATION_DAYS = ARGS.simulation_days

# ----------------------------------------------------------------
# Core-only behavior configuration
# ----------------------------------------------------------------
# All 1,000 selected agents are core users. Both CDR and IPDR volumes are
# exact per-agent counts controlled by command-line arguments.

CROSSLINK_CALL_SHARE = 0.06     # fallback when no per-agent CSV behavior is available

DEST_GATEWAYS = ["142.250.190.46", "157.240.22.35", "104.244.42.1", "31.13.71.36"]  # generic app endpoints

rng = np.random.default_rng(SEED)

# Diurnal shape used to pick which hour a call/session lands in --
# grounded in your real hourly pattern when available, else a default
# day/night shape.
if REAL_CALL_HOURLY_MULT is not None:
    call_hour_probs = np.array([REAL_CALL_HOURLY_MULT.get(h, 1.0) for h in range(24)], dtype=float)
else:
    call_hour_probs = np.array([5.0 if 9 <= h <= 22 else 0.5 for h in range(24)], dtype=float)
call_hour_probs = call_hour_probs / call_hour_probs.sum()

if REAL_SESSION_HOURLY_MULT is not None:
    session_hour_probs = np.array([REAL_SESSION_HOURLY_MULT.get(h, 1.0) for h in range(24)], dtype=float)
else:
    session_hour_probs = np.array([4.0 if 8 <= h <= 23 else 1.0 for h in range(24)], dtype=float)
session_hour_probs = session_hour_probs / session_hour_probs.sum()

subscriber_id_arr = df_subscribers['subscriber_id'].to_numpy()
msisdn_arr = df_subscribers['msisdn'].to_numpy()
home_tower_arr = df_subscribers['home_tower_id'].to_numpy()
telecom_circle_arr = df_subscribers['telecom_circle'].to_numpy()
is_core_arr = df_subscribers['is_core_user'].to_numpy()

# ----------------------------------------------------------------
# CSR-style flattened neighbor structure, built entirely with vectorized
# numpy ops (sort + bincount + cumsum) rather than a per-node Python
# loop, so it scales to millions of nodes / edges.
# ----------------------------------------------------------------
edge_arr = np.array(list(G.edges()), dtype=np.int64)
edges_both = np.concatenate([edge_arr, edge_arr[:, [1, 0]]], axis=0)
sort_order = np.argsort(edges_both[:, 0], kind='stable')
edges_sorted = edges_both[sort_order]
degrees = np.bincount(edges_sorted[:, 0], minlength=TOTAL_USERS)
offsets = np.zeros(TOTAL_USERS + 1, dtype=np.int64)
offsets[1:] = np.cumsum(degrees)
neighbors_flat = edges_sorted[:, 1]


def sample_random_neighbor(callers, offsets, neighbors_flat, degrees, rng):
    """Vectorized: picks one random real neighbor for each caller in `callers`."""
    deg = degrees[callers]
    neighbor = callers.copy()
    connected = deg > 0
    if connected.any():
        connected_callers = callers[connected]
        connected_deg = deg[connected]
        rand_offset = (rng.random(connected.sum()) * connected_deg).astype(np.int64)
        idx = offsets[connected_callers] + rand_offset
        neighbor[connected] = neighbors_flat[idx]
    return neighbor  # isolated nodes fall back to self


def assign_template_dummy_entities(receiver_values, caller_indices, dummy_mask, rng):
    """Give each star's non-friend calls one-off/two-off dummy contacts.

    Dummy IDs are scoped to their center, so a dummy contact belongs to one
    star only. Each generated dummy is used at most twice, matching the
    requested one-off/two-off interaction interpretation.
    """
    for node in range(TOTAL_USERS):
        positions = np.flatnonzero(dummy_mask & (caller_indices == node))
        if not len(positions):
            continue
        dummy_count = int(np.ceil(len(positions) / TEMPLATE_DUMMY_MAX_INTERACTIONS))
        dummy_ids = np.asarray(
            [f"DUMMY_{node + 1:04d}_{index + 1:05d}" for index in range(dummy_count)],
            dtype=object,
        )
        values = np.repeat(dummy_ids, TEMPLATE_DUMMY_MAX_INTERACTIONS)[:len(positions)]
        rng.shuffle(values)
        receiver_values[positions] = values


total_calls_per_user = np.full(TOTAL_USERS, ARGS.interactions_per_user, dtype=np.int64)
total_sessions_per_user = np.full(TOTAL_USERS, ARGS.sessions_per_user, dtype=np.int64)

total_calls = int(total_calls_per_user.sum())
total_sessions = int(total_sessions_per_user.sum())
print(
    f"  Target volume: exactly {ARGS.interactions_per_user} CDR interactions per agent "
    f"({total_calls} rows total) and exactly {ARGS.sessions_per_user} IPDR sessions "
    f"per agent ({total_sessions} rows total) across {TOTAL_USERS} core users."
)

# ---- CDR generation, fully vectorized across every user at once ----
caller_idx = np.repeat(np.arange(TOTAL_USERS), total_calls_per_user)
day_idx = rng.integers(0, SIMULATION_DAYS, size=total_calls)
if REAL_AGENT_CALL_PATTERNS is not None:
    hour_idx = np.concatenate([
        rng.choice(pattern['hours'], size=ARGS.interactions_per_user, replace=True)
        for pattern in REAL_AGENT_CALL_PATTERNS
    ])
    durations = np.concatenate([
        rng.choice(pattern['durations'], size=ARGS.interactions_per_user, replace=True)
        for pattern in REAL_AGENT_CALL_PATTERNS
    ])
    crosslink_share_by_user = np.array([
        pattern['crosslink_share'] for pattern in REAL_AGENT_CALL_PATTERNS
    ], dtype=float)
else:
    hour_idx = rng.choice(24, size=total_calls, p=call_hour_probs)
    durations = rng.integers(10, 600, total_calls)
    crosslink_share_by_user = np.full(TOTAL_USERS, CROSSLINK_CALL_SHARE, dtype=float)
mins = rng.integers(0, 60, total_calls)
secs = rng.integers(0, 60, total_calls)
seconds_offset = day_idx.astype(np.int64) * 86400 + hour_idx * 3600 + mins * 60 + secs
timestamps = START_WINDOW + pd.to_timedelta(seconds_offset, unit='s')

is_crosslink_call = rng.random(total_calls) < crosslink_share_by_user[caller_idx]
receiver_vals = np.empty(total_calls, dtype=object)

n_external = int((~is_crosslink_call).sum())
external_numbers = rng.integers(6000000000, 9999999999, size=n_external)
receiver_vals[~is_crosslink_call] = [f"+91{n}" for n in external_numbers]

crosslink_callers = caller_idx[is_crosslink_call]
neighbor_idx = np.empty(len(crosslink_callers), dtype=np.int64)
needs_graph_fallback = np.ones(len(crosslink_callers), dtype=bool)
if REAL_AGENT_CALL_PATTERNS is not None:
    for node, pattern in enumerate(REAL_AGENT_CALL_PATTERNS):
        positions = np.flatnonzero(crosslink_callers == node)
        if len(positions) and len(pattern['contacts']):
            neighbor_idx[positions] = rng.choice(
                pattern['contacts'], size=len(positions), replace=True
            )
            needs_graph_fallback[positions] = False
if needs_graph_fallback.any():
    neighbor_idx[needs_graph_fallback] = sample_random_neighbor(
        crosslink_callers[needs_graph_fallback], offsets, neighbors_flat, degrees, rng
    )
receiver_vals[is_crosslink_call] = msisdn_arr[neighbor_idx]

contact_type = np.full(total_calls, 'external_contact', dtype=object)
contact_type[is_crosslink_call] = 'known_friend'
if TEMPLATE_AGENT_MODE:
    assign_template_dummy_entities(
        receiver_vals,
        caller_idx,
        ~is_crosslink_call,
        rng,
    )
    contact_type[~is_crosslink_call] = 'dummy_entity'

late_hour_mask = hour_idx >= 18
routing_towers = home_tower_arr[caller_idx].astype(object)
n_late = int(late_hour_mask.sum())
if n_late:
    late_circles = telecom_circle_arr[caller_idx[late_hour_mask]]
    late_tower_nums = rng.integers(2000, 2200, n_late)
    routing_towers[late_hour_mask] = [f"TWR_{c}_{int(t)}" for c, t in zip(late_circles, late_tower_nums)]

df_cdr = pd.DataFrame({
    'caller_id': subscriber_id_arr[caller_idx],       # Relational Foreign Key to Users table
    'receiver_msisdn': receiver_vals,                  # may be another subscriber OR a pure external number
    'contact_type': contact_type,
    'timestamp': timestamps,
    'duration_seconds': durations,
    'routing_tower': routing_towers,
})
df_cdr.insert(0, 'call_id', [f"CALL_{100000 + i}" for i in range(len(df_cdr))])
df_cdr['is_core_user'] = is_core_arr[caller_idx]

# ---- IPDR generation: each subscriber's OWN session log, vectorized ----
# (mirrors your uploaded IPDR file, which tracks one subscriber's own
# public-IP sessions rather than pairing two subscribers together)
subscriber_idx_ipdr = np.repeat(np.arange(TOTAL_USERS), total_sessions_per_user)
day_idx_s = rng.integers(0, SIMULATION_DAYS, size=total_sessions)
if REAL_AGENT_SESSION_PATTERNS is not None:
    hour_idx_s = np.concatenate([
        rng.choice(pattern['hours'], size=ARGS.sessions_per_user, replace=True)
        for pattern in REAL_AGENT_SESSION_PATTERNS
    ])
    dest_ips = np.concatenate([
        rng.choice(pattern['destinations'], size=ARGS.sessions_per_user, replace=True)
        for pattern in REAL_AGENT_SESSION_PATTERNS
    ])
    mb = np.concatenate([
        rng.choice(pattern['volumes'], size=ARGS.sessions_per_user, replace=True)
        for pattern in REAL_AGENT_SESSION_PATTERNS
    ]).round(3)
    destination_ports = np.concatenate([
        rng.choice(
            pattern['destination_ports'],
            size=ARGS.sessions_per_user,
            replace=True,
            p=pattern['destination_port_weights'],
        )
        for pattern in REAL_AGENT_SESSION_PATTERNS
    ])
else:
    hour_idx_s = rng.choice(24, size=total_sessions, p=session_hour_probs)
    dest_ips = rng.choice(DEST_GATEWAYS, size=total_sessions)
    mb = rng.uniform(0.5, 25.0, size=total_sessions).round(3)
    destination_ports = rng.choice(
        COMMON_DESTINATION_PORTS,
        size=total_sessions,
        p=COMMON_DESTINATION_PORT_WEIGHTS,
    )
mins_s = rng.integers(0, 60, total_sessions)
secs_s = rng.integers(0, 60, total_sessions)
seconds_offset_s = day_idx_s.astype(np.int64) * 86400 + hour_idx_s * 3600 + mins_s * 60 + secs_s
timestamps_s = START_WINDOW + pd.to_timedelta(seconds_offset_s, unit='s')

df_ipdr = pd.DataFrame({
    'subscriber_id': subscriber_id_arr[subscriber_idx_ipdr],   # Relational Foreign Key to Users table
    'timestamp': timestamps_s,
    'destination_ip': dest_ips,
    'destination_port': destination_ports.astype(np.int64),
    'megabytes_transferred': mb,
})
df_ipdr.insert(0, 'session_id', [f"SES_{500000 + i}" for i in range(len(df_ipdr))])
df_ipdr['is_core_user'] = is_core_arr[subscriber_idx_ipdr]

print("Step 3: Compiling Tables and Establishing Relational AI Metadata Mapping...")

# Group tables into a multi-table relational package
data_tables = {
    'subscribers': df_subscribers,
    'call_logs': df_cdr,
    'internet_sessions': df_ipdr
}

# Define the data dictionary schema programmatically for the SDV engine
metadata = MultiTableMetadata()
metadata.detect_from_dataframes(data_tables)

# 1. Map primary keys for each table
expected_primary_keys = {
    'subscribers': 'subscriber_id',
    'call_logs': 'call_id',
    'internet_sessions': 'session_id',
}
detected_tables = metadata.to_dict()['tables']
for table_name, column_name in expected_primary_keys.items():
    if detected_tables[table_name].get('primary_key') != column_name:
        metadata.set_primary_key(table_name=table_name, column_name=column_name)

# 2. Enforce structural relational links (Foreign Key constraints)
expected_relationships = [
    {
        'parent_table_name': 'subscribers',
        'child_table_name': 'call_logs',
        'parent_primary_key': 'subscriber_id',
        'child_foreign_key': 'caller_id',
    },
    {
        'parent_table_name': 'subscribers',
        'child_table_name': 'internet_sessions',
        'parent_primary_key': 'subscriber_id',
        'child_foreign_key': 'subscriber_id',
    },
]
for relationship in expected_relationships:
    if relationship not in metadata.relationships:
        metadata.add_relationship(**relationship)

print("\n>>> SIMULATION PIPELINE COMPLETE <<<")
print(f"Generated {len(df_subscribers)} distinct profiles.")
print(f"Captured {len(df_cdr)} relational voice records mapping to GAN-generated graph links.")
print(f"Captured {len(df_ipdr)} synced data session packets mapping to GAN-generated graph links.")

# Save the GUI-facing data files. Clean mode uses distinct filenames and is
# trained from the filtered input; normal mode preserves the unfiltered path.
output_suffix = "_clean" if ARGS.output_mode == "clean" else ""
write_csv_safely(df_subscribers, f"subscribers{output_suffix}.csv", index=False)
write_csv_safely(df_cdr, f"call_logs{output_suffix}.csv", index=False)
write_csv_safely(df_ipdr, f"internet_sessions{output_suffix}.csv", index=False)
metadata_path = "telecom_relational_metadata.json"
if os.path.exists(metadata_path):
    os.remove(metadata_path)
metadata.save_to_json(metadata_path)
run_metadata = {
    "metadata_version": 1,
    "identity_answer": ARGS.identity_answer,
    "generation_mode": EFFECTIVE_GENERATION_MODE or normalize_generation_mode(ARGS.generation_mode) or 'cdr_only',
    "identity_mapping": IDENTITY_MAPPING.public(),
    "identity_mappings": IDENTITY_MAPPING.metadata(),
    "identity_validation": IDENTITY_VALIDATION,
    "identity_decision": IDENTITY_DECISION,
    "source_profile_plan": SOURCE_PROFILE_PLAN,
    "source_lineage_statement": (
        "Generated subscriber IDs are distinct synthetic identities. Source profile fields describe "
        "behavioral lineage only and are not personal identity proof."
    ),
    "relationship_evidence_statement": (
        "Generated relationship edges are synthetic evidence and are clearly separated from observed source evidence."
    ),
}
with open("synthnet_run_metadata.json", "w", encoding="utf-8") as metadata_handle:
    json.dump(run_metadata, metadata_handle, indent=2, default=str)
print("All datasets and structural metadata files saved cleanly to disk.")

if ARGS.skip_schema_exports:
    print("Schema-matching exports skipped as requested.")
    raise SystemExit(0)


# ======================================================================
# SCHEMA-MATCHING EXPORTS
# ======================================================================
# These reproduce the exact column layout of a real Indian telecom
# nodal-office CDR export and IPDR export, but every value below is
# generated synthetically by this script (fake names/addresses via
# Faker, fake MSISDN/IMEI/IMSI, fake IPs and cell IDs) -- no real
# subscriber data is read or reproduced. Useful if downstream tooling
# expects files in that exact shape.


def stable_numeric_id(value, digits=14):
    """Create a reproducible numeric identifier without Python's randomized hash()."""
    digest = hashlib.sha256(str(value).encode('utf-8')).hexdigest()
    return f"{int(digest[:16], 16) % (10 ** digits):0{digits}d}"

def export_cdr_real_schema(df_cdr, df_subscribers):
    sub_lookup = df_subscribers.set_index('subscriber_id')
    rows = []
    for i, rec in enumerate(df_cdr.itertuples(index=False), start=1):
        caller = sub_lookup.loc[rec.caller_id]
        ts = rec.timestamp
        call_type = random.choice(['VOICE', 'SMS'])
        stable_cell_id = stable_numeric_id(rec.routing_tower)
        rows.append({
            'SL_NO': i,
            'Mobile_No': caller['msisdn'],
            'Call_Type': call_type,
            'Type_Of_Connec': caller['connection_type'],
            'Other_Party_No': rec.receiver_msisdn,
            'LRN_B_Party_No': '',
            'LRN_DESCRIPTION': '',
            'Call_Date': ts.strftime('%d/%m/%Y'),
            'Call_Initiation_Time(CIT)': ts.strftime('%H:%M:%S'),
            'Call_Duration': rec.duration_seconds if call_type == 'VOICE' else 0,
            'First_Cell_Desc': rec.routing_tower,
            'First_Cell_id': stable_cell_id,
            'Last_Cell_Desc': rec.routing_tower,
            'Last_Cell_ID': stable_cell_id,
            'SMSC_No': f"91{random.randint(7000000000, 9999999999)}",
            'Service_Type': call_type,
            'IMEI': caller['imei'],
            'IMSI': caller['imsi'],
            'Original_Originated_Party': '',
            'Roaming Circle': caller['telecom_circle'],
            'MSC_ID': f"{random.randint(1000000000, 9999999999)}",
            'IN_TG': '',
            'OUT_TG': '',
            'First_LAT': round(random.uniform(8.0, 35.0), 5),
            'First_Long': round(random.uniform(68.0, 97.0), 5),
            'LRN_LSA': caller['telecom_circle'],
        })
    return pd.DataFrame(rows)


def export_ipdr_real_schema(df_ipdr, df_subscribers):
    sub_lookup = df_subscribers.set_index('subscriber_id')
    rows = []
    for rec in df_ipdr.itertuples(index=False):
        sub = sub_lookup.loc[rec.subscriber_id]
        start = rec.timestamp
        end = start + timedelta(minutes=random.randint(1, 45))
        src_ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
        rows.append({
            'Name of Person/Organization': '',
            'Address': '',
            'Contact No.': '',
            'Alternate Contact No.': '',
            'E-mail Address': '',
            'Landline/MSISDN/MDN/Leased Circuit ID for Internet Access': sub['msisdn'],
            'User Id for internet Access based on authentication': rec.subscriber_id,
            'Source IP Address': src_ip,
            'Source Port': random.randint(1024, 65535),
            'Translated IP Address': f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            'Translated Port': random.randint(1024, 65535),
            'Destination IP Address': rec.destination_ip,
            'Destination Port': int(rec.destination_port),
            'Static/Dynamic IP Address Allocation': 'Dynamic',
            'IST Start Time of Public IP address allocation (hh:mm:ss)': start.strftime('%H:%M:%S'),
            'IST End Time of Public IP address allocation (hh:mm:ss)': end.strftime('%H:%M:%S'),
            'Start Date of Public IP Address allocation (dd/mm/yyyy)': start.strftime('%Y/%m/%d'),
            'End Date of Public IP address allocation (dd/mm/yyyy)': end.strftime('%Y/%m/%d'),
            'Source MAC-ID Address/Other device Identification number': sub['imei'],
            'IMSI': sub['imsi'],
            'PGW IP address': f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            'Access Point Name': sub['apn'],
            'First CELL ID': stable_numeric_id(sub['home_tower_id']),
            'Last CELL ID': stable_numeric_id(sub['home_tower_id']),
            'TIME1 (dd/MM/yyyy HH:mm:ss)': start.strftime('%Y/%m/%d %H:%M:%S'),
            'Session Duration (Seconds)': int((end - start).total_seconds()),
            'Data Volume Up Link': int(rec.megabytes_transferred * 1024 * 0.4),
            'Data Volume Down Link': int(rec.megabytes_transferred * 1024 * 0.6),
            'Roaming Circle Indicator': 'HOME',
            'Roaming Circle': sub['telecom_circle'],
            'SIM Type': '',
        })
    return pd.DataFrame(rows)


print("Step 4: Writing schema-matching CDR/IPDR exports for all core agents...")
df_cdr_core = df_cdr[df_cdr['is_core_user']].drop(columns=['is_core_user'])
df_ipdr_core = df_ipdr[df_ipdr['is_core_user']].drop(columns=['is_core_user'])

df_cdr_real_schema = export_cdr_real_schema(df_cdr_core, df_subscribers)
df_ipdr_real_schema = export_ipdr_real_schema(df_ipdr_core, df_subscribers)

cdr_schema_path = write_csv_safely(df_cdr_real_schema, "call_logs_cdr_schema.csv", index=False)
ipdr_schema_path = write_csv_safely(df_ipdr_real_schema, "internet_sessions_ipdr_schema.csv", index=False)

for source_path, clean_path, number_columns in [
    (cdr_schema_path, "call_logs_cdr_schema_clean.csv", FINAL_CDR_NUMBER_COLUMNS),
    (ipdr_schema_path, "internet_sessions_ipdr_schema_clean.csv", FINAL_IPDR_NUMBER_COLUMNS),
]:
    filter_result = filter_cdr_file(
        source_path,
        clean_path,
        letter_columns=number_columns,
        keywords=tuple(split_csv_arg(ARGS.cdr_filter_keywords)),
        strict_columns=False,
    )
    print(
        f"  Cleaned {source_path}: {filter_result['rows_removed']} removed, "
        f"{filter_result['rows_kept']} kept -> {filter_result['output']}"
    )

print(f"  call_logs_cdr_schema.csv: {len(df_cdr_real_schema)} rows (core users only)")
print(f"  internet_sessions_ipdr_schema.csv: {len(df_ipdr_real_schema)} rows (core users only)")
print("All schema-matching exports saved. Every value in them is synthetic.")
