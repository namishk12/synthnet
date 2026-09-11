from __future__ import annotations

import pandas as pd
import pytest

from identity_validation import (
    IdentityMapping,
    apply_identity_answer,
    normalize_identifier,
    resolve_source_schema,
    plan_source_profiles,
    parse_timestamp_series,
    validate_cdr_ipdr,
)
from ipdr_correlation import correlate_ipdr


def _write_sources(tmp_path, *, cdr_rows=None, ipdr_rows=None):
    cdr = tmp_path / "cdr.csv"
    ipdr = tmp_path / "ipdr.csv"
    if cdr_rows is None:
        cdr_rows = [
            {"Mobile_No": "+91 98765 43210", "Other_Party_No": "9000000001", "Call_Date": "01/06/2026", "Call_Initiation_Time(CIT)": "10:00:00"},
            {"Mobile_No": '="9876543211"', "Other_Party_No": "9000000002", "Call_Date": "02/06/2026", "Call_Initiation_Time(CIT)": "11:00:00"},
            {"Mobile_No": "9876543210", "Other_Party_No": "9000000002", "Call_Date": "03/06/2026", "Call_Initiation_Time(CIT)": "12:00:00"},
        ]
    if ipdr_rows is None:
        ipdr_rows = [
            {"subscriber_id": "919876543210", "timestamp": "2026-06-01T10:05:00+05:30", "destination_ip": "10.0.0.1", "destination_port": "443"},
            {"subscriber_id": "919876543211", "timestamp": "2026-06-02T11:05:00+05:30", "destination_ip": "10.0.0.2", "destination_port": "443"},
        ]
    cdr_frame = pd.DataFrame(cdr_rows)
    if cdr_frame.empty and not len(cdr_frame.columns):
        cdr_frame = pd.DataFrame(columns=["Mobile_No", "Other_Party_No", "timestamp"])
    ipdr_frame = pd.DataFrame(ipdr_rows)
    if ipdr_frame.empty and not len(ipdr_frame.columns):
        ipdr_frame = pd.DataFrame(columns=["subscriber_id", "timestamp"])
    cdr_frame.to_csv(cdr, index=False)
    ipdr_frame.to_csv(ipdr, index=False)
    return cdr, ipdr


def test_normalization_handles_excel_whitespace_and_phone_formats_without_touching_generic_ids():
    assert normalize_identifier(" +91 (98765) 43210 ", field_name="Mobile_No") == "919876543210"
    assert normalize_identifier('="9876543210"', field_name="Mobile_No") == "919876543210"
    assert normalize_identifier("9876543210", field_name="caller_id") == "919876543210"
    assert normalize_identifier("9876543210", field_name="receiver_id") == "919876543210"
    assert normalize_identifier("  Source  001 ", field_name="source_identifier") == "source 001"
    assert normalize_identifier("1234567890", field_name="subscriber_id") == "1234567890"


def test_full_chunked_validation_distinguishes_subjects_from_contacts_and_reports_dates(tmp_path):
    cdr, ipdr = _write_sources(tmp_path)
    result = validate_cdr_ipdr(cdr, ipdr, chunk_size=1)
    assert result["all_records_scanned"] is True
    assert result["sources"]["cdr"]["chunk_count"] == 3
    assert result["sources"]["cdr"]["subject_column"] == "Mobile_No"
    assert result["sources"]["ipdr"]["subject_column"] == "subscriber_id"
    assert result["matching"]["cdr_subject_count"] == 2
    assert result["matching"]["ipdr_subscriber_count"] == 2
    assert result["matching"]["matched_count"] == 2
    assert "9000000001" not in result["matching"]["matched_identifiers"]
    assert result["temporal"]["overlaps"] is True
    assert result["identity_evidence"]["identifier_agreement_proves_personal_identity"] is False


def test_same_person_blocks_conflicting_subjects(tmp_path):
    cdr, ipdr = _write_sources(
        tmp_path,
        cdr_rows=[{"Mobile_No": "A", "Other_Party_No": "B", "timestamp": "2026-06-01 10:00:00"}],
        ipdr_rows=[{"subscriber_id": "C", "timestamp": "2026-06-01 10:00:00"}],
    )
    result = validate_cdr_ipdr(cdr, ipdr, chunk_size=1)
    decision = apply_identity_answer(result, "same_person", generation_mode="combined", confirmation=True)
    assert decision["allowed"] is False
    assert decision["confirmation_is_not_an_override"] is True


def test_explicit_alias_mapping_allows_a_consistent_single_subject_mapping(tmp_path):
    cdr, ipdr = _write_sources(
        tmp_path,
        cdr_rows=[{"Mobile_No": "CDR-1", "Other_Party_No": "B", "timestamp": "2026-06-01 10:00:00"}],
        ipdr_rows=[{"subscriber_id": "IPDR-1", "timestamp": "2026-06-01 10:00:00"}],
    )
    result = validate_cdr_ipdr(cdr, ipdr, mapping={"IPDR-1": "CDR-1"})
    assert result["matching"]["matched_count"] == 1
    assert result["mapping"]["mapping_fingerprint"]
    assert apply_identity_answer(result, "same_person", generation_mode="combined")["allowed"] is True


def test_ambiguous_alias_is_quarantined(tmp_path):
    cdr, ipdr = _write_sources(
        tmp_path,
        cdr_rows=[{"Mobile_No": "A", "Other_Party_No": "B", "timestamp": "2026-06-01 10:00:00"}],
        ipdr_rows=[{"subscriber_id": "alias", "timestamp": "2026-06-01 10:00:00"}],
    )
    mapping = IdentityMapping.from_mapping([
        {"alias": "alias", "canonical": "A"},
        {"alias": "alias", "canonical": "B"},
    ])
    result = validate_cdr_ipdr(cdr, ipdr, mapping=mapping)
    assert result["mapping"]["ambiguous_alias_count"] > 0
    assert apply_identity_answer(result, "same_person", generation_mode="combined")["allowed"] is False


def test_multiple_people_reports_partial_coverage(tmp_path):
    cdr, ipdr = _write_sources(
        tmp_path,
        cdr_rows=[
            {"Mobile_No": "A", "Other_Party_No": "B", "timestamp": "2026-06-01 10:00:00"},
            {"Mobile_No": "B", "Other_Party_No": "A", "timestamp": "2026-06-01 11:00:00"},
        ],
        ipdr_rows=[
            {"subscriber_id": "A", "timestamp": "2026-06-01 10:00:00"},
            {"subscriber_id": "C", "timestamp": "2026-06-01 11:00:00"},
        ],
    )
    result = validate_cdr_ipdr(cdr, ipdr)
    decision = apply_identity_answer(result, "multiple_person_dataset", generation_mode="multi_person_conditioned")
    assert decision["allowed"] is True
    assert result["matching"]["unmatched_ipdr_subscribers"] == ["c"]
    assert any("Partial coverage" in warning for warning in decision["warnings"])


def test_empty_sources_and_non_overlapping_dates_are_explicit(tmp_path):
    cdr, ipdr = _write_sources(
        tmp_path,
        cdr_rows=[],
        ipdr_rows=[{"subscriber_id": "A", "timestamp": "2028-06-01 10:00:00"}],
    )
    result = validate_cdr_ipdr(cdr, ipdr)
    assert result["sources"]["cdr"]["row_count"] == 0
    assert result["temporal"]["status"] == "unavailable"
    cdr, ipdr = _write_sources(
        tmp_path,
        cdr_rows=[{"Mobile_No": "A", "Other_Party_No": "B", "timestamp": "2026-01-01 10:00:00"}],
        ipdr_rows=[{"subscriber_id": "A", "timestamp": "2026-06-01 10:00:00"}],
    )
    assert validate_cdr_ipdr(cdr, ipdr)["temporal"]["status"] == "non_overlapping"


def test_missing_time_is_unavailable_and_date_only_is_parsed(tmp_path):
    cdr = tmp_path / "cdr.csv"
    cdr.write_text("caller_id,receiver_id\nA,B\n", encoding="utf-8")
    schema = resolve_source_schema(cdr, "cdr")
    assert schema["timestamp"] is None
    assert parse_timestamp_series(pd.read_csv(cdr, dtype=str), schema).isna().all()

    dated = tmp_path / "dated.csv"
    dated.write_text("caller_id,receiver_id,Call_Date\nA,B,2026-06-01\n", encoding="utf-8")
    dated_schema = resolve_source_schema(dated, "cdr")
    parsed = parse_timestamp_series(pd.read_csv(dated, dtype=str), dated_schema)
    assert parsed.iloc[0].isoformat() == "2026-05-31T18:30:00+00:00"


def test_single_person_template_is_explicit_and_expansion_is_reproducible():
    with pytest.raises(ValueError):
        plan_source_profiles(["one"], 4)
    plan = plan_source_profiles(["one"], 4, generation_mode="single_person_template", seed=7)
    assert len(plan["agent_ids"]) == 4
    assert len(set(plan["agent_ids"])) == 4
    assert plan["all_source_profiles_preserved"] is True
    assert plan == plan_source_profiles(["one"], 4, generation_mode="single_person_template", seed=7)


def test_three_source_people_expand_to_one_hundred_without_dropping_source_ids():
    plan = plan_source_profiles(["p1", "p2", "p3"], 100, seed=42)
    assert len(plan["agent_ids"]) == 100
    assert {"p1", "p2", "p3"}.issubset(plan["agent_ids"])
    assert len(set(plan["agent_ids"])) == 100
    assert len(set(plan["source_profile_for_agent"].values())) == 3


def test_ipdr_only_single_person_has_a_valid_zero_pair_result(tmp_path):
    _, ipdr = _write_sources(
        tmp_path,
        cdr_rows=[],
        ipdr_rows=[{"subscriber_id": "only", "timestamp": "2026-06-01 10:00:00", "destination_ip": "10.0.0.1"}],
    )
    result = correlate_ipdr(ipdr, chunk_size=1)
    assert result["subscriber_count"] == 1
    assert result["pair_count"] == 0
    assert list(result["pairs"].columns) == [
        "person_a", "person_b", "ipdr_correlation_score", "shared_destinations",
        "shared_ports", "shared_time_buckets", "evidence_type",
    ]
