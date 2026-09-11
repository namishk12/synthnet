#!/usr/bin/env python3
"""Build browser-friendly per-agent JSON from the synthetic telecom exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


CIRCLE_CENTERS = {
    "GJ": (22.2587, 71.1924),
    "DL": (28.6139, 77.2090),
    "KA": (15.3173, 75.7139),
    "MH": (19.7515, 75.7139),
    "RJ": (27.0238, 74.2179),
    "UP": (26.8467, 80.9462),
    "WB": (22.9868, 87.8550),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        yield from csv.DictReader(handle)


def source_file(directory: Path, *names: str) -> Path:
    """Return the first available normal/clean generator artifact."""
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"None of the required files were found in {directory}: {', '.join(names)}"
    )


def number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def valid_coordinate(lat: float, lng: float) -> bool:
    return 6.0 <= lat <= 38.5 and 67.0 <= lng <= 98.5


def latest(*values: str) -> str:
    usable = [value for value in values if value]
    return max(usable) if usable else ""


def compact_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def tabular_rows(rows: list[dict[str, str]], columns: list[str]) -> list[list[str]]:
    """Preserve every CSV field without repeating column names on every event."""
    return [[row.get(column, "") for column in columns] for row in rows]


def main() -> int:
    args = parse_args()
    source = Path(args.input_dir).resolve()
    output = Path(args.output_dir).resolve()
    agent_dir = output / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)

    subscribers_path = source_file(source, "subscribers_clean.csv", "subscribers.csv")
    calls_path = source_file(source, "call_logs_clean.csv", "call_logs.csv")
    sessions_path = source_file(
        source,
        "internet_sessions_clean.csv",
        "internet_sessions.csv",
    )
    friend_edges_path = source_file(
        source,
        "conditional_gan_friend_edges_clean.csv",
        "conditional_gan_friend_edges.csv",
    )

    subscribers = list(read_rows(subscribers_path))
    by_id = {row["subscriber_id"]: row for row in subscribers}
    by_msisdn = {row["msisdn"]: row for row in subscribers}

    outgoing_calls: dict[str, list[dict[str, str]]] = defaultdict(list)
    incoming_calls: dict[str, list[dict[str, str]]] = defaultdict(list)
    sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
    contacts: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    locations: dict[str, dict[tuple[float, float, str], dict[str, Any]]] = defaultdict(dict)
    call_columns: list[str] = []
    session_columns: list[str] = []

    def contact_record(
        agent_id: str,
        other_msisdn: str,
        other_name: str,
        contact_type: str,
    ) -> dict[str, Any]:
        existing = contacts[agent_id].get(other_msisdn)
        if existing is not None:
            return existing
        other = by_msisdn.get(other_msisdn)
        record = {
            "msisdn": other_msisdn,
            "name": other["fake_full_name"] if other else other_name,
            "subscriberId": other["subscriber_id"] if other else "",
            "contactType": "core_agent" if other else (contact_type or "external_contact"),
            "outgoingCount": 0,
            "incomingCount": 0,
            "durationSeconds": 0,
            "lastSeen": "",
            "lastCallType": "",
            "isKnownFriend": contact_type == "known_friend",
            "friendScore": None,
            "friendSource": "",
        }
        contacts[agent_id][other_msisdn] = record
        return record

    call_count = 0
    for row in read_rows(calls_path):
        if not call_columns:
            call_columns = list(row.keys())
        call_count += 1
        caller_id = row.get("caller_id", "")
        if caller_id not in by_id:
            continue
        timestamp = row.get("timestamp", "")
        duration = int(round(number(row.get("duration_seconds"))))
        outgoing_calls[caller_id].append(row)

        receiver_msisdn = row.get("receiver_msisdn", "")
        outgoing_contact = contact_record(
            caller_id,
            receiver_msisdn,
            row.get("receiver_name", ""),
            row.get("contact_type", ""),
        )
        outgoing_contact["outgoingCount"] += 1
        outgoing_contact["durationSeconds"] += duration
        outgoing_contact["lastSeen"] = latest(outgoing_contact["lastSeen"], timestamp)
        outgoing_contact["lastCallType"] = row.get("call_type", "")
        outgoing_contact["isKnownFriend"] = (
            outgoing_contact["isKnownFriend"] or row.get("contact_type") == "known_friend"
        )

        receiver = by_msisdn.get(receiver_msisdn)
        if receiver:
            receiver_id = receiver["subscriber_id"]
            incoming_calls[receiver_id].append(row)
            incoming_contact = contact_record(
                receiver_id,
                row.get("caller_msisdn", ""),
                row.get("caller_name", ""),
                "known_friend",
            )
            incoming_contact["incomingCount"] += 1
            incoming_contact["durationSeconds"] += duration
            incoming_contact["lastSeen"] = latest(incoming_contact["lastSeen"], timestamp)
            incoming_contact["lastCallType"] = row.get("call_type", "")
            incoming_contact["isKnownFriend"] = True

        lat = number(row.get("first_lat"), default=float("nan"))
        lng = number(row.get("first_long"), default=float("nan"))
        if valid_coordinate(lat, lng):
            tower = (
                row.get("first_cell_desc")
                or row.get("routing_tower")
                or row.get("first_cell_id")
                or "Unknown tower"
            )
            key = (round(lat, 5), round(lng, 5), tower)
            point = locations[caller_id].setdefault(
                key,
                {
                    "lat": round(lat, 5),
                    "lng": round(lng, 5),
                    "tower": tower,
                    "count": 0,
                    "lastSeen": "",
                },
            )
            point["count"] += 1
            point["lastSeen"] = latest(point["lastSeen"], timestamp)

    session_count = 0
    for row in read_rows(sessions_path):
        if not session_columns:
            session_columns = list(row.keys())
        session_count += 1
        subscriber_id = row.get("subscriber_id", "")
        if subscriber_id in by_id:
            sessions[subscriber_id].append(row)

    friend_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    condition_fields = (
        "ipdr_condition_domain_similarity",
        "ipdr_condition_application_similarity",
        "ipdr_condition_port_similarity",
        "ipdr_condition_temporal_similarity",
        "ipdr_condition_session_similarity",
        "ipdr_condition_behaviour_similarity",
        "ipdr_condition_rare_event_bonus",
        "ipdr_condition_dbscan_same_cluster",
        "ipdr_condition_spectral_same_cluster",
        "ipdr_condition_isolation_forest_score",
        "ipdr_condition_isolation_forest_anomaly",
        "ipdr_condition_cdr_ipdr_temporal_coincidences",
    )
    for row in read_rows(friend_edges_path):
        left_id = row.get("subscriber_id_u", "")
        right_id = row.get("subscriber_id_v", "")
        if left_id not in by_id or right_id not in by_id:
            continue
        score = number(row.get("gan_friend_score"), default=float("nan"))
        score_value = round(score, 4) if math.isfinite(score) else None
        ipdr_score = number(row.get("ipdr_correlation_score"), default=float("nan"))
        ipdr_score_value = round(ipdr_score, 4) if math.isfinite(ipdr_score) else None
        evidence = {
            field.removeprefix("ipdr_condition_"): row.get(field, "")
            for field in condition_fields
            if row.get(field, "") != ""
        }
        for agent_id, other_id in ((left_id, right_id), (right_id, left_id)):
            other = by_id[other_id]
            edge = {
                "subscriberId": other_id,
                "name": other["fake_full_name"],
                "msisdn": other["msisdn"],
                "source": row.get("edge_source", ""),
                "score": score_value,
                "ipdrCorrelationScore": ipdr_score_value,
                "observed": row.get("is_observed_training_edge", "").lower() == "true",
                "ganInferred": row.get("is_gan_inferred_friend", "").lower() == "true",
                "sameState": row.get("same_state", "").lower() == "true",
                "truthSource": row.get("truth_source", ""),
                "evidence": evidence,
            }
            friend_edges[agent_id].append(edge)
            contact = contacts[agent_id].get(other["msisdn"])
            if contact:
                contact["friendScore"] = score_value
                contact["friendSource"] = edge["source"]
                contact["isKnownFriend"] = True

    agent_index: list[dict[str, Any]] = []
    global_hourly_calls = [0] * 24
    global_hourly_sessions = [0] * 24

    for subscriber in subscribers:
        agent_id = subscriber["subscriber_id"]
        outgoing = outgoing_calls.get(agent_id, [])
        incoming = incoming_calls.get(agent_id, [])
        agent_sessions = sessions.get(agent_id, [])
        agent_contacts = list(contacts.get(agent_id, {}).values())
        location_points = list(locations.get(agent_id, {}).values())

        if location_points:
            weighted_lats = [
                point["lat"]
                for point in location_points
                for _ in range(max(1, int(point["count"])))
            ]
            weighted_lngs = [
                point["lng"]
                for point in location_points
                for _ in range(max(1, int(point["count"])))
            ]
            map_lat, map_lng = median(weighted_lats), median(weighted_lngs)
        else:
            map_lat, map_lng = CIRCLE_CENTERS.get(
                subscriber.get("telecom_circle", ""),
                (22.5937, 78.9629),
            )

        hourly_calls = [0] * 24
        hourly_sessions = [0] * 24
        call_types: Counter[str] = Counter()
        total_call_seconds = 0
        for row in outgoing + incoming:
            timestamp = row.get("timestamp", "")
            try:
                hour = int(timestamp[11:13])
            except (TypeError, ValueError):
                hour = -1
            if 0 <= hour <= 23:
                hourly_calls[hour] += 1
            call_types[row.get("call_type") or "Unknown"] += 1
            total_call_seconds += int(round(number(row.get("duration_seconds"))))

        destination_ports: Counter[str] = Counter()
        destination_ips: Counter[str] = Counter()
        apns: Counter[str] = Counter()
        total_data_mb = 0.0
        total_upload_kb = 0.0
        total_download_kb = 0.0
        for row in agent_sessions:
            timestamp = row.get("timestamp", "")
            try:
                hour = int(timestamp[11:13])
            except (TypeError, ValueError):
                hour = -1
            if 0 <= hour <= 23:
                hourly_sessions[hour] += 1
            destination_ports[row.get("destination_port") or "Unknown"] += 1
            destination_ips[row.get("destination_ip") or "Unknown"] += 1
            apns[row.get("apn") or "Unknown"] += 1
            total_data_mb += number(row.get("megabytes_transferred"))
            total_upload_kb += number(row.get("data_volume_up_kb"))
            total_download_kb += number(row.get("data_volume_down_kb"))

        for hour in range(24):
            global_hourly_calls[hour] += hourly_calls[hour]
            global_hourly_sessions[hour] += hourly_sessions[hour]

        agent_contacts.sort(
            key=lambda item: (
                -(item["incomingCount"] + item["outgoingCount"]),
                item["name"],
            )
        )
        location_points.sort(key=lambda item: (-item["count"], item["tower"]))
        outgoing.sort(key=lambda row: row.get("timestamp", ""), reverse=True)
        incoming.sort(key=lambda row: row.get("timestamp", ""), reverse=True)
        agent_sessions.sort(key=lambda row: row.get("timestamp", ""), reverse=True)
        friend_edges[agent_id].sort(
            key=lambda item: (
                -(item["score"] if item["score"] is not None else -1),
                item["name"],
            )
        )

        latest_call = latest(
            outgoing[0].get("timestamp", "") if outgoing else "",
            incoming[0].get("timestamp", "") if incoming else "",
        )
        latest_session = agent_sessions[0].get("timestamp", "") if agent_sessions else ""
        latest_activity = latest(latest_call, latest_session)

        known_friend_count = sum(1 for item in agent_contacts if item["isKnownFriend"])
        core_contact_count = sum(1 for item in agent_contacts if item["subscriberId"])
        stats = {
            "outgoingCalls": len(outgoing),
            "incomingCalls": len(incoming),
            "totalCallSeconds": total_call_seconds,
            "uniqueContacts": len(agent_contacts),
            "knownFriends": known_friend_count,
            "coreContacts": core_contact_count,
            "sessions": len(agent_sessions),
            "totalDataMb": round(total_data_mb, 3),
            "totalUploadKb": round(total_upload_kb, 3),
            "totalDownloadKb": round(total_download_kb, 3),
            "friendEdges": len(friend_edges[agent_id]),
            "ganFriendEdges": sum(1 for edge in friend_edges[agent_id] if edge["ganInferred"]),
            "locationPoints": len(location_points),
            "latestActivity": latest_activity,
        }

        index_record = {
            "id": agent_id,
            "name": subscriber.get("fake_full_name", ""),
            "msisdn": subscriber.get("msisdn", ""),
            "circle": subscriber.get("telecom_circle", ""),
            "tower": subscriber.get("home_tower_id", ""),
            "device": subscriber.get("device_capability", ""),
            "connection": subscriber.get("connection_type", ""),
            "simType": subscriber.get("sim_type", ""),
            "location": {"lat": round(map_lat, 5), "lng": round(map_lng, 5)},
            "stats": stats,
        }
        agent_index.append(index_record)

        detail = {
            "profile": subscriber,
            "index": index_record,
            "stats": stats,
            "contacts": agent_contacts,
            "friendEdges": friend_edges[agent_id],
            "schemas": {
                "callColumns": call_columns,
                "sessionColumns": session_columns,
            },
            "calls": {
                "outgoing": tabular_rows(outgoing, call_columns),
                "incoming": tabular_rows(incoming, call_columns),
            },
            "sessions": tabular_rows(agent_sessions, session_columns),
            "locations": location_points,
            "charts": {
                "hourlyCalls": hourly_calls,
                "hourlySessions": hourly_sessions,
                "callTypes": [
                    {"label": label, "value": value}
                    for label, value in call_types.most_common()
                ],
                "destinationPorts": [
                    {"label": label, "value": value}
                    for label, value in destination_ports.most_common(10)
                ],
                "destinationIps": [
                    {"label": label, "value": value}
                    for label, value in destination_ips.most_common(10)
                ],
                "apns": [
                    {"label": label, "value": value}
                    for label, value in apns.most_common(10)
                ],
            },
        }
        compact_write(agent_dir / f"{agent_id}.json", detail)

    agent_index.sort(key=lambda item: item["id"])
    totals = {
        "agents": len(agent_index),
        "calls": call_count,
        "sessions": session_count,
        "friendEdges": sum(len(edges) for edges in friend_edges.values()) // 2,
        "incomingCoreCalls": sum(len(rows) for rows in incoming_calls.values()),
        "totalDataMb": round(
            sum(item["stats"]["totalDataMb"] for item in agent_index),
            3,
        ),
    }
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataset": "Synthetic Telecom CGAN — 100 agent rerun",
        "totals": totals,
        "hourlyCalls": global_hourly_calls,
        "hourlySessions": global_hourly_sessions,
        "agents": agent_index,
    }
    compact_write(output / "agent-index.json", payload)
    print(
        f"Built {len(agent_index)} agent profiles from "
        f"{call_count:,} calls and {session_count:,} sessions -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
