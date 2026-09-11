#!/usr/bin/env python3
"""Local HTTP server for uploading CDR CSVs and running the CGAN pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from identity_validation import (
    IDENTITY_ANSWER_LABELS,
    IDENTITY_ANSWERS,
    IDENTITY_QUESTION,
    IdentityMapping,
    apply_identity_answer,
    load_identity_mapping,
    validate_cdr_ipdr,
)


ROOT = Path(__file__).resolve().parent
GUI_FILE = ROOT / "cgan_gui.html"
GENERATOR = ROOT / "Synthetic_data_cgan.py"
JOBS_ROOT = ROOT / ".cgan_gui" / "jobs"
MAX_UPLOAD_BYTES = 1_000_000_000
MAX_LOG_LINES = 160
SMOKE_MODE = os.environ.get("CGAN_GUI_SMOKE") == "1"

JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()
ACTIVE_JOB_ID: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_boolean(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"{field_name} must be a boolean.")


def public_job(job: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in job.items()
        if key not in {
            "directory",
            "input_path",
            "ipdr_input_path",
            "identity_mapping_path",
            "process",
        }
    }


def persist_job(job: dict[str, object]) -> None:
    directory = Path(job["directory"])
    temp_path = directory / "status.json.tmp"
    final_path = directory / "status.json"
    temp_path.write_text(json.dumps(public_job(job), indent=2), encoding="utf-8")
    os.replace(temp_path, final_path)
    workflow_metadata = {
        "metadata_version": 1,
        "job_id": job.get("id"),
        "source_filename": job.get("source_filename"),
        "ipdr_filename": job.get("ipdr_filename"),
        "identity_answer": job.get("identity_answer"),
        "identity_confirmation": bool(job.get("identity_confirmation", False)),
        "identity_mappings": job.get("identity_mappings"),
        "identity_mapping_summary": (job.get("validation") or {}).get("mapping") if isinstance(job.get("validation"), dict) else None,
        "schema_overrides": job.get("schema_overrides", {}),
        "validation": job.get("validation"),
        "identity_decision": job.get("identity_decision"),
        "generation_mode": job.get("generation_mode"),
        "timezone": job.get("timezone"),
    }
    metadata_temp = directory / "dataset_metadata.json.tmp"
    metadata_final = directory / "dataset_metadata.json"
    metadata_temp.write_text(json.dumps(workflow_metadata, indent=2, default=str), encoding="utf-8")
    os.replace(metadata_temp, metadata_final)


def update_job(job_id: str, **changes: object) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(changes)
        job["updated_at"] = utc_now()
        persist_job(job)


def append_log(job_id: str, line: str) -> None:
    line = line.rstrip()
    if not line:
        return
    with JOBS_LOCK:
        job = JOBS[job_id]
        log = list(job.get("log", []))
        log.append(line)
        job["log"] = log[-MAX_LOG_LINES:]
        job["message"] = line

        progress = int(job.get("progress", 0))
        match = re.search(r"step\s+(\d+)/(\d+)", line)
        if match and int(match.group(2)):
            progress = max(progress, 12 + int(36 * int(match.group(1)) / int(match.group(2))))
        elif "Step 0:" in line:
            progress = max(progress, 12)
        elif "Graph GAN produced" in line:
            progress = max(progress, 50)
        elif "Step 1:" in line:
            progress = max(progress, 55)
        elif "Exported" in line and "friend" in line.lower():
            progress = max(progress, 62)
        elif "Step 2:" in line:
            progress = max(progress, 66)
        elif "Step 3:" in line:
            progress = max(progress, 76)
        elif "SIMULATION PIPELINE COMPLETE" in line:
            progress = max(progress, 82)
        elif "saved cleanly" in line:
            progress = max(progress, 90)
        elif "Schema-matching exports skipped" in line:
            progress = max(progress, 96)
        job["progress"] = progress
        job["updated_at"] = utc_now()
        persist_job(job)


def output_definitions(clean: bool) -> list[tuple[str, str, str]]:
    suffix = "_clean" if clean else ""
    return [
        ("friends", "CGAN friendship edges", f"conditional_gan_friend_edges{suffix}.csv"),
        ("inferred", "GAN inferred friend edges", f"gan_inferred_friend_edges{suffix}.csv"),
        (
            "truth",
            "Positive and negative friendship truth pairs",
            f"friendship_truth_pairs{suffix}.csv",
        ),
        ("subscribers", "Subscribers", f"subscribers{suffix}.csv"),
        ("calls", "Call logs", f"call_logs{suffix}.csv"),
        ("sessions", "Internet sessions", f"internet_sessions{suffix}.csv"),
        ("metadata", "SynthNet run metadata", "synthnet_run_metadata.json"),
    ]


def _identity_mapping_for_job(job: dict[str, object]) -> IdentityMapping:
    mapping_path = job.get("identity_mapping_path")
    inline_mapping = job.get("identity_mappings")
    if mapping_path and str(mapping_path).lower().endswith(".json"):
        try:
            inline_from_file = json.loads(Path(str(mapping_path)).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read identity mapping JSON: {mapping_path}") from error
        inline_mapping = inline_from_file
        mapping_path = None
    return load_identity_mapping(
        str(mapping_path) if mapping_path else None,
        inline_mapping,
    )


def validate_job_inputs(job: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Revalidate the exact current files and schema/mapping choices."""
    cdr_path = Path(str(job["input_path"]))
    ipdr_path = Path(str(job["ipdr_input_path"])) if job.get("ipdr_input_path") else None
    mapping = _identity_mapping_for_job(job)
    validation = validate_cdr_ipdr(
        cdr_path,
        ipdr_path,
        mapping=mapping,
        overrides=job.get("schema_overrides") if isinstance(job.get("schema_overrides"), dict) else None,
        chunk_size=int(job.get("validation_chunk_size", 100_000)),
        default_timezone=str(job.get("timezone", "Asia/Kolkata")),
    )
    decision = apply_identity_answer(
        validation,
        job.get("identity_answer"),
        generation_mode=job.get("generation_mode"),
        confirmation=bool(job.get("identity_confirmation", False)),
    )
    return validation, decision


def save_identity_mapping_json(job: dict[str, object]) -> None:
    mappings = job.get("identity_mappings")
    if mappings is None:
        job["identity_mapping_path"] = None
        return
    path = Path(job["directory"]) / "identity_mapping.json"
    path.write_text(json.dumps(mappings, indent=2), encoding="utf-8")
    job["identity_mapping_path"] = str(path)


def update_identity_state(job_id: str, *, status: str | None = None, message: str | None = None) -> tuple[dict[str, object], dict[str, object]]:
    with JOBS_LOCK:
        job = JOBS[job_id]
    validation, decision = validate_job_inputs(job)
    changes: dict[str, object] = {
        "validation": validation,
        "identity_decision": decision,
        "identity_question": IDENTITY_QUESTION if validation["sources"]["cdr"]["present"] and validation["sources"]["ipdr"]["present"] else None,
        "identity_options": [
            {"value": value, "label": IDENTITY_ANSWER_LABELS[value]}
            for value in IDENTITY_ANSWERS
        ],
    }
    if status is not None:
        changes["status"] = status
    if message is not None:
        changes["message"] = message
    update_job(job_id, **changes)
    return validation, decision


def generator_mode_for_job(job: dict[str, object]) -> str:
    mode = str(job.get("generation_mode") or "cdr_only")
    # The local CGAN executable generates from a CDR graph. Separate-source
    # decisions are therefore represented by a CDR-only generation run while
    # the supplied IPDR remains stored and available for its own correlation.
    if mode == "separate_sources":
        return "cdr_only"
    return mode


def run_job(job_id: str) -> None:
    global ACTIVE_JOB_ID
    with JOBS_LOCK:
        job = JOBS[job_id]
        directory = Path(job["directory"])
        input_path = Path(job["input_path"])
        ipdr_input_path = Path(job["ipdr_input_path"]) if job.get("ipdr_input_path") else None
        clean = bool(job["clean"])
        identity_answer = job.get("identity_answer")
        generation_mode = generator_mode_for_job(job)
        identity_mapping_path = Path(job["identity_mapping_path"]) if job.get("identity_mapping_path") else None

    # The files can change on disk after upload (or while a caller is editing
    # a mapping). Re-scan before launching the subprocess and never let a stale
    # approval authorize a different byte-level input.
    awaiting_status = "awaiting_identity_answer" if ipdr_input_path is not None else "awaiting_generation_mode"
    try:
        fresh_validation, fresh_decision = validate_job_inputs(job)
    except Exception as error:
        append_log(job_id, f"Revalidation required: {error}")
        update_job(
            job_id,
            status=awaiting_status,
            message="Revalidation failed; correct the input/schema/mapping and review the evidence again.",
        )
        return
    stored_validation = job.get("validation") if isinstance(job.get("validation"), dict) else {}
    if stored_validation.get("validation_fingerprint") != fresh_validation.get("validation_fingerprint"):
        update_job(
            job_id,
            status=awaiting_status,
            validation=fresh_validation,
            identity_decision=fresh_decision,
            message="Input files or identity/schema mappings changed; review the new validation evidence before running.",
        )
        return
    if not fresh_decision.get("allowed"):
        update_job(
            job_id,
            status=awaiting_status,
            validation=fresh_validation,
            identity_decision=fresh_decision,
            message="Identity processing is blocked; correct the files or mapping before running.",
        )
        return
    if str(fresh_decision.get("generation_mode") or "") in {"review_required", "ipdr_only_correlation"}:
        update_job(
            job_id,
            status="awaiting_generation_mode",
            validation=fresh_validation,
            identity_decision=fresh_decision,
            message="Choose an executable generation or correlation operation before running.",
        )
        return
    use_ipdr_for_generation = ipdr_input_path is not None and str(fresh_decision.get("generation_mode")) in {
        "combined",
        "multi_person_conditioned",
        "single_person_template",
    }

    command = [
        sys.executable,
        "-u",
        str(GENERATOR),
        "--cdr-input",
        str(input_path),
        "--synthetic-users",
        "1000",
        "--core-users",
        "1000",
        "--interactions-per-user",
        "1000",
        "--sessions-per-user",
        "1000",
        "--output-mode",
        "clean" if clean else "normal",
        "--skip-schema-exports",
        "--generation-mode",
        generation_mode,
        "--chunk-size",
        str(job.get("validation_chunk_size", 100_000)),
        "--timezone",
        str(job.get("timezone", "Asia/Kolkata")),
    ]
    if not clean:
        command.append("--skip-cdr-filter")
    if use_ipdr_for_generation:
        command.extend(["--ipdr-input", str(ipdr_input_path)])
    if identity_answer:
        command.extend(["--identity-answer", str(identity_answer)])
    if identity_mapping_path is not None:
        command.extend(["--identity-map-json", str(identity_mapping_path)])
    if SMOKE_MODE:
        command.extend([
            "--interactions-per-user", "2",
            "--sessions-per-user", "2",
            "--gan-steps", "1",
            "--gan-batch-size", "32",
            "--candidate-sample-size", "1000",
            "--gan-inferred-top-fraction", "0.01",
            "--simulation-days", "1",
            "--device", "cpu",
        ])

    update_job(
        job_id,
        status="running",
        progress=8,
        message="Starting the CGAN pipeline",
        started_at=utc_now(),
    )

    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        with JOBS_LOCK:
            JOBS[job_id]["process"] = process
        assert process.stdout is not None
        for line in process.stdout:
            append_log(job_id, line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"CGAN process exited with code {return_code}.")

        outputs = []
        for key, label, filename in output_definitions(clean):
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Expected output was not created: {filename}")
            outputs.append(
                {
                    "key": key,
                    "label": label,
                    "filename": filename,
                    "size": path.stat().st_size,
                }
            )

        zip_path = directory / ("cgan_clean_outputs.zip" if clean else "cgan_outputs.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for output in outputs:
                archive.write(directory / str(output["filename"]), arcname=str(output["filename"]))

        update_job(
            job_id,
            status="complete",
            progress=100,
            message="All output files, including positive and negative truth pairs, are ready",
            completed_at=utc_now(),
            outputs=outputs,
            zip_filename=zip_path.name,
            zip_size=zip_path.stat().st_size,
        )
    except Exception as error:
        append_log(job_id, f"ERROR: {error}")
        update_job(
            job_id,
            status="error",
            message=str(error),
            completed_at=utc_now(),
        )
    finally:
        with JOBS_LOCK:
            if ACTIVE_JOB_ID == job_id:
                ACTIVE_JOB_ID = None


class CGANRequestHandler(BaseHTTPRequestHandler):
    server_version = "CGANGui/1.0"

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path, *, download_name: str | None = None) -> None:
        if not path.is_file():
            self.send_json({"error": "File not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("X-Content-Type-Options", "nosniff")
        if download_name:
            safe_name = download_name.replace('"', "")
            self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
        self.end_headers()
        with path.open("rb") as source:
            shutil.copyfileobj(source, self.wfile, length=1024 * 1024)

    def receive_csv_body(self, target_path: Path) -> int:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            raise ValueError("The uploaded CSV is empty.")
        if content_length > MAX_UPLOAD_BYTES:
            raise OverflowError("CSV exceeds the 1 GB local upload limit.")

        remaining = content_length
        with target_path.open("wb") as target:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ConnectionError("Upload ended before the complete CSV was received.")
                target.write(chunk)
                remaining -= len(chunk)
        with target_path.open("r", encoding="utf-8-sig", errors="replace") as source:
            sample = source.read(65536)
        if not sample.strip() or "," not in sample:
            raise ValueError("The uploaded file does not look like a comma-separated CSV.")
        return content_length

    def receive_json_body(self) -> dict[str, object]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            raise ValueError("The JSON request body is empty.")
        if content_length > 10_000_000:
            raise OverflowError("JSON request exceeds the 10 MB local limit.")
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("The request body must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("The request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/cgan_gui.html"}:
            self.send_file(GUI_FILE)
            return
        if path == "/api/health":
            with JOBS_LOCK:
                active = ACTIVE_JOB_ID
            self.send_json({"ok": True, "active_job_id": active})
            return

        match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})", path)
        if match:
            job_id = match.group(1)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                payload = public_job(job) if job else None
            if payload is None:
                self.send_json({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(payload)
            return

        match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/validation", path)
        if match:
            job_id = match.group(1)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                payload = {
                    "validation": job.get("validation") if job else None,
                    "identity_decision": job.get("identity_decision") if job else None,
                    "identity_question": job.get("identity_question") if job else None,
                    "identity_options": job.get("identity_options") if job else [],
                }
            if not job:
                self.send_json({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(payload)
            return

        match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/files/([a-z]+)", path)
        if match:
            job_id, key = match.groups()
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                outputs = list(job.get("outputs", [])) if job else []
            output = next((item for item in outputs if item["key"] == key), None)
            if not job or not output:
                self.send_json({"error": "Output not found"}, HTTPStatus.NOT_FOUND)
                return
            file_path = Path(job["directory"]) / str(output["filename"])
            self.send_file(file_path, download_name=str(output["filename"]))
            return

        match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/download-all", path)
        if match:
            job_id = match.group(1)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job or job.get("status") != "complete":
                self.send_json({"error": "Output archive is not ready"}, HTTPStatus.NOT_FOUND)
                return
            zip_path = Path(job["directory"]) / str(job["zip_filename"])
            self.send_file(zip_path, download_name=zip_path.name)
            return

        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        global ACTIVE_JOB_ID
        parsed = urlparse(self.path)

        identity_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/identity", parsed.path)
        if identity_match:
            job_id = identity_match.group(1)
            try:
                payload = self.receive_json_body()
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    if not job:
                        self.send_json({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
                        return
                    if job.get("status") == "awaiting_ipdr":
                        self.send_json({"error": "Upload the IPDR file before submitting the identity decision."}, HTTPStatus.CONFLICT)
                        return
                    if job.get("status") not in {"awaiting_identity_answer", "awaiting_generation_mode"}:
                        self.send_json({"error": "This job is not waiting for an identity decision."}, HTTPStatus.CONFLICT)
                        return
                    if "answer" in payload or "identity_answer" in payload:
                        job["identity_answer"] = payload.get("answer", payload.get("identity_answer"))
                    if "generation_mode" in payload:
                        job["generation_mode"] = payload.get("generation_mode")
                    if "confirmation" in payload or "identity_confirmation" in payload:
                        job["identity_confirmation"] = parse_boolean(
                            payload.get("confirmation", payload.get("identity_confirmation")),
                            field_name="identity_confirmation",
                        )
                    if "mappings" in payload or "identity_mapping" in payload:
                        mappings = payload.get("mappings", payload.get("identity_mapping"))
                        if mappings is not None and not isinstance(mappings, (dict, list)):
                            raise ValueError("Identity mappings must be a JSON object, a list of mapping records, or null.")
                        job["identity_mappings"] = mappings
                    if "schema_overrides" in payload:
                        job["schema_overrides"] = payload["schema_overrides"]
                    save_identity_mapping_json(job)
                validation, decision = update_identity_state(job_id)
                if not decision.get("allowed"):
                    update_job(
                        job_id,
                        status="awaiting_identity_answer",
                        message="Identity evidence needs correction or an explicit mapping before this operation can run.",
                    )
                    with JOBS_LOCK:
                        response_job = public_job(JOBS[job_id])
                    self.send_json(response_job, HTTPStatus.CONFLICT)
                    return
                mode = str(decision.get("generation_mode") or "")
                if mode in {"review_required", "ipdr_only_correlation"}:
                    update_job(
                        job_id,
                        status="awaiting_generation_mode",
                        message="Choose an executable operation from the available operations shown by the identity check.",
                    )
                    with JOBS_LOCK:
                        response_job = public_job(JOBS[job_id])
                    self.send_json(response_job, HTTPStatus.CONFLICT)
                    return
                update_job(
                    job_id,
                    status="queued",
                    progress=4,
                    message="Identity decision accepted; preparing the model",
                    identity_answer=decision.get("answer"),
                    generation_mode=mode,
                )
                thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
                thread.start()
                with JOBS_LOCK:
                    response_job = public_job(JOBS[job_id])
                self.send_json(response_job, HTTPStatus.ACCEPTED)
            except Exception as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return

        schema_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/schema", parsed.path)
        if schema_match:
            job_id = schema_match.group(1)
            try:
                payload = self.receive_json_body()
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    if not job:
                        self.send_json({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
                        return
                    if job.get("status") in {"running", "queued", "complete"}:
                        self.send_json({"error": "Schema cannot change after generation has started."}, HTTPStatus.CONFLICT)
                        return
                    overrides = payload.get("overrides", payload.get("schema_overrides", payload))
                    if not isinstance(overrides, dict):
                        raise ValueError("Schema overrides must be a JSON object.")
                    job["schema_overrides"] = overrides
                    # A schema edit invalidates the prior identity decision;
                    # the user must review the newly scanned evidence.
                    job["identity_answer"] = None
                    job["identity_confirmation"] = False
                    job["generation_mode"] = None
                validation, decision = update_identity_state(job_id)
                has_both_sources = bool(
                    validation["sources"]["cdr"]["present"]
                    and validation["sources"]["ipdr"]["present"]
                )
                update_job(
                    job_id,
                    status="awaiting_identity_answer" if has_both_sources else "awaiting_generation_mode",
                    message="Schema changed; identity evidence was revalidated and needs review.",
                )
                with JOBS_LOCK:
                    response_job = public_job(JOBS[job_id])
                self.send_json(response_job)
            except Exception as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return

        ipdr_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/ipdr", parsed.path)
        if ipdr_match:
            job_id = ipdr_match.group(1)
            query = parse_qs(parsed.query)
            filename = unquote(query.get("filename", [""])[0]).strip()
            if not filename.lower().endswith(".csv"):
                self.send_json({"error": "Choose a .csv IPDR file."}, HTTPStatus.BAD_REQUEST)
                return
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                valid = bool(job and job.get("status") == "awaiting_ipdr")
            if not valid:
                self.send_json({"error": "This job is not waiting for an IPDR upload."}, HTTPStatus.CONFLICT)
                return
            directory = Path(job["directory"])
            ipdr_path = directory / "ipdr_input.csv"
            try:
                ipdr_size = self.receive_csv_body(ipdr_path)
                with JOBS_LOCK:
                    job["ipdr_input_path"] = str(ipdr_path)
                    job["ipdr_filename"] = Path(filename).name
                    job["ipdr_size"] = ipdr_size
                validation, decision = update_identity_state(
                    job_id,
                    status="awaiting_identity_answer",
                    message="Both CSV files uploaded; review the identity evidence before combining activity.",
                )
                with JOBS_LOCK:
                    response_job = public_job(JOBS[job_id])
                self.send_json(response_job, HTTPStatus.ACCEPTED)
            except Exception as error:
                with JOBS_LOCK:
                    if ACTIVE_JOB_ID == job_id:
                        ACTIVE_JOB_ID = None
                update_job(job_id, status="error", message=str(error), completed_at=utc_now())
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return

        if parsed.path != "/api/jobs":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        query = parse_qs(parsed.query)
        filename = unquote(query.get("filename", [""])[0]).strip()
        clean = query.get("clean", ["false"])[0].lower() == "true"
        has_ipdr = query.get("has_ipdr", ["false"])[0].lower() == "true"
        if not filename.lower().endswith(".csv"):
            self.send_json({"error": "Choose a .csv file."}, HTTPStatus.BAD_REQUEST)
            return

        with JOBS_LOCK:
            if ACTIVE_JOB_ID is not None:
                self.send_json(
                    {"error": "Another CGAN job is already running. Wait for it to finish."},
                    HTTPStatus.CONFLICT,
                )
                return
            job_id = uuid.uuid4().hex
            ACTIVE_JOB_ID = job_id

        directory = JOBS_ROOT / job_id
        directory.mkdir(parents=True, exist_ok=False)
        input_path = directory / "input.csv"
        try:
            content_length = self.receive_csv_body(input_path)

            job = {
                "id": job_id,
                "status": "awaiting_ipdr" if has_ipdr else "queued",
                "progress": 4,
                "message": "CDR uploaded; waiting for IPDR" if has_ipdr else "CSV uploaded; preparing the model",
                "clean": clean,
                "has_ipdr": has_ipdr,
                "source_filename": Path(filename).name,
                "source_size": content_length,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "log": [],
                "outputs": [],
                "directory": str(directory),
                "input_path": str(input_path),
                "ipdr_input_path": None,
                "identity_question": None,
                "identity_options": [],
                "identity_answer": None,
                "identity_confirmation": False,
                "identity_mappings": None,
                "schema_overrides": {},
                "generation_mode": "cdr_only" if not has_ipdr else None,
                "timezone": "Asia/Kolkata",
                "validation_chunk_size": 100_000,
                "validation": None,
                "identity_decision": None,
            }
            with JOBS_LOCK:
                JOBS[job_id] = job
                persist_job(job)
            # Even CDR-only uploads are checked and recorded before the
            # generator starts. Both-source uploads stop here until the
            # explicit identity answer is submitted.
            validation, decision = update_identity_state(
                job_id,
                status="awaiting_ipdr" if has_ipdr else "queued",
                message="CDR uploaded; waiting for IPDR" if has_ipdr else "CSV validated; preparing the model",
            )
            if not has_ipdr and not decision.get("allowed"):
                raise ValueError("CDR input validation blocked generation: " + " ".join(decision.get("blocking_reasons", [])))
            if not has_ipdr:
                single_source = int(validation["sources"]["cdr"].get("subject_count", 0)) == 1
                if single_source:
                    update_job(
                        job_id,
                        status="awaiting_generation_mode",
                        generation_mode=None,
                        message="One CDR subject was detected; choose CDR-only generation or explicit single-person template expansion.",
                    )
                else:
                    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
                    thread.start()
            self.send_json(public_job(job), HTTPStatus.ACCEPTED)
        except Exception as error:
            with JOBS_LOCK:
                if ACTIVE_JOB_ID == job_id:
                    ACTIVE_JOB_ID = None
                JOBS.pop(job_id, None)
            shutil.rmtree(directory, ignore_errors=True)
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Telecom CGAN HTML GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    if not GUI_FILE.is_file() or not GENERATOR.is_file():
        raise FileNotFoundError("The GUI and generator must be beside cgan_gui_server.py.")
    server = ThreadingHTTPServer((args.host, args.port), CGANRequestHandler)
    print(f"Telecom CGAN Studio: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
