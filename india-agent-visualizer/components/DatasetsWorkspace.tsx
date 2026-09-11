"use client";

import {
  AlertTriangle,
  Database,
  FileSpreadsheet,
  History,
  RotateCcw,
  Search,
  ShieldCheck,
  Trash2,
  Upload,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  formatDateTime,
  formatIndianNumber,
  nexusFetch,
  type IdentityDecision,
  type IdentityValidation,
  type DatasetRecord,
  type RunRecord,
} from "../lib/nexus-api";
import { NexusShell } from "./NexusShell";
import { RunMonitor } from "./RunMonitor";

type Preview = {
  columns: string[];
  resolved: Record<string, string | null>;
  preview: Array<Record<string, string>>;
  quality_warnings: string[];
  identity_validation?: IdentityValidation | null;
  identity_decision?: IdentityDecision | null;
};

type RetentionSettings = {
  retention_days: number;
  auto_purge: boolean;
};

function IdentityEvidence({
  validation,
  decision,
}: {
  validation: IdentityValidation;
  decision?: IdentityDecision | null;
}) {
  const cdr = validation.sources?.cdr ?? {};
  const ipdr = validation.sources?.ipdr ?? {};
  const matching = validation.matching ?? {};
  const temporal = validation.temporal ?? {};
  const dateRange = (source: typeof cdr) => `${source.date_range?.start ?? "—"} → ${source.date_range?.end ?? "—"}`;
  return (
    <section className="identity-evidence">
      <div className="panel-heading"><div><span className="nexus-eyebrow">IDENTITY EVIDENCE</span><strong>Full-file validation result</strong></div><span className="dataset-kind analyst-labelled">{validation.status ?? "unknown"}</span></div>
      <div className="identity-evidence-grid">
        <div><span>CDR subject column</span><strong>{cdr.subject_column ?? "—"}</strong></div>
        <div><span>IPDR subscriber column</span><strong>{ipdr.subscriber_column ?? ipdr.subject_column ?? "—"}</strong></div>
        <div><span>CDR / IPDR subjects</span><strong>{matching.cdr_subject_count ?? 0} / {matching.ipdr_subscriber_count ?? 0}</strong></div>
        <div><span>CDR contacts (not subjects)</span><strong>{cdr.contact_count ?? 0}</strong></div>
        <div><span>Matched identifiers</span><strong>{matching.matched_count ?? 0}</strong></div>
        <div><span>Unmatched CDR / IPDR</span><strong>{matching.unmatched_cdr_count ?? 0} / {matching.unmatched_ipdr_count ?? 0}</strong></div>
        <div><span>CDR observation period</span><strong>{dateRange(cdr)}</strong></div>
        <div><span>IPDR observation period</span><strong>{dateRange(ipdr)}</strong></div>
        <div><span>Temporal status</span><strong>{temporal.status ?? "unavailable"}</strong></div>
      </div>
      <p className="identity-evidence-copy">
        Matched identifiers: {(matching.matched_identifiers ?? []).slice(0, 12).join(", ") || "none"}.<br />
        Unmatched CDR subjects: {(matching.unmatched_cdr_subjects ?? []).slice(0, 12).join(", ") || "none"}.<br />
        Unmatched IPDR subscribers: {(matching.unmatched_ipdr_subscribers ?? []).slice(0, 12).join(", ") || "none"}.
      </p>
      <p className="identity-evidence-copy">
        {(temporal.explanation ?? "Temporal evidence unavailable.") + " "}
        {validation.identity_evidence?.statement ?? "Identifier agreement supports matching; it does not prove personal identity." + " "}
        {(decision?.warnings ?? []).join(" ")}
        {(decision?.available_operations?.length ?? 0) > 0 ? ` Available operations: ${decision?.available_operations?.join(", ")}.` : ""}
      </p>
    </section>
  );
}

export function DatasetsWorkspace() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewDatasetId, setPreviewDatasetId] = useState("");
  const [previewRole, setPreviewRole] = useState<"cdr" | "ipdr">("cdr");
  const [schemaEdits, setSchemaEdits] = useState<Record<string, string | null>>({});
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [retention, setRetention] = useState<RetentionSettings>({
    retention_days: 30,
    auto_purge: false,
  });
  const [savingRetention, setSavingRetention] = useState(false);
  const [uploadHasCdr, setUploadHasCdr] = useState(false);
  const [uploadHasIpdr, setUploadHasIpdr] = useState(false);
  const [uploadIdentityAnswer, setUploadIdentityAnswer] = useState("");
  const [uploadGenerationMode, setUploadGenerationMode] = useState("");
  const [uploadConfirmation, setUploadConfirmation] = useState(false);
  const [uploadMappings, setUploadMappings] = useState("");

  const refresh = useCallback(async () => {
    const [datasetRows, runRows, retentionSettings] = await Promise.all([
      nexusFetch<DatasetRecord[]>("/datasets"),
      nexusFetch<RunRecord[]>("/runs"),
      nexusFetch<RetentionSettings>("/settings/retention"),
    ]);
    setDatasets(datasetRows);
    setRuns(runRows);
    setRetention(retentionSettings);
  }, []);
  useEffect(() => {
    queueMicrotask(() => {
      void refresh().catch((reason: Error) => setError(reason.message));
    });
  }, [refresh]);
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("run");
    if (requested) {
      queueMicrotask(() => setSelectedRun(requested));
    }
  }, []);

  const visibleDatasets = useMemo(() => {
    const normalized = query.toLowerCase();
    return datasets.filter((dataset) => `${dataset.name} ${dataset.id} ${dataset.kind}`.toLowerCase().includes(normalized));
  }, [datasets, query]);

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setUploading(true);
    setError("");
    const form = new FormData(event.currentTarget);
    const hasCdr = form.get("cdr") instanceof File && (form.get("cdr") as File).size > 0;
    const hasIpdr = form.getAll("ipdr_files").some((value) => value instanceof File && value.size > 0);
    if (hasCdr && hasIpdr) {
      if (!uploadIdentityAnswer) {
        setError("Choose how the CDR and IPDR subjects relate before creating a combined dataset.");
        setUploading(false);
        return;
      }
      if (!uploadConfirmation) {
        setError("Review the identity evidence and confirm the review before continuing.");
        setUploading(false);
        return;
      }
      if (uploadMappings.trim()) {
        try {
          JSON.parse(uploadMappings);
        } catch {
          setError("Explicit alias mappings must be valid JSON.");
          setUploading(false);
          return;
        }
      }
      form.set("identity_answer", uploadIdentityAnswer);
      form.set("generation_mode", uploadGenerationMode || "");
      form.set("identity_confirmation", String(uploadConfirmation));
      if (uploadMappings.trim()) form.set("identity_mappings", uploadMappings.trim());
    }
    try {
      await nexusFetch<DatasetRecord>("/datasets", {
        method: "POST",
        body: form,
      });
      event.currentTarget.reset();
      setUploadHasCdr(false);
      setUploadHasIpdr(false);
      setUploadIdentityAnswer("");
      setUploadGenerationMode("");
      setUploadConfirmation(false);
      setUploadMappings("");
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setUploading(false);
    }
  }

  async function inspect(datasetId: string, role: "cdr" | "ipdr") {
    setError("");
    try {
      const payload = await nexusFetch<Preview>(`/datasets/${datasetId}/preview/${role}`);
      let identityValidation: IdentityValidation | null = null;
      let identityDecision: IdentityDecision | null = null;
      try {
        const identity = await nexusFetch<{ validation?: IdentityValidation; identity_decision?: IdentityDecision }>(`/datasets/${datasetId}/validation`);
        identityValidation = identity.validation ?? null;
        identityDecision = identity.identity_decision ?? null;
      } catch {
        // Older datasets are intentionally shown as unknown until an explicit
        // full-file validation is available.
      }
      setPreview({ ...payload, identity_validation: identityValidation, identity_decision: identityDecision });
      setPreviewDatasetId(datasetId);
      setPreviewRole(role);
      setSchemaEdits(payload.resolved);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function saveSchema() {
    if (!previewDatasetId) return;
    setError("");
    try {
      await nexusFetch(`/datasets/${previewDatasetId}/schema`, {
        method: "PATCH",
        body: JSON.stringify({
          kind: previewRole,
          overrides: schemaEdits,
        }),
      });
      await inspect(previewDatasetId, previewRole);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function saveRetention() {
    setSavingRetention(true);
    setError("");
    try {
      const updated = await nexusFetch<RetentionSettings>("/settings/retention", {
        method: "PATCH",
        body: JSON.stringify(retention),
      });
      setRetention(updated);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSavingRetention(false);
    }
  }

  async function trash(dataset: DatasetRecord) {
    if (!window.confirm(`Move “${dataset.name}” to recoverable trash? No files will be erased.`)) return;
    setError("");
    try {
      await nexusFetch<DatasetRecord>(`/datasets/${dataset.id}`, { method: "DELETE" });
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function restore(dataset: DatasetRecord) {
    setError("");
    try {
      await nexusFetch<DatasetRecord>(`/datasets/${dataset.id}/restore`, { method: "POST" });
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  return (
    <NexusShell title="Datasets & Run History" eyebrow="VERSIONED WORKSPACE" actions={<span className="data-badge"><Database size={14} /> Persistent local registry</span>}>
      <section className="dataset-upload nexus-panel">
        <div className="panel-heading"><div><span className="nexus-eyebrow">NEW DATASET</span><strong>Upload CDR, IPDR, labels, or an output ZIP</strong></div><Upload size={20} /></div>
        <form onSubmit={upload}>
          <label className="field"><span>Dataset name</span><input name="name" placeholder="Investigation or experiment name" required /></label>
          <label className="field"><span>Data classification</span><select name="kind" defaultValue="uploaded"><option value="uploaded">Uploaded / unknown</option><option value="real">Real restricted data</option><option value="synthetic">Synthetic data</option><option value="analyst-labelled">Analyst-labelled</option></select></label>
          <label className="file-field"><FileSpreadsheet size={17} /><span><strong>CDR CSV</strong><small>Required for generation</small></span><input type="file" name="cdr" accept=".csv" onChange={(event) => setUploadHasCdr(Boolean(event.target.files?.length))} /></label>
          <label className="file-field"><FileSpreadsheet size={17} /><span><strong>IPDR CSV file(s)</strong><small>Select multiple compatible exports; they are merged with source lineage preserved</small></span><input type="file" name="ipdr_files" accept=".csv" multiple onChange={(event) => setUploadHasIpdr(Boolean(event.target.files?.length))} /></label>
          <label className="file-field"><FileSpreadsheet size={17} /><span><strong>Subscribers</strong><small>Identity aliases and map</small></span><input type="file" name="subscribers" accept=".csv" /></label>
          <label className="file-field"><FileSpreadsheet size={17} /><span><strong>Base truth</strong><small>Reviewed or synthetic labels</small></span><input type="file" name="base_truth" accept=".csv" /></label>
          <label className="file-field"><Upload size={17} /><span><strong>Existing output ZIP</strong><small>Import a complete run</small></span><input type="file" name="archive" accept=".zip" /></label>
          {uploadHasCdr && uploadHasIpdr ? (
            <section className="identity-gate" aria-labelledby="dataset-identity-question">
              <span className="nexus-eyebrow">IDENTITY REVIEW REQUIRED</span>
              <h3 id="dataset-identity-question">Do these CDR and IPDR files belong to the same person?</h3>
              <p>This determines how calling and internet activity are combined. Identifier agreement supports matching; it does not prove personal identity. The backend scans every record before accepting the answer.</p>
              <div className="identity-radio-grid" role="radiogroup" aria-label="CDR and IPDR identity relationship">
                {[
                  ["same_person", "Same person"],
                  ["different_people", "Different people"],
                  ["multiple_person_dataset", "Multiple-person dataset"],
                  ["not_sure", "Not sure"],
                ].map(([value, label]) => (
                  <label key={value}><input type="radio" name="dataset_identity_answer" value={value} checked={uploadIdentityAnswer === value} onChange={() => setUploadIdentityAnswer(value)} /> {label}</label>
                ))}
              </div>
              <div className="identity-gate-fields">
                <label className="field"><span>Generation / processing mode</span><select value={uploadGenerationMode} onChange={(event) => setUploadGenerationMode(event.target.value)}><option value="">Choose after evidence review</option><option value="combined">Combine one consistent subject mapping</option><option value="multi_person_conditioned">Match each subject independently</option><option value="single_person_template">Explicit single-person template expansion</option><option value="cdr_only">CDR-only generation</option><option value="ipdr_only_correlation">IPDR-only correlation</option><option value="separate_sources">Keep sources separate</option></select></label>
                <label className="field"><span>Explicit subscriber/alias mapping <b>optional</b></span><textarea value={uploadMappings} onChange={(event) => setUploadMappings(event.target.value)} placeholder='{"ipdr-id":"cdr-id"}' /></label>
              </div>
              <label className="consent-row"><input type="checkbox" checked={uploadConfirmation} onChange={(event) => setUploadConfirmation(event.target.checked)} /> <span>I reviewed the detected evidence. This checkbox records review and cannot bypass contradictory data.</span></label>
            </section>
          ) : null}
          <button type="submit" className="button button-primary" disabled={uploading}><Upload size={16} /> {uploading ? "Uploading…" : "Create dataset"}</button>
        </form>
      </section>
      <section className="nexus-panel retention-panel">
        <div>
          <span className="nexus-eyebrow">RECOVERABILITY & RETENTION</span>
          <strong><ShieldCheck size={18} /> Safe local data controls</strong>
          <p>Trash is recoverable and never erases files automatically. Retention records the intended review window; permanent purge remains disabled.</p>
        </div>
        <label className="field"><span>Retention window</span><select value={retention.retention_days} onChange={(event) => setRetention((current) => ({ ...current, retention_days: Number(event.target.value) }))}><option value={7}>7 days</option><option value={30}>30 days</option><option value={90}>90 days</option><option value={180}>180 days</option><option value={365}>1 year</option></select></label>
        <button type="button" className="button button-ghost" disabled={savingRetention} onClick={saveRetention}>{savingRetention ? "Saving…" : "Save administrator setting"}</button>
      </section>
      {error ? <div className="inline-error"><AlertTriangle size={15} /> {error}</div> : null}
      <div className="dataset-history-grid">
        <section className="nexus-panel">
          <div className="panel-heading"><div><span className="nexus-eyebrow">DATASETS</span><strong>{datasets.length} registered versions</strong></div><label className="table-search"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search" /></label></div>
          <div className="dataset-list">
            {visibleDatasets.map((dataset) => (
              <div key={dataset.id} className={dataset.status === "trashed" ? "trashed" : ""}>
                <span className={`dataset-kind ${dataset.kind}`}>{dataset.kind}</span>
                <div><strong>{dataset.name}</strong><small>{dataset.id} · {formatDateTime(dataset.created_at)}</small><span>{formatIndianNumber(dataset.row_counts?.cdr_compact ?? dataset.row_counts?.cdr ?? 0)} CDR · {formatIndianNumber(dataset.row_counts?.ipdr_compact ?? dataset.row_counts?.ipdr ?? 0)} IPDR · {dataset.status} · identity {String(dataset.metadata?.identity_validation_status ?? "unknown")}</span></div>
                <div className="dataset-actions">
                  {dataset.status === "trashed" ? (
                    <button type="button" onClick={() => restore(dataset)}><RotateCcw size={13} /> Restore</button>
                  ) : (
                    <>
                      <button type="button" onClick={() => inspect(dataset.id, "cdr")}>CDR schema</button>
                      <button type="button" onClick={() => inspect(dataset.id, "ipdr")}>IPDR schema</button>
                      <button type="button" className="danger" onClick={() => trash(dataset)}><Trash2 size={13} /> Trash</button>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
        <section className="nexus-panel">
          <div className="panel-heading"><div><span className="nexus-eyebrow">RUN HISTORY</span><strong>{runs.length} jobs</strong></div><History size={19} /></div>
          <div className="run-list history">
            {runs.map((run) => (
              <button type="button" className={selectedRun === run.id ? "selected" : ""} onClick={() => setSelectedRun(run.id)} key={run.id}>
                <span className={`status-dot ${run.status}`} /><div><strong>{run.kind}</strong><small>{run.id} · {formatDateTime(run.updated_at)}</small></div><span>{Math.round(run.progress)}%</span>
              </button>
            ))}
          </div>
        </section>
      </div>
      {selectedRun ? <RunMonitor runId={selectedRun} /> : null}
      {preview ? (
        <section className="nexus-panel schema-preview">
          <div className="panel-heading"><div><span className="nexus-eyebrow">SCHEMA PREVIEW</span><strong>{preview.columns.length} source headers</strong></div><div className="schema-preview-actions"><button type="button" className="button button-primary small" onClick={saveSchema}>Save mapping</button><button type="button" className="button button-ghost small" onClick={() => setPreview(null)}>Close</button></div></div>
          <div className="resolved-schema editable">{Object.entries(preview.resolved).map(([field, column]) => <label key={field}><b>{field}</b><i>→</i><select value={schemaEdits[field] ?? ""} onChange={(event) => setSchemaEdits((current) => ({ ...current, [field]: event.target.value || null }))}><option value="">Unresolved / auto</option>{preview.columns.map((candidate) => <option value={candidate} key={candidate}>{candidate}</option>)}</select><small>Detected: {column ?? "unresolved"}</small></label>)}</div>
          {preview.identity_validation ? <IdentityEvidence validation={preview.identity_validation} decision={preview.identity_decision} /> : null}
          <div className="data-table-scroll"><table className="data-table"><thead><tr>{preview.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{preview.preview.map((row, index) => <tr key={index}>{preview.columns.map((column) => <td key={column}>{row[column]}</td>)}</tr>)}</tbody></table></div>
        </section>
      ) : null}
    </NexusShell>
  );
}
