"use client";

import { Archive, Download, FileText, Filter } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  artifactDownloadUrl,
  formatDateTime,
  formatIndianNumber,
  nexusFetch,
} from "../lib/nexus-api";
import { NexusShell } from "./NexusShell";

type ReportArtifact = {
  id: string;
  run_id: string;
  run_kind: string;
  kind: string;
  filename: string;
  size: number;
  created_at: string;
};

export function ReportsWorkspace() {
  const [allArtifacts, setAllArtifacts] = useState<ReportArtifact[]>([]);
  const [kind, setKind] = useState("all");
  const [error, setError] = useState("");
  useEffect(() => {
    nexusFetch<ReportArtifact[]>("/artifacts?completed_only=true&limit=2000")
      .then(setAllArtifacts)
      .catch((reason: Error) => setError(reason.message));
  }, []);
  const artifacts = useMemo(
    () => allArtifacts.filter((item) => kind === "all" || item.kind === kind),
    [allArtifacts, kind],
  );
  const kinds = [...new Set(allArtifacts.map((artifact) => artifact.kind))].sort();
  return (
    <NexusShell title="Reports & Downloads" eyebrow="ARTIFACT LIBRARY" actions={<span className="data-badge"><Archive size={14} /> {artifacts.length} files</span>}>
      <section className="report-hero">
        <div><FileText size={26} /><h2>Every output remains tied to its run.</h2><p>Download evidence tables, full schemas, graph exports, PDF dossiers, model artifacts, or complete ZIP archives.</p></div>
        <label className="field report-filter"><span><Filter size={14} /> Artifact type</span><select value={kind} onChange={(event) => setKind(event.target.value)}><option value="all">All types</option>{kinds.map((value) => <option value={value} key={value}>{value}</option>)}</select></label>
      </section>
      {error ? <div className="inline-error">{error}</div> : null}
      <section className="artifact-library">
        {artifacts.map((artifact) => (
          <a href={artifactDownloadUrl(artifact.id)} className="artifact-card" key={artifact.id}>
            <span className="artifact-file-icon"><FileText size={19} /></span>
            <div><strong>{artifact.filename}</strong><small>{artifact.kind} · {formatIndianNumber(artifact.size)} bytes</small><span>{artifact.run_kind} · {artifact.run_id} · {formatDateTime(artifact.created_at)}</span></div>
            <Download size={17} />
          </a>
        ))}
        {!artifacts.length ? <div className="empty-state">No completed-run artifacts match this filter.</div> : null}
      </section>
    </NexusShell>
  );
}
