"use client";

import {
  AlertTriangle,
  Fingerprint,
  Gauge,
  Network,
  Play,
  RadioTower,
  Search,
  ShieldCheck,
  Timer,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  formatIndianNumber,
  nexusFetch,
  type DatasetRecord,
  type RunRecord,
} from "../lib/nexus-api";
import { NexusShell } from "./NexusShell";
import { RunMonitor } from "./RunMonitor";

type PairPage = { total: number; rows: Array<Record<string, string>> };

const signals = [
  ["Domains", "Jaccard + rarity + TF-IDF"],
  ["Applications", "Overlap + TF-IDF"],
  ["Ports", "Frequency + IDF cosine"],
  ["Temporal", "Hourly + weekly + EMD"],
  ["Sessions", "Duration + frequency"],
  ["Behaviour", "37-D cosine + distance"],
  ["Rare events", "Capped rarity bonus"],
];

export function CorrelatorWorkspace() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [source, setSource] = useState("");
  const [exhaustive, setExhaustive] = useState(true);
  const [threshold, setThreshold] = useState(0.4);
  const [rdns, setRdns] = useState(false);
  const [rdnsConsent, setRdnsConsent] = useState(false);
  const [runId, setRunId] = useState("");
  const [completed, setCompleted] = useState(false);
  const [pairs, setPairs] = useState<PairPage | null>(null);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    Promise.all([
      nexusFetch<DatasetRecord[]>("/datasets"),
      nexusFetch<RunRecord[]>("/runs"),
    ]).then(([datasetRows, runRows]) => {
      setDatasets(datasetRows);
      setRuns(runRows);
      const requestedRunId = new URLSearchParams(window.location.search).get("run");
      const correlation = runRows.find(
        (run) => run.kind === "correlation"
          && run.status === "complete"
          && (!requestedRunId || run.id === requestedRunId),
      );
      const generator = correlation?.parent_run_id
        ? runRows.find((run) => run.id === correlation.parent_run_id)
        : runRows.find((run) => run.kind === "generator" && run.status === "complete");
      setSource(generator ? `run:${generator.id}` : datasetRows[0] ? `dataset:${datasetRows[0].id}` : "");
      if (correlation) {
        setRunId(correlation.id);
        setCompleted(true);
      }
    }).catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!runId || !completed) return;
    const query = search ? `?search=${encodeURIComponent(search)}&page_size=25` : "?page_size=25";
    nexusFetch<PairPage>(`/runs/${runId}/correlations${query}`).then(setPairs).catch(() => setPairs(null));
  }, [completed, runId, search]);

  const sourceOptions = useMemo(() => [
    ...runs
      .filter((run) => run.kind === "generator" && run.status === "complete")
      .map((run) => ({ value: `run:${run.id}`, label: `Generated run · ${run.id}` })),
    ...datasets.map((dataset) => ({ value: `dataset:${dataset.id}`, label: `Dataset · ${dataset.name}` })),
  ], [datasets, runs]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    setCompleted(false);
    try {
      const [sourceType, sourceId] = source.split(":", 2);
      const run = await nexusFetch<RunRecord>("/runs/correlation", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: sourceType === "dataset" ? sourceId : null,
          source_run_id: sourceType === "run" ? sourceId : null,
          exhaustive,
          max_exhaustive_people: 2500,
          max_candidate_pairs: 2000000,
          adaptive_alpha: 0.5,
          base_weights: [0.18, 0.12, 0.12, 0.20, 0.12, 0.18, 0.08],
          graph_threshold: threshold,
          probable_positive_threshold: 0.70,
          probable_negative_threshold: 0.25,
          enable_rdns: rdns,
          rdns_consent: rdns && rdnsConsent,
          rdns_timeout_seconds: 1.5,
          dbscan_eps: 1.1,
          dbscan_min_samples: 3,
          isolation_contamination: 0.05,
          temporal_window_seconds: 300,
        }),
      });
      setRunId(run.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <NexusShell
      title="IPDR Correlator"
      eyebrow="BEHAVIOURAL RELATIONSHIP EVIDENCE"
      actions={<span className="evidence-badge"><Fingerprint size={14} /> Explainable scoring</span>}
    >
      <section className="signal-grid">
        {signals.map(([name, detail], index) => (
          <div key={name}><span>{String(index + 1).padStart(2, "0")}</span><strong>{name}</strong><small>{detail}</small></div>
        ))}
      </section>
      <div className="two-column-workspace">
        <form className="nexus-panel config-panel" onSubmit={submit}>
          <div className="panel-heading">
            <div><span className="nexus-eyebrow">CORRELATION CONFIG</span><strong>Pair-scoring engine</strong></div>
            <RadioTower size={19} />
          </div>
          <label className="field full">
            <span>IPDR source</span>
            <select value={source} onChange={(event) => setSource(event.target.value)} required>
              {sourceOptions.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}
            </select>
          </label>
          <label className="field full">
            <span>Graph edge threshold <b>{threshold.toFixed(2)}</b></span>
            <input type="range" min={0} max={1} step={0.05} value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
            <small>All pair scores remain downloadable; this threshold controls graph inclusion.</small>
          </label>
          <div className="switch-stack">
            <label className="switch-row">
              <input type="checkbox" checked={exhaustive} onChange={(event) => setExhaustive(event.target.checked)} />
              <span><strong>Exhaustive pair scoring</strong><small>For larger datasets, disable to use rare-item and temporal candidate blocking.</small></span>
            </label>
            <label className="switch-row">
              <input type="checkbox" checked={rdns} onChange={(event) => { setRdns(event.target.checked); if (!event.target.checked) setRdnsConsent(false); }} />
              <span><strong>Reverse DNS enrichment</strong><small>Off by default. Destination IPs remain local when disabled.</small></span>
            </label>
            {rdns ? (
              <label className="consent-row">
                <input type="checkbox" checked={rdnsConsent} onChange={(event) => setRdnsConsent(event.target.checked)} />
                <span>I explicitly consent to network DNS lookups for destination IPs.</span>
              </label>
            ) : null}
          </div>
          <div className="correlation-facts">
            <span><Gauge size={15} /> Variance-adaptive weighting</span>
            <span><Network size={15} /> DBSCAN + spectral corroboration</span>
            <span><Timer size={15} /> Isolation Forest is evidence, not a positive bonus</span>
          </div>
          {rdns && !rdnsConsent ? <div className="inline-warning"><AlertTriangle size={15} /> Consent is required before rDNS can run.</div> : null}
          {error ? <div className="inline-error">{error}</div> : null}
          <button className="button button-primary full" disabled={!source || submitting || (rdns && !rdnsConsent)} type="submit">
            <Play size={16} /> {submitting ? "Starting…" : "Run IPDR correlation"}
          </button>
        </form>
        <div className="monitor-column">
          {runId ? (
            <RunMonitor runId={runId} onUpdate={(run) => setCompleted(run.status === "complete")} />
          ) : (
            <div className="nexus-panel empty-workspace">
              <ShieldCheck size={27} />
              <h3>Behavioural evidence, with context</h3>
              <p>Every score includes its components, provenance, cluster corroboration, anomaly flag, and plain-language explanation.</p>
            </div>
          )}
        </div>
      </div>
      {completed ? (
        <section className="nexus-panel result-table-panel">
          <div className="panel-heading">
            <div><span className="nexus-eyebrow">TOP CORRELATIONS</span><strong>{formatIndianNumber(pairs?.total ?? 0)} canonical pairs</strong></div>
            <label className="table-search"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search either person" /></label>
          </div>
          <div className="data-table-scroll">
            <table className="data-table">
              <thead><tr><th>Person A</th><th>Person B</th><th>Score</th><th>Category</th><th>Confidence</th><th>Explanation</th></tr></thead>
              <tbody>
                {(pairs?.rows ?? []).map((row) => (
                  <tr key={row.pair_id}>
                    <td>{row.person_a}</td><td>{row.person_b}</td>
                    <td><strong className="score-value">{Number(row.ipdr_correlation_score).toFixed(3)}</strong></td>
                    <td><span className={`category-pill ${row.category.replace(" ", "-")}`}>{row.category}</span></td>
                    <td>{row.confidence_region}</td><td className="explanation-cell">{row.explanation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </NexusShell>
  );
}
