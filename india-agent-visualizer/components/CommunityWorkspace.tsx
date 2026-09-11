"use client";

import {
  Activity,
  AlertTriangle,
  Check,
  Cpu,
  Network,
  Play,
  Search,
  ShieldAlert,
  Users,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { formatIndianNumber, nexusFetch, type RunRecord } from "../lib/nexus-api";
import { NexusShell } from "./NexusShell";
import { RunMonitor } from "./RunMonitor";

type GraphPayload = {
  summary: Record<string, number>;
  nodes: Array<Record<string, string | number>>;
  edges: Array<Record<string, string | number>>;
};
type PairPage = { total: number; rows: Array<Record<string, string>> };

export function CommunityWorkspace() {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [sourceRunId, setSourceRunId] = useState("");
  const [edgeThreshold, setEdgeThreshold] = useState(0.4);
  const [windowSeconds, setWindowSeconds] = useState(300);
  const [runId, setRunId] = useState("");
  const [completed, setCompleted] = useState(false);
  const [graph, setGraph] = useState<GraphPayload | null>(null);
  const [pairs, setPairs] = useState<PairPage | null>(null);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    nexusFetch<RunRecord[]>("/runs").then((items) => {
      const correlations = items.filter((run) => run.kind === "correlation" && run.status === "complete");
      const requestedRunId = new URLSearchParams(window.location.search).get("run");
      const community = items.find(
        (run) => run.kind === "community"
          && run.status === "complete"
          && (!requestedRunId || run.id === requestedRunId),
      );
      setRuns(correlations);
      setSourceRunId(
        community?.parent_run_id
          ?? correlations[0]?.id
          ?? "",
      );
      if (community) {
        setRunId(community.id);
        setCompleted(true);
      }
    }).catch((reason: Error) => setError(reason.message));
  }, []);
  useEffect(() => {
    if (!completed || !runId) return;
    nexusFetch<GraphPayload>(`/runs/${runId}/graph`).then(setGraph).catch(() => setGraph(null));
  }, [completed, runId]);
  useEffect(() => {
    if (!completed || !runId) return;
    const query = search ? `?search=${encodeURIComponent(search)}&page_size=25` : "?page_size=25";
    nexusFetch<PairPage>(`/runs/${runId}/pairs${query}`).then(setPairs).catch(() => setPairs(null));
  }, [completed, runId, search]);

  const topNodes = useMemo(
    () => [...(graph?.nodes ?? [])].sort(
      (left, right) => Number(right.composite_centrality ?? 0) - Number(left.composite_centrality ?? 0),
    ).slice(0, 8),
    [graph],
  );

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setCompleted(false);
    try {
      const run = await nexusFetch<RunRecord>("/runs/community", {
        method: "POST",
        body: JSON.stringify({
          source_run_id: sourceRunId,
          edge_threshold: edgeThreshold,
          temporal_window_seconds: windowSeconds,
          maximum_temporal_matches: 100000,
        }),
      });
      setRunId(run.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function reviewPair(pairId: string, label: 0 | 1) {
    if (!runId) return;
    try {
      await nexusFetch(`/runs/${runId}/pairs/${encodeURIComponent(pairId)}/review`, {
        method: "PUT",
        body: JSON.stringify({
          label,
          note: "",
          reviewer: "local_analyst",
        }),
      });
      setPairs((current) => current ? {
        ...current,
        rows: current.rows.map((row) => row.pair_id === pairId ? {
          ...row,
          analyst_review_status: label === 1 ? "confirmed_positive" : "confirmed_negative",
        } : row),
      } : current);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  return (
    <NexusShell
      title="Community Intelligence"
      eyebrow="CROSS-DATASET SHADOW-NET"
      actions={<span className="evidence-badge"><Network size={14} /> Louvain graph analysis</span>}
    >
      <div className="two-column-workspace">
        <form className="nexus-panel config-panel" onSubmit={submit}>
          <div className="panel-heading">
            <div><span className="nexus-eyebrow">GRAPH CONFIGURATION</span><strong>Relationship evidence network</strong></div>
            <Network size={19} />
          </div>
          <label className="field full">
            <span>Completed correlation run</span>
            <select value={sourceRunId} onChange={(event) => setSourceRunId(event.target.value)} required>
              {runs.map((run) => <option value={run.id} key={run.id}>{run.id} · {run.message}</option>)}
            </select>
          </label>
          <label className="field full">
            <span>Minimum edge probability <b>{edgeThreshold.toFixed(2)}</b></span>
            <input type="range" min={0} max={1} step={0.05} value={edgeThreshold} onChange={(event) => setEdgeThreshold(Number(event.target.value))} />
          </label>
          <label className="field full">
            <span>CDR/IPDR temporal window</span>
            <select value={windowSeconds} onChange={(event) => setWindowSeconds(Number(event.target.value))}>
              <option value={60}>±1 minute</option>
              <option value={300}>±5 minutes</option>
              <option value={600}>±10 minutes</option>
              <option value={1800}>±30 minutes</option>
            </select>
          </label>
          <div className="analysis-feature-list">
            <span><Users size={15} /><strong>Louvain communities</strong><small>Internal correlation and leaders</small></span>
            <span><Activity size={15} /><strong>Four centralities</strong><small>Degree, bridge, closeness, eigenvector</small></span>
            <span><ShieldAlert size={15} /><strong>Device audit</strong><small>SIM-swap and shared-device patterns</small></span>
            <span><Cpu size={15} /><strong>Temporal synthesis</strong><small>Calls matched with both people’s sessions</small></span>
          </div>
          {error ? <div className="inline-error"><AlertTriangle size={15} /> {error}</div> : null}
          <button className="button button-primary full" type="submit" disabled={!sourceRunId}>
            <Play size={16} /> Build community intelligence
          </button>
        </form>
        <div className="monitor-column">
          {runId ? <RunMonitor runId={runId} onUpdate={(run) => setCompleted(run.status === "complete")} /> : (
            <div className="nexus-panel empty-workspace"><Network size={27} /><h3>Reveal structure without losing provenance</h3><p>Direct CDR, inferred IPDR, temporal, device, and model evidence remain separate on every edge.</p></div>
          )}
        </div>
      </div>
      {graph ? (
        <>
          <section className="summary-grid graph-summary">
            <div><Users size={17} /><span>Nodes</span><strong>{formatIndianNumber(graph.summary.nodes)}</strong></div>
            <div><Network size={17} /><span>Edges</span><strong>{formatIndianNumber(graph.summary.edges)}</strong></div>
            <div><Activity size={17} /><span>Communities</span><strong>{formatIndianNumber(graph.summary.communities)}</strong></div>
            <div><ShieldAlert size={17} /><span>Device flags</span><strong>{formatIndianNumber(graph.summary.device_anomalies)}</strong></div>
            <div><Cpu size={17} /><span>Temporal matches</span><strong>{formatIndianNumber(graph.summary.temporal_matches)}</strong></div>
          </section>
          <section className="community-results-grid">
            <div className="nexus-panel">
              <div className="panel-heading"><div><span className="nexus-eyebrow">CENTRALITY</span><strong>Highest-positioned nodes</strong></div></div>
              <div className="centrality-list">
                {topNodes.map((node, index) => (
                  <div key={String(node.id)}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{String(node.name || node.id)}</strong><small>Community {String(node.community)} · {String(node.state || "—")}</small></div><b>{Number(node.composite_centrality).toFixed(3)}</b></div>
                ))}
              </div>
            </div>
            <div className="nexus-panel network-field" aria-label="Abstract community layout">
              <div className="network-grid-lines" />
              {topNodes.map((node, index) => (
                <span
                  className={`network-node community-${Number(node.community) % 5}`}
                  style={{ left: `${14 + (index * 31) % 74}%`, top: `${15 + (index * 43) % 66}%`, width: `${18 + Number(node.composite_centrality) * 38}px`, height: `${18 + Number(node.composite_centrality) * 38}px` }}
                  title={String(node.name || node.id)}
                  key={String(node.id)}
                />
              ))}
              <div className="network-field-label"><span>STRUCTURAL OVERVIEW</span><strong>{graph.summary.communities} detected communities</strong></div>
            </div>
          </section>
        </>
      ) : null}
      {completed ? (
        <section className="nexus-panel result-table-panel">
          <div className="panel-heading">
            <div><span className="nexus-eyebrow">PAIR EVIDENCE</span><strong>{formatIndianNumber(pairs?.total ?? 0)} relationships</strong></div>
            <label className="table-search"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search person" /></label>
          </div>
          <div className="data-table-scroll">
            <table className="data-table"><thead><tr><th>Pair</th><th>CDR</th><th>IPDR</th><th>Temporal</th><th>Probability</th><th>Confidence</th><th>Provenance</th><th>Review</th></tr></thead>
              <tbody>{(pairs?.rows ?? []).map((row) => <tr key={row.pair_id}><td><strong>{row.person_a}</strong><small>{row.person_b}</small></td><td>{row.direct_cdr_interaction_count}</td><td>{Number(row.ipdr_correlation_score).toFixed(3)}</td><td>{row.cdr_ipdr_temporal_coincidences}</td><td><strong className="score-value">{Number(row.relationship_probability).toFixed(3)}</strong></td><td>{row.confidence}</td><td>{row.label_provenance}</td><td><div className="review-actions"><button type="button" className={row.analyst_review_status === "confirmed_positive" ? "active positive" : ""} onClick={() => reviewPair(row.pair_id, 1)} title="Confirm interaction"><Check size={14} /></button><button type="button" className={row.analyst_review_status === "confirmed_negative" ? "active negative" : ""} onClick={() => reviewPair(row.pair_id, 0)} title="Confirm no interaction"><X size={14} /></button></div></td></tr>)}</tbody>
            </table>
          </div>
        </section>
      ) : null}
    </NexusShell>
  );
}
