"use client";

import {
  ArrowRight,
  BrainCircuit,
  Database,
  FileArchive,
  GitBranch,
  Map,
  Network,
  PhoneCall,
  RadioTower,
  Sparkles,
  Users,
  Wifi,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import {
  formatDateTime,
  formatIndianNumber,
  nexusFetch,
  type DatasetRecord,
  type ModelRecord,
  type RunRecord,
} from "../lib/nexus-api";
import { NexusShell } from "./NexusShell";

type DashboardPayload = {
  summary: {
    datasets: number;
    active_jobs: number;
    agents: number;
    cdr_events: number;
    ipdr_sessions: number;
    relationship_edges: number;
    models: number;
  };
  datasets: DatasetRecord[];
  runs: RunRecord[];
  models: ModelRecord[];
  latest_model: ModelRecord | null;
  privacy_notice: string;
};

const modules = [
  {
    href: "/generator",
    title: "Synthetic Data Generator",
    copy: "Population-aware agents, realistic CDR/IPDR, and IPDR-conditioned relationship edges.",
    icon: Sparkles,
    tone: "amber",
    metric: "CGAN",
  },
  {
    href: "/correlator",
    title: "IPDR Correlator",
    copy: "Explainable domain, application, port, temporal, session, and behaviour similarity.",
    icon: RadioTower,
    tone: "cyan",
    metric: "7 signals",
  },
  {
    href: "/community",
    title: "Community Intelligence",
    copy: "Louvain communities, centrality, device anomalies, and cross-dataset timelines.",
    icon: Network,
    tone: "violet",
    metric: "Shadow-Net",
  },
  {
    href: "/models",
    title: "Neural Network Studio",
    copy: "Train, evaluate, activate, and use relationship models without moving files.",
    icon: BrainCircuit,
    tone: "green",
    metric: "PyTorch",
  },
  {
    href: "/explorer",
    title: "India Network Explorer",
    copy: "India-only map with agent zoom, complete incoming/outgoing activity, and profiles.",
    icon: Map,
    tone: "cyan",
    metric: "36 regions",
  },
  {
    href: "/pipeline",
    title: "Run Full Pipeline",
    copy: "Correlate → condition CGAN → generate → detect communities → train and package.",
    icon: GitBranch,
    tone: "amber",
    metric: "One run",
  },
  {
    href: "/datasets",
    title: "Datasets & Run History",
    copy: "Upload, inspect schemas, review quality warnings, retry jobs, and preserve provenance.",
    icon: Database,
    tone: "violet",
    metric: "Versioned",
  },
  {
    href: "/reports",
    title: "Reports & Downloads",
    copy: "CSV, Excel, JSON, graph, model, PDF dossier, and complete ZIP artifacts.",
    icon: FileArchive,
    tone: "green",
    metric: "Auditable",
  },
];

export function HomeDashboard() {
  const [data, setData] = useState<DashboardPayload | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    nexusFetch<DashboardPayload>("/dashboard")
      .then(setData)
      .catch((reason: Error) => setError(reason.message));
    const timer = window.setInterval(() => {
      nexusFetch<DashboardPayload>("/dashboard").then(setData).catch(() => undefined);
    }, 5_000);
    return () => window.clearInterval(timer);
  }, []);
  const summary = data?.summary ?? {
    datasets: 0,
    active_jobs: 0,
    agents: 0,
    cdr_events: 0,
    ipdr_sessions: 0,
    relationship_edges: 0,
    models: 0,
  };
  return (
    <NexusShell
      title="Telecom intelligence, in one connected workspace."
      eyebrow="COMMAND CENTRE"
      actions={
        <Link className="button button-primary" href="/pipeline">
          <GitBranch size={16} /> Run full pipeline
        </Link>
      }
    >
      <section className="command-hero">
        <div className="command-copy">
          <span className="hero-chip">
            <i /> {data ? "Local engine ready" : error ? "Interface preview · local engine offline" : "Connecting to local engine"}
          </span>
          <h2>From raw telecom schemas to explainable relationship models.</h2>
          <p>
            Generate synthetic India-wide data, correlate IPDR behaviour, inspect
            communities, and train the neural network from the same immutable run.
          </p>
          <div className="hero-actions">
            <Link className="button button-primary" href="/pipeline">
              Start an end-to-end run <ArrowRight size={16} />
            </Link>
            <Link className="button button-ghost" href="/datasets">
              <Database size={16} /> Add a dataset
            </Link>
          </div>
        </div>
        <div className="signal-orbit" aria-hidden="true">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <span className="signal-core">NI</span>
          <i className="node n1" />
          <i className="node n2" />
          <i className="node n3" />
          <i className="node n4" />
        </div>
      </section>

      {error ? (
        <div className="inline-warning">
          The hosted interface cannot access files or run local GPU training. Open the
          local Nexus India app for live datasets, pipelines, downloads, and predictions;
          the bundled India explorer remains available here as a product preview.
        </div>
      ) : null}

      <section className="summary-grid">
        <div><Users size={17} /><span>Agents</span><strong>{formatIndianNumber(summary.agents)}</strong></div>
        <div><PhoneCall size={17} /><span>CDR events</span><strong>{formatIndianNumber(summary.cdr_events)}</strong></div>
        <div><Wifi size={17} /><span>IPDR sessions</span><strong>{formatIndianNumber(summary.ipdr_sessions)}</strong></div>
        <div><Network size={17} /><span>Evidence edges</span><strong>{formatIndianNumber(summary.relationship_edges)}</strong></div>
        <div><BrainCircuit size={17} /><span>Models</span><strong>{formatIndianNumber(summary.models)}</strong></div>
      </section>

      <section className="module-section">
        <div className="section-title">
          <div>
            <span className="nexus-eyebrow">WORKSPACES</span>
            <h2>Choose where to work</h2>
          </div>
          <span>{summary.active_jobs} active job{summary.active_jobs === 1 ? "" : "s"}</span>
        </div>
        <div className="module-grid">
          {modules.map((module) => {
            const Icon = module.icon;
            return (
              <Link href={module.href} className={`module-card ${module.tone}`} key={module.href}>
                <div className="module-icon"><Icon size={22} /></div>
                <span className="module-metric">{module.metric}</span>
                <h3>{module.title}</h3>
                <p>{module.copy}</p>
                <span className="module-link">Open workspace <ArrowRight size={14} /></span>
              </Link>
            );
          })}
        </div>
      </section>

      <section className="home-lower-grid">
        <div className="nexus-panel recent-runs">
          <div className="panel-heading">
            <div><span className="nexus-eyebrow">RUN HISTORY</span><strong>Recent activity</strong></div>
            <Link href="/datasets">View all</Link>
          </div>
          <div className="run-list">
            {(data?.runs ?? []).slice(0, 6).map((run) => (
              <Link href={`/datasets?run=${run.id}`} className="run-list-row" key={run.id}>
                <span className={`status-dot ${run.status}`} />
                <div><strong>{run.kind}</strong><small>{run.stage} · {formatDateTime(run.updated_at)}</small></div>
                <span>{Math.round(run.progress)}%</span>
              </Link>
            ))}
            {!data?.runs?.length ? <div className="empty-state">No pipeline runs yet.</div> : null}
          </div>
        </div>
        <div className="nexus-panel model-card-panel">
          <div className="panel-heading">
            <div><span className="nexus-eyebrow">ACTIVE MODEL</span><strong>Relationship classifier</strong></div>
            <BrainCircuit size={20} />
          </div>
          {data?.latest_model ? (
            <>
              <div className="model-id">{data.latest_model.id}</div>
              <div className="model-status-row">
                <span className={`status-badge ${data.latest_model.status}`}>{data.latest_model.status}</span>
                <span>{data.latest_model.active ? "Production selection" : "Latest candidate"}</span>
              </div>
              <Link className="button button-ghost full" href="/models">Open model studio <ArrowRight size={15} /></Link>
            </>
          ) : (
            <>
              <p>No model has been trained in the unified workspace yet.</p>
              <Link className="button button-ghost full" href="/models">Train first model <ArrowRight size={15} /></Link>
            </>
          )}
        </div>
      </section>

      <div className="evidence-notice">
        <Network size={18} />
        <div>
          <strong>Evidence has provenance.</strong>
          <span>{data?.privacy_notice ?? "IPDR similarity is a soft behavioural signal, not proof of friendship."}</span>
        </div>
      </div>
    </NexusShell>
  );
}
