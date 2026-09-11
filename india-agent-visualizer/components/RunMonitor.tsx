"use client";

import {
  AlertTriangle,
  Check,
  Circle,
  Clock3,
  Download,
  LoaderCircle,
  RotateCcw,
  Square,
  X,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import {
  artifactDownloadUrl,
  formatDateTime,
  formatIndianNumber,
  nexusFetch,
  type RunRecord,
} from "../lib/nexus-api";

function StageIcon({ status }: { status: string }) {
  if (status === "complete") return <Check size={15} />;
  if (status === "running") return <LoaderCircle className="spin" size={15} />;
  if (status === "failed") return <X size={15} />;
  return <Circle size={12} />;
}

function TrainingLossChart({ lines }: { lines: string[] }) {
  const points = lines.flatMap((line) => {
    const match = line.match(
      /Epoch\s+(\d+)\/(\d+):\s+train loss=([0-9.]+),\s+validation loss=([0-9.]+)/i,
    );
    return match
      ? [{
        epoch: Number(match[1]),
        total: Number(match[2]),
        train: Number(match[3]),
        validation: Number(match[4]),
      }]
      : [];
  });
  if (!points.length) return null;
  const width = 520;
  const height = 150;
  const padding = 18;
  const values = points.flatMap((point) => [point.train, point.validation]);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const range = Math.max(maximum - minimum, 1e-6);
  const maxEpoch = Math.max(...points.map((point) => point.total), 1);
  const coordinates = (key: "train" | "validation") => points.map((point) => {
    const x = padding + (point.epoch / maxEpoch) * (width - padding * 2);
    const y = height - padding - ((point[key] - minimum) / range) * (height - padding * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const latest = points.at(-1);
  return (
    <div className="training-loss-chart">
      <div>
        <span className="nexus-eyebrow">LIVE LOSS CURVE</span>
        <strong>Epoch {latest?.epoch} / {latest?.total}</strong>
        <small><i className="train" /> train {latest?.train.toFixed(4)} <i className="validation" /> validation {latest?.validation.toFixed(4)}</small>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training and validation loss chart">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <polyline className="train" points={coordinates("train")} />
        <polyline className="validation" points={coordinates("validation")} />
      </svg>
    </div>
  );
}

export function RunMonitor({
  runId,
  onUpdate,
  compact = false,
}: {
  runId: string;
  onUpdate?: (run: RunRecord) => void;
  compact?: boolean;
}) {
  const [run, setRun] = useState<RunRecord | null>(null);
  const [error, setError] = useState("");
  const refresh = useCallback(async () => {
    if (!runId) return;
    try {
      const payload = await nexusFetch<RunRecord>(`/runs/${runId}`);
      setRun(payload);
      setError("");
      onUpdate?.(payload);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : String(requestError));
    }
  }, [onUpdate, runId]);

  const shouldPoll = !run || ["queued", "running"].includes(run.status);
  useEffect(() => {
    let disposed = false;
    queueMicrotask(() => {
      if (!disposed) void refresh();
    });
    if (!shouldPoll) return () => {
      disposed = true;
    };
    const timer = window.setInterval(() => {
      void refresh();
    }, 1_500);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [refresh, shouldPoll]);

  async function cancel() {
    await nexusFetch(`/runs/${runId}/cancel`, { method: "POST" });
    await refresh();
  }

  async function retry() {
    const next = await nexusFetch<RunRecord>(`/runs/${runId}/retry`, { method: "POST" });
    window.location.assign(`/datasets?run=${next.id}`);
  }

  if (error) {
    return <div className="inline-error"><AlertTriangle size={16} /> {error}</div>;
  }
  if (!run) {
    return <div className="monitor-loading"><LoaderCircle className="spin" /> Loading run…</div>;
  }
  const active = ["queued", "running"].includes(run.status);
  return (
    <section className={`run-monitor ${compact ? "compact" : ""}`}>
      <div className="run-monitor-head">
        <div>
          <span className={`status-dot ${run.status}`} />
          <div>
            <span className="nexus-eyebrow">{run.kind} · {run.id}</span>
            <strong>{run.stage}</strong>
            <small>{run.message}</small>
          </div>
        </div>
        <div className="run-head-actions">
          {active ? (
            <button className="button button-ghost danger" type="button" onClick={cancel}>
              <Square size={15} /> Cancel
            </button>
          ) : run.status === "failed" ? (
            <button className="button button-ghost" type="button" onClick={retry}>
              <RotateCcw size={15} /> Retry
            </button>
          ) : null}
          <span className={`status-badge ${run.status}`}>{run.status}</span>
        </div>
      </div>
      <div className="progress-track" aria-label={`${run.progress}% complete`}>
        <i style={{ width: `${Math.max(0, Math.min(100, run.progress))}%` }} />
      </div>
      <div className="run-progress-copy">
        <strong>{Math.round(run.progress)}%</strong>
        <span><Clock3 size={13} /> Updated {formatDateTime(run.updated_at)}</span>
      </div>
      {!compact ? (
        <>
          <div className="stage-grid">
            {(run.stages ?? []).map((stage) => (
              <div className={`stage-row ${stage.status}`} key={stage.name}>
                <span className="stage-icon"><StageIcon status={stage.status} /></span>
                <div>
                  <strong>{stage.name}</strong>
                  <small>
                    {stage.status === "complete"
                      ? `${formatIndianNumber(stage.output_count ?? 0)} outputs`
                      : stage.status === "running"
                        ? `${Math.round(stage.progress)}%`
                        : stage.error ?? stage.status}
                  </small>
                </div>
              </div>
            ))}
          </div>
          {run.error ? (
            <div className="inline-error"><AlertTriangle size={16} /> {run.error}</div>
          ) : null}
          <TrainingLossChart lines={run.log ?? []} />
          {(run.artifacts?.length ?? 0) > 0 ? (
            <div className="artifact-panel">
              <div className="panel-heading">
                <div>
                  <span className="nexus-eyebrow">ARTIFACTS</span>
                  <strong>{run.artifacts?.length} files ready</strong>
                </div>
              </div>
              <div className="artifact-list">
                {run.artifacts?.map((artifact) => (
                  <a
                    href={artifactDownloadUrl(artifact.id)}
                    className="artifact-row"
                    key={artifact.id}
                  >
                    <div>
                      <strong>{artifact.filename}</strong>
                      <small>{artifact.kind} · {formatIndianNumber(artifact.size)} bytes</small>
                    </div>
                    <Download size={16} />
                  </a>
                ))}
              </div>
            </div>
          ) : null}
          {(run.children?.length ?? 0) > 0 ? (
            <div className="artifact-panel">
              <div className="panel-heading">
                <div>
                  <span className="nexus-eyebrow">PIPELINE LINEAGE</span>
                  <strong>{run.children?.length} connected stage runs</strong>
                </div>
              </div>
              <div className="artifact-list">
                {run.children?.map((child) => (
                  <div className="artifact-row" key={child.id}>
                    <div>
                      <strong>{child.kind}</strong>
                      <small>{child.id} · {child.stage}</small>
                    </div>
                    <span className={`status-badge ${child.status}`}>{child.status}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
          {(run.log?.length ?? 0) > 0 ? (
            <details className="run-log">
              <summary>Run log ({run.log?.length} recent lines)</summary>
              <pre>{run.log?.join("\n")}</pre>
            </details>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
