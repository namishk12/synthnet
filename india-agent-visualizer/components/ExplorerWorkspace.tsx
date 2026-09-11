"use client";

import { ArrowLeft, Map, RotateCcw } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { nexusFetch, type RunRecord } from "../lib/nexus-api";
import { IndiaAgentDashboard } from "./IndiaAgentDashboard";

export function ExplorerWorkspace() {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [runId, setRunId] = useState("");
  useEffect(() => {
    nexusFetch<RunRecord[]>("/runs").then((items) => {
      const generated = items.filter((run) => run.kind === "generator" && run.status === "complete");
      setRuns(generated);
      setRunId(generated[0]?.id ?? "");
    }).catch(() => undefined);
  }, []);
  return (
    <div className="explorer-workspace">
      <div className="explorer-launchbar">
        <Link href="/" className="button button-ghost small"><ArrowLeft size={15} /> Command centre</Link>
        <div><Map size={16} /><strong>India Network Explorer</strong><span>Complete incoming and outgoing agent activity</span></div>
        <label>
          <span>Dataset run</span>
          <select value={runId} onChange={(event) => setRunId(event.target.value)}>
            <option value="">Bundled 100-agent sample</option>
            {runs.map((run) => <option value={run.id} key={run.id}>{run.id}</option>)}
          </select>
        </label>
        <button type="button" className="button button-ghost small" onClick={() => window.location.reload()}><RotateCcw size={14} /> Reload</button>
      </div>
      <IndiaAgentDashboard key={runId || "sample"} runId={runId} />
    </div>
  );
}
