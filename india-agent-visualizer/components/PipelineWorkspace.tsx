"use client";

import {
  BrainCircuit,
  Check,
  GitBranch,
  Network,
  Play,
  RadioTower,
  Sparkles,
} from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { nexusFetch, type DatasetRecord, type IdentityDecision, type IdentityValidation, type RunRecord } from "../lib/nexus-api";
import { NexusShell } from "./NexusShell";
import { RunMonitor } from "./RunMonitor";

const pipeline = [
  { label: "Schema validation", icon: Check },
  { label: "IPDR correlation", icon: RadioTower },
  { label: "IPDR-conditioned CGAN", icon: Sparkles },
  { label: "Community intelligence", icon: Network },
  { label: "Neural-network training", icon: BrainCircuit },
];

const identityAnswers = [
  ["same_person", "Same person"],
  ["different_people", "Different people"],
  ["multiple_person_dataset", "Multiple-person dataset"],
  ["not_sure", "Not sure"],
] as const;

export function PipelineWorkspace() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [agents, setAgents] = useState(100);
  const [messages, setMessages] = useState(100);
  const [epochs, setEpochs] = useState(100);
  const [seed, setSeed] = useState(42);
  const [trainModel, setTrainModel] = useState(true);
  const [runId, setRunId] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [identityAnswer, setIdentityAnswer] = useState("");
  const [generationMode, setGenerationMode] = useState("");
  const [identityConfirmation, setIdentityConfirmation] = useState(false);
  const [identityMappings, setIdentityMappings] = useState("");

  function restoreIdentityState(dataset?: DatasetRecord) {
    const metadata = dataset?.metadata ?? {};
    if (metadata.identity_validation_status === "validated" && typeof metadata.identity_answer === "string") {
      setIdentityAnswer(metadata.identity_answer);
      setGenerationMode(typeof metadata.generation_mode === "string" ? metadata.generation_mode : "");
    } else {
      setIdentityAnswer("");
      setGenerationMode("");
    }
    setIdentityConfirmation(false);
    setIdentityMappings("");
  }

  useEffect(() => {
    Promise.all([
      nexusFetch<DatasetRecord[]>("/datasets"),
      nexusFetch<RunRecord[]>("/runs"),
    ]).then(([items, runs]) => {
      const usableDatasets = items.filter((item) => item.status !== "trashed");
      setDatasets(usableDatasets);
      const firstDataset = usableDatasets[0];
      setDatasetId(firstDataset?.id ?? "");
      restoreIdentityState(firstDataset);
      const requestedRun = new URLSearchParams(window.location.search).get("run");
      const activePipeline = runs.find(
        (run) => run.kind === "pipeline" && ["queued", "running"].includes(run.status),
      );
      setRunId(requestedRun ?? activePipeline?.id ?? "");
    }).catch((reason: Error) => setError(reason.message));
  }, []);

  function handleDatasetChange(nextId: string) {
    setDatasetId(nextId);
    restoreIdentityState(datasets.find((item) => item.id === nextId));
  }

  const selectedDataset = datasets.find((item) => item.id === datasetId);
  const hasCdr = Number(selectedDataset?.row_counts?.cdr_compact ?? selectedDataset?.row_counts?.cdr ?? 0) > 0;
  const hasIpdr = Number(selectedDataset?.row_counts?.ipdr_compact ?? selectedDataset?.row_counts?.ipdr ?? 0) > 0;
  const needsIdentityReview = hasCdr && hasIpdr;
  const selectedValidation = selectedDataset?.metadata?.identity_validation as IdentityValidation | undefined;
  const selectedDecision = selectedDataset?.metadata?.identity_decision as IdentityDecision | undefined;
  const cdrSubjectCount = Number(selectedValidation?.sources?.cdr?.subject_count ?? 0);
  const needsSinglePersonMode = hasCdr && !hasIpdr && cdrSubjectCount === 1;

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      if (needsIdentityReview && !identityAnswer) {
        throw new Error("Choose how the CDR and IPDR subjects relate before running the pipeline.");
      }
      if (needsIdentityReview && !generationMode) {
        throw new Error("Choose a processing mode after reviewing the identity evidence.");
      }
      if (needsIdentityReview && !identityConfirmation) {
        throw new Error("Review the full-file identity evidence and confirm the review before running the pipeline.");
      }
      if (needsSinglePersonMode && !generationMode) {
        throw new Error("Choose CDR-only generation or the explicit single-person template mode.");
      }
      let parsedMappings: unknown = null;
      if (identityMappings.trim()) {
        try {
          parsedMappings = JSON.parse(identityMappings);
        } catch {
          throw new Error("Explicit subscriber/alias mappings must be valid JSON.");
        }
      }
      const run = await nexusFetch<RunRecord>("/runs/pipeline", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: datasetId,
          agent_count: agents,
          messages_per_agent: messages,
          same_state_multiplier: 2,
          seed,
          device: "auto",
          output_mode: "clean",
          gan_steps: 600,
          candidate_sample_size: Math.max(100_000, agents * agents * 10),
          relationship_positive_threshold: 0.70,
          graph_threshold: 0.40,
          temporal_window_seconds: 300,
          train_model: trainModel,
          epochs,
          batch_size: 64,
          learning_rate: 0.001,
          identity_answer: needsIdentityReview ? identityAnswer : null,
          generation_mode: needsIdentityReview || needsSinglePersonMode ? generationMode : "cdr_only",
          identity_confirmation: needsIdentityReview && identityConfirmation,
          identity_mappings: parsedMappings,
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
      title="Run Full Pipeline"
      eyebrow="END-TO-END ORCHESTRATION"
      actions={<span className="pipeline-pill"><GitBranch size={14} /> Resumable stages</span>}
    >
      <section className="pipeline-rail" aria-label="Full pipeline sequence">
        {pipeline.map((item, index) => {
          const Icon = item.icon;
          return (
            <div key={item.label}>
              <span><Icon size={17} /></span>
              <strong>{item.label}</strong>
              {index < pipeline.length - 1 ? <i /> : null}
            </div>
          );
        })}
      </section>
      <div className="two-column-workspace pipeline-layout">
        <form className="nexus-panel config-panel" onSubmit={submit}>
          <div className="panel-heading">
            <div><span className="nexus-eyebrow">PIPELINE INPUT</span><strong>One run, complete lineage</strong></div>
            <GitBranch size={19} />
          </div>
          <label className="field full">
            <span>Source dataset</span>
            <select value={datasetId} onChange={(event) => handleDatasetChange(event.target.value)} required>
              {datasets.map((dataset) => <option value={dataset.id} key={dataset.id}>{dataset.name}</option>)}
            </select>
          </label>
          {needsIdentityReview ? (
            <section className="identity-gate generator-identity-gate" aria-labelledby="pipeline-identity-question">
              <span className="nexus-eyebrow">IDENTITY DECISION CONTROLS PROCESSING</span>
              <h3 id="pipeline-identity-question">Do these CDR and IPDR files belong to the same person?</h3>
              <p>This determines how calling and internet activity are combined. No answer is preselected. Identifier agreement supports matching; it does not prove personal identity.</p>
              <div className="identity-radio-grid" role="radiogroup" aria-label="CDR and IPDR identity relationship">
                {identityAnswers.map(([value, label]) => (
                  <label key={value}><input type="radio" name="pipeline_identity_answer" value={value} checked={identityAnswer === value} onChange={() => setIdentityAnswer(value)} /> {label}</label>
                ))}
              </div>
              <div className="identity-gate-fields">
                <label className="field"><span>Generation / processing mode</span><select value={generationMode} onChange={(event) => setGenerationMode(event.target.value)}><option value="">Choose after evidence review</option><option value="combined">Combine one consistent subject mapping</option><option value="multi_person_conditioned">Match each subject independently</option><option value="single_person_template">Explicit single-person template expansion</option><option value="cdr_only">CDR-only generation</option><option value="ipdr_only_correlation">IPDR-only correlation</option><option value="separate_sources">Keep sources separate</option></select></label>
                <label className="field"><span>Explicit subscriber/alias mapping <b>optional</b></span><textarea value={identityMappings} onChange={(event) => setIdentityMappings(event.target.value)} placeholder='{"ipdr-id":"cdr-id"}' /></label>
              </div>
              {selectedValidation ? (
                <div className="identity-evidence">
                  <div className="identity-evidence-grid">
                    <div><span>CDR subject column</span><strong>{selectedValidation.sources?.cdr?.subject_column ?? "—"}</strong></div>
                    <div><span>IPDR subscriber column</span><strong>{selectedValidation.sources?.ipdr?.subscriber_column ?? selectedValidation.sources?.ipdr?.subject_column ?? "—"}</strong></div>
                    <div><span>CDR / IPDR subjects</span><strong>{selectedValidation.matching?.cdr_subject_count ?? 0} / {selectedValidation.matching?.ipdr_subscriber_count ?? 0}</strong></div>
                    <div><span>CDR contacts (not subjects)</span><strong>{selectedValidation.sources?.cdr?.contact_count ?? 0}</strong></div>
                    <div><span>Matched / unmatched</span><strong>{selectedValidation.matching?.matched_count ?? 0} / {(selectedValidation.matching?.unmatched_cdr_count ?? 0) + (selectedValidation.matching?.unmatched_ipdr_count ?? 0)}</strong></div>
                    <div><span>CDR observation period</span><strong>{selectedValidation.sources?.cdr?.date_range?.start ?? "—"} → {selectedValidation.sources?.cdr?.date_range?.end ?? "—"}</strong></div>
                    <div><span>IPDR observation period</span><strong>{selectedValidation.sources?.ipdr?.date_range?.start ?? "—"} → {selectedValidation.sources?.ipdr?.date_range?.end ?? "—"}</strong></div>
                    <div><span>Temporal status</span><strong>{selectedValidation.temporal?.status ?? "unavailable"}</strong></div>
                  </div>
                  <p className="identity-evidence-copy">Matched: {(selectedValidation.matching?.matched_identifiers ?? []).slice(0, 12).join(", ") || "none"}. Unmatched CDR: {(selectedValidation.matching?.unmatched_cdr_subjects ?? []).slice(0, 12).join(", ") || "none"}. Unmatched IPDR: {(selectedValidation.matching?.unmatched_ipdr_subscribers ?? []).slice(0, 12).join(", ") || "none"}. {selectedValidation.temporal?.explanation ?? "Temporal evidence unavailable."} {selectedValidation.identity_evidence?.statement ?? "Identifier agreement supports matching; it does not prove personal identity."} {(selectedDecision?.warnings ?? []).join(" ")} {(selectedDecision?.available_operations?.length ?? 0) > 0 ? `Available operations: ${selectedDecision?.available_operations?.join(", ")}.` : ""}</p>
                </div>
              ) : (
                <p className="identity-evidence-copy">This dataset has no saved full-file validation result yet. Existing datasets remain unknown until checked.</p>
              )}
              <label className="consent-row"><input type="checkbox" checked={identityConfirmation} onChange={(event) => setIdentityConfirmation(event.target.checked)} /> <span>I reviewed the detected evidence. This records review and cannot bypass contradictory data.</span></label>
            </section>
          ) : null}
          {needsSinglePersonMode ? (
            <section className="identity-gate generator-identity-gate" aria-labelledby="pipeline-cdr-only-question">
              <span className="nexus-eyebrow">EXPLICIT SINGLE-SOURCE MODE</span>
              <h3 id="pipeline-cdr-only-question">Choose how to run this one-person CDR source</h3>
              <p>CDR-only processing remains available for multi-subject inputs. This one-subject run needs the explicit template mode and preserves distinct synthetic identities with source lineage.</p>
              <label className="field"><span>Generation mode</span><select value={generationMode} onChange={(event) => setGenerationMode(event.target.value)}><option value="">Choose a mode</option><option value="single_person_template">Explicit single-person template expansion</option></select></label>
            </section>
          ) : null}
          <div className="field-grid">
            <label className="field">
              <span>Synthetic agents</span>
              <input type="number" min={4} max={2000} value={agents} onChange={(event) => setAgents(Number(event.target.value))} />
            </label>
            <label className="field">
              <span>Events per agent</span>
              <input type="number" min={1} max={2000} value={messages} onChange={(event) => setMessages(Number(event.target.value))} />
            </label>
            <label className="field">
              <span>Training epochs</span>
              <input type="number" min={1} max={5000} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} disabled={!trainModel} />
            </label>
            <label className="field">
              <span>Reproducibility seed</span>
              <input type="number" min={0} value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </label>
          </div>
          <label className="switch-row">
            <input type="checkbox" checked={trainModel} onChange={(event) => setTrainModel(event.target.checked)} />
            <span><strong>Train neural network automatically</strong><small>The generated artifacts become an immutable training snapshot.</small></span>
          </label>
          <div className="pipeline-assumptions">
            <span><Check size={14} /> rDNS remains offline unless separately consented</span>
            <span><Check size={14} /> IPDR edges remain soft inferred evidence</span>
            <span><Check size={14} /> Complete run lineage and ZIP manifest</span>
          </div>
          {error ? <div className="inline-error">{error}</div> : null}
          <button className="button button-primary full" disabled={!datasetId || submitting || (needsIdentityReview && (!identityAnswer || !generationMode || !identityConfirmation)) || (needsSinglePersonMode && !generationMode)} type="submit">
            <Play size={16} /> {submitting ? "Starting pipeline…" : "Run complete pipeline"}
          </button>
        </form>
        <div className="monitor-column">
          {runId ? (
            <RunMonitor runId={runId} />
          ) : (
            <div className="nexus-panel empty-workspace pipeline-empty">
              <GitBranch size={28} />
              <h3>The pipeline is ready</h3>
              <p>
                Source IPDR evidence conditions the graph GAN. Generated records
                then feed community analysis and model training without manual downloads.
              </p>
            </div>
          )}
        </div>
      </div>
    </NexusShell>
  );
}
