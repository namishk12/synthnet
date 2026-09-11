"use client";

import { ChevronDown, Cpu, Database, MapPin, Play, RotateCcw, Settings2, Sparkles } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import {
  nexusFetch,
  type IdentityDecision,
  type IdentityValidation,
  type DatasetRecord,
  type RunRecord,
} from "../lib/nexus-api";
import { NexusShell } from "./NexusShell";
import { RunMonitor } from "./RunMonitor";

export function GeneratorWorkspace() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [agents, setAgents] = useState(100);
  const [messages, setMessages] = useState(100);
  const [stateMultiplier, setStateMultiplier] = useState(2);
  const [sameStateActivityFraction, setSameStateActivityFraction] = useState(0.9);
  const [stateCount, setStateCount] = useState("36");
  const [seed, setSeed] = useState(42);
  const [ganSteps, setGanSteps] = useState(600);
  const [ganBatchSize, setGanBatchSize] = useState(1024);
  const [candidateSampleSize, setCandidateSampleSize] = useState(100_000);
  const [inferredTopFraction, setInferredTopFraction] = useState(0.01);
  const [negativeTruthRatio, setNegativeTruthRatio] = useState(1);
  const [simulationDays, setSimulationDays] = useState(7);
  const [cityJitterScale, setCityJitterScale] = useState(1);
  const [populationCentresFile, setPopulationCentresFile] = useState("");
  const [correlationRunId, setCorrelationRunId] = useState("");
  const [relationshipPositiveThreshold, setRelationshipPositiveThreshold] = useState(0.7);
  const [device, setDevice] = useState("auto");
  const [outputMode, setOutputMode] = useState("clean");
  const [cleanCdr, setCleanCdr] = useState(true);
  const [schemaExports, setSchemaExports] = useState(true);
  const [identityAnswer, setIdentityAnswer] = useState("");
  const [generationMode, setGenerationMode] = useState("");
  const [identityConfirmation, setIdentityConfirmation] = useState(false);
  const [identityMappings, setIdentityMappings] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [runId, setRunId] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  function restoreIdentityState(dataset?: DatasetRecord) {
    const metadata = dataset?.metadata ?? {};
    const savedAnswer = metadata.identity_answer;
    const savedMode = metadata.generation_mode;
    if (metadata.identity_validation_status === "validated" && typeof savedAnswer === "string") {
      setIdentityAnswer(savedAnswer);
      setGenerationMode(typeof savedMode === "string" ? savedMode : "");
    } else {
      setIdentityAnswer("");
      setGenerationMode("");
    }
    setIdentityConfirmation(false);
    setIdentityMappings("");
  }

  useEffect(() => {
    nexusFetch<DatasetRecord[]>("/datasets").then((items) => {
      setDatasets(items);
      const firstDataset = items[0];
      setDatasetId(firstDataset?.id || "");
      restoreIdentityState(firstDataset);
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
        throw new Error("Choose how the CDR and IPDR subjects relate before generation.");
      }
      if (needsIdentityReview && !generationMode) {
        throw new Error("Choose a generation or correlation mode after reviewing the identity evidence.");
      }
      if (needsIdentityReview && !identityConfirmation) {
        throw new Error("Review the full-file identity evidence and confirm the review before generation.");
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
      const run = await nexusFetch<RunRecord>("/runs/generator", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: datasetId,
          agent_count: agents,
          messages_per_agent: messages,
          same_state_multiplier: stateMultiplier,
          same_state_activity_fraction: sameStateActivityFraction,
          state_count: stateCount === "random" ? null : Number(stateCount),
          seed,
          output_mode: outputMode,
          device,
          gan_steps: ganSteps,
          gan_batch_size: ganBatchSize,
          candidate_sample_size: candidateSampleSize,
          inferred_top_fraction: inferredTopFraction,
          negative_truth_ratio: negativeTruthRatio,
          simulation_days: simulationDays,
          city_jitter_scale: cityJitterScale,
          population_centres_file: populationCentresFile.trim() || null,
          clean_cdr: cleanCdr,
          include_schema_exports: schemaExports,
          correlation_run_id: correlationRunId.trim() || null,
          relationship_positive_threshold: relationshipPositiveThreshold,
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

  function resetAdvancedConfiguration() {
    setGanBatchSize(1024);
    setCandidateSampleSize(100_000);
    setSameStateActivityFraction(0.9);
    setStateCount("36");
    setInferredTopFraction(0.01);
    setNegativeTruthRatio(1);
    setSimulationDays(7);
    setCityJitterScale(1);
    setPopulationCentresFile("");
    setCorrelationRunId("");
    setRelationshipPositiveThreshold(0.7);
  }

  return (
    <NexusShell
      title="Synthetic Data Generator"
      eyebrow="CONDITIONAL GRAPH GAN"
      actions={<span className="synthetic-badge"><Sparkles size={14} /> Synthetic output</span>}
    >
      <div className="workspace-intro">
        <div>
          <h2>Generate population-aware telecom behaviour.</h2>
          <p>
            Source CDR/IPDR distributions condition the simulation. Faker identities,
            India-wide city placement, and relationship provenance remain explicit.
          </p>
        </div>
        <div className="intro-proof">
          <span><MapPin size={16} /> 28 states + 8 UTs</span>
          <span><Cpu size={16} /> CPU/GPU training</span>
          <span><Database size={16} /> CSV, Excel & ZIP</span>
        </div>
      </div>
      <div className="two-column-workspace">
        <form className="nexus-panel config-panel" onSubmit={submit}>
          <div className="panel-heading">
            <div><span className="nexus-eyebrow">RUN CONFIGURATION</span><strong>Generation controls</strong></div>
            <Settings2 size={19} />
          </div>
          <label className="field full">
            <span>Conditioning dataset</span>
            <select value={datasetId} onChange={(event) => handleDatasetChange(event.target.value)} required>
              {datasets.map((dataset) => (
                <option value={dataset.id} key={dataset.id}>{dataset.name} · {dataset.kind}</option>
              ))}
            </select>
            <small>CDR is required. IPDR is used automatically when available.</small>
          </label>
          {needsIdentityReview ? (
            <section className="identity-gate generator-identity-gate" aria-labelledby="generator-identity-question">
              <span className="nexus-eyebrow">IDENTITY DECISION CONTROLS PROCESSING</span>
              <h3 id="generator-identity-question">Do these CDR and IPDR files belong to the same person?</h3>
              <p>This determines how calling and internet activity are combined. No answer is preselected. Identifier agreement supports matching; it does not prove personal identity.</p>
              <div className="identity-radio-grid" role="radiogroup" aria-label="CDR and IPDR identity relationship">
                {[
                  ["same_person", "Same person"],
                  ["different_people", "Different people"],
                  ["multiple_person_dataset", "Multiple-person dataset"],
                  ["not_sure", "Not sure"],
                ].map(([value, label]) => (
                  <label key={value}><input type="radio" name="generator_identity_answer" value={value} checked={identityAnswer === value} onChange={() => setIdentityAnswer(value)} /> {label}</label>
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
                    <div><span>Matched identifiers</span><strong>{selectedValidation.matching?.matched_count ?? 0}</strong></div>
                    <div><span>Unmatched CDR / IPDR</span><strong>{selectedValidation.matching?.unmatched_cdr_count ?? 0} / {selectedValidation.matching?.unmatched_ipdr_count ?? 0}</strong></div>
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
            <section className="identity-gate generator-identity-gate" aria-labelledby="generator-cdr-only-question">
              <span className="nexus-eyebrow">EXPLICIT SINGLE-SOURCE MODE</span>
              <h3 id="generator-cdr-only-question">Choose how to generate from this one-person CDR source</h3>
              <p>CDR-only generation remains available for multi-subject inputs. This one-subject run needs the explicit template mode, so no profile is copied implicitly.</p>
              <label className="field"><span>Generation mode</span><select value={generationMode} onChange={(event) => setGenerationMode(event.target.value)}><option value="">Choose a mode</option><option value="single_person_template">Explicit single-person template expansion</option></select></label>
              {selectedValidation ? <p className="identity-evidence-copy">Full-file validation found {cdrSubjectCount} CDR subject. Generated identities remain distinct and retain source lineage.</p> : null}
            </section>
          ) : null}
          <div className="field-grid">
            <label className="field">
              <span>Agents</span>
              <input type="number" min={4} max={2000} value={agents} onChange={(event) => setAgents(Number(event.target.value))} />
              <small>4–2,000 core agents</small>
            </label>
            <label className="field">
              <span>Events per agent</span>
              <input type="number" min={1} max={2000} value={messages} onChange={(event) => setMessages(Number(event.target.value))} />
              <small>Applied to CDR and IPDR</small>
            </label>
            <label className="field">
              <span>Same-state multiplier</span>
              <input type="number" min={1} max={10} step={0.1} value={stateMultiplier} onChange={(event) => setStateMultiplier(Number(event.target.value))} />
              <small>Synthetic prior only</small>
            </label>
            <label className="field">
              <span>Reproducibility seed</span>
              <input type="number" min={0} value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </label>
            <label className="field">
              <span>GAN training steps</span>
              <input type="number" min={1} max={20000} value={ganSteps} onChange={(event) => setGanSteps(Number(event.target.value))} />
            </label>
            <label className="field">
              <span>Compute device</span>
              <select value={device} onChange={(event) => setDevice(event.target.value)}>
                <option value="auto">Auto</option>
                <option value="cpu">CPU</option>
                <option value="cuda">CUDA GPU</option>
              </select>
            </label>
            <label className="field">
              <span>Output mode</span>
              <select value={outputMode} onChange={(event) => setOutputMode(event.target.value)}>
                <option value="clean">Clean</option>
                <option value="normal">Normal</option>
                <option value="both">Both</option>
              </select>
            </label>
          </div>
          <div className="switch-stack">
            <label className="switch-row">
              <input type="checkbox" checked={cleanCdr} onChange={(event) => setCleanCdr(event.target.checked)} />
              <span><strong>Clean CDR</strong><small>Only removes rows where resolved Other_Party_No contains letters.</small></span>
            </label>
            <label className="switch-row">
              <input type="checkbox" checked={schemaExports} onChange={(event) => setSchemaExports(event.target.checked)} />
              <span><strong>Full schema exports</strong><small>Retain complete nodal-office CDR and IPDR layouts.</small></span>
            </label>
          </div>
          <section className={`advanced-config ${advancedOpen ? "is-open" : ""}`} aria-label="Advanced CGAN configuration">
            <div className="advanced-config-header">
              <div>
                <span className="nexus-eyebrow">ADVANCED CONFIGURATION</span>
                <strong>All generator fields</strong>
                <small>Every supported GeneratorConfig value is available here. Defaults preserve the validated workflow.</small>
              </div>
              <button
                className="button button-ghost small advanced-toggle"
                type="button"
                aria-expanded={advancedOpen}
                aria-controls="advanced-generator-fields"
                onClick={() => setAdvancedOpen((current) => !current)}
              >
                {advancedOpen ? "Hide fields" : "Show fields"}
                <ChevronDown size={14} style={{ transform: advancedOpen ? "rotate(180deg)" : undefined }} />
              </button>
            </div>
            {advancedOpen ? (
              <div id="advanced-generator-fields" className="advanced-config-body">
                <div className="advanced-config-note">
                  <Settings2 size={15} />
                  <span>These values map directly to the local CGAN generator API. Empty optional paths use the built-in India population matrix and automatic correlation lineage.</span>
                </div>
                <div className="field-grid">
                  <label className="field">
                    <span>GAN batch size</span>
                    <input type="number" min={16} max={65536} value={ganBatchSize} onChange={(event) => setGanBatchSize(Number(event.target.value))} />
                    <small>16–65,536 samples per training batch</small>
                  </label>
                  <label className="field">
                    <span>Candidate sample size</span>
                    <input type="number" min={100} max={10_000_000} value={candidateSampleSize} onChange={(event) => setCandidateSampleSize(Number(event.target.value))} />
                    <small>100–10,000,000 candidate relationships</small>
                  </label>
                  <label className="field">
                    <span>Inferred top fraction</span>
                    <input type="number" min={0.0001} max={0.25} step={0.001} value={inferredTopFraction} onChange={(event) => setInferredTopFraction(Number(event.target.value))} />
                    <small>Top 0–25% of scored candidates become soft edges</small>
                  </label>
                  <label className="field">
                    <span>Negative truth ratio</span>
                    <input type="number" min={0} max={20} step={0.1} value={negativeTruthRatio} onChange={(event) => setNegativeTruthRatio(Number(event.target.value))} />
                    <small>Negative examples sampled per positive example</small>
                  </label>
                  <label className="field">
                    <span>Simulation days</span>
                    <input type="number" min={1} max={365} value={simulationDays} onChange={(event) => setSimulationDays(Number(event.target.value))} />
                    <small>1–365 days of synthetic activity</small>
                  </label>
                  <label className="field">
                    <span>City jitter scale</span>
                    <input type="number" min={0} max={10} step={0.1} value={cityJitterScale} onChange={(event) => setCityJitterScale(Number(event.target.value))} />
                    <small>0 keeps agents at city centres; 10 widens the offset</small>
                  </label>
                  <label className="field">
                    <span>Within-state activity share <b>{Math.round(sameStateActivityFraction * 100)}%</b></span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={sameStateActivityFraction}
                      onChange={(event) => setSameStateActivityFraction(Number(event.target.value))}
                      aria-label="Within-state activity share"
                    />
                    <small>Target share of synthetic-to-synthetic CDR contacts whose receiver is in the caller&apos;s state.</small>
                  </label>
                  <label className="field">
                    <span>States / UTs represented</span>
                    <select value={stateCount} onChange={(event) => setStateCount(event.target.value)}>
                      <option value="random">Random count (seeded)</option>
                      {Array.from({ length: 36 }, (_, index) => index + 1).map((count) => (
                        <option value={String(count)} key={count}>
                          {count === 36 ? "All 36 states / UTs" : `${count} states / UTs`}
                        </option>
                      ))}
                    </select>
                    <small>Fixed count guarantees representation when there are enough agents; Random is reproducible from the seed.</small>
                  </label>
                  <label className="field">
                    <span>Relationship positive threshold</span>
                    <input type="number" min={0} max={1} step={0.01} value={relationshipPositiveThreshold} onChange={(event) => setRelationshipPositiveThreshold(Number(event.target.value))} />
                    <small>IPDR/CGAN evidence threshold for probable-positive labels</small>
                  </label>
                </div>
                <div className="field-grid advanced-path-grid">
                  <label className="field">
                    <span>Population centres file <b>optional</b></span>
                    <input type="text" value={populationCentresFile} onChange={(event) => setPopulationCentresFile(event.target.value)} placeholder="data/india_population_centres.csv" />
                    <small>Local CSV override; blank uses the bundled India city/town matrix.</small>
                  </label>
                  <label className="field">
                    <span>Correlation run ID <b>optional</b></span>
                    <input type="text" value={correlationRunId} onChange={(event) => setCorrelationRunId(event.target.value)} placeholder="RUN_..." />
                    <small>Reuse a completed IPDR correlation run as conditioning lineage.</small>
                  </label>
                </div>
                <div className="advanced-config-footer">
                  <span><i className="amber" /> 21 API fields exposed · optional overrides stay local</span>
                  <button className="button button-ghost small" type="button" onClick={resetAdvancedConfiguration}>
                    <RotateCcw size={13} /> Reset advanced defaults
                  </button>
                </div>
              </div>
            ) : null}
          </section>
          {error ? <div className="inline-error">{error}</div> : null}
          <button className="button button-primary full" disabled={submitting || !datasetId || (needsIdentityReview && (!identityAnswer || !generationMode || !identityConfirmation)) || (needsSinglePersonMode && !generationMode)} type="submit">
            <Play size={16} /> {submitting ? "Starting…" : `Generate ${agents.toLocaleString("en-IN")} agents`}
          </button>
        </form>
        <div className="monitor-column">
          {runId ? (
            <RunMonitor runId={runId} />
          ) : (
            <div className="nexus-panel empty-workspace">
              <Sparkles size={26} />
              <h3>No generation run selected</h3>
              <p>Configure the run and start it. Progress, logs, and every downloadable artifact will appear here.</p>
              <div className="provenance-stack">
                <span><i className="amber" /> Faker identity fields</span>
                <span><i className="cyan" /> Source-conditioned distributions</span>
                <span><i className="violet" /> GAN-inferred soft edges</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </NexusShell>
  );
}
