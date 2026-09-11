"use client";

import {
  Activity,
  BrainCircuit,
  Check,
  Cpu,
  Download,
  FileSpreadsheet,
  Gauge,
  Play,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  artifactDownloadUrl,
  formatDateTime,
  nexusFetch,
  type ArtifactRecord,
  type ModelRecord,
  type RunRecord,
} from "../lib/nexus-api";
import { NexusShell } from "./NexusShell";
import { RunMonitor } from "./RunMonitor";

type BatchPredictionResult = {
  row_count: number;
  artifact: ArtifactRecord;
  preview: Array<Record<string, string>>;
};

type TrainingPreview = {
  rows: number;
  class_counts: {
    positive: number;
    negative: number;
    unresolved: number;
  };
  label_sources: Record<string, number>;
  feature_names: string[];
  warnings: string[];
  snapshot_file: string;
};

type PredictionEvidence = {
  feature: string;
  value: number;
  relative_input_weight: number;
  note: string;
};

type ModelReviewResult = {
  kind: string;
  total: number;
  rows: Array<Record<string, string>>;
};

function findMetric(metrics: Record<string, unknown>, name: string): number | null {
  const direct = metrics[name];
  if (typeof direct === "number") return direct;
  for (const sectionName of ["test", "held_out_test", "validation"]) {
    const section = metrics[sectionName];
    if (section && typeof section === "object") {
      const value = (section as Record<string, unknown>)[name];
      if (typeof value === "number") return value;
    }
  }
  return null;
}

export function ModelsWorkspace() {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [sourceRunId, setSourceRunId] = useState("");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.001);
  const [seed, setSeed] = useState(42);
  const [device, setDevice] = useState("auto");
  const [modelType, setModelType] = useState("compare");
  const [labelPolicy, setLabelPolicy] = useState("all_weighted");
  const [hiddenLayers, setHiddenLayers] = useState("64,32");
  const [dropout, setDropout] = useState(0.2);
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(20);
  const [thresholdStrategy, setThresholdStrategy] = useState("macro_f1");
  const [splitStrategy, setSplitStrategy] = useState("group_disjoint");
  const [calibration, setCalibration] = useState("platt");
  const [treeBaseline, setTreeBaseline] = useState(true);
  const [trainingPreview, setTrainingPreview] = useState<TrainingPreview | null>(null);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [runId, setRunId] = useState("");
  const [personA, setPersonA] = useState("");
  const [personB, setPersonB] = useState("");
  const [prediction, setPrediction] = useState<Record<string, string> | null>(null);
  const [predictionEvidence, setPredictionEvidence] = useState<PredictionEvidence[]>([]);
  const [predicting, setPredicting] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchPredictionResult | null>(null);
  const [batchPredicting, setBatchPredicting] = useState(false);
  const [falsePositives, setFalsePositives] = useState<ModelReviewResult | null>(null);
  const [falseNegatives, setFalseNegatives] = useState<ModelReviewResult | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    const [runRows, modelRows] = await Promise.all([
      nexusFetch<RunRecord[]>("/runs"),
      nexusFetch<ModelRecord[]>("/models"),
    ]);
    const usable = runRows
      .filter(
        (run) => run.status === "complete"
          && (
            ["community", "pipeline"].includes(run.kind)
            || (
              run.kind === "generator"
              && Number(run.config.agent_count ?? 0) >= 10
            )
          ),
      )
      .sort((left, right) => {
        const rank = (run: RunRecord) => (
          run.kind === "community" ? 0 : run.kind === "pipeline" ? 1 : 2
        );
        return rank(left) - rank(right);
      });
    setRuns(usable);
    setModels(modelRows);
    setSourceRunId((current) => current || usable[0]?.id || "");
  }, []);
  useEffect(() => {
    queueMicrotask(() => {
      void refresh().catch((reason: Error) => setError(reason.message));
    });
  }, [refresh]);
  useEffect(() => {
    let disposed = false;
    if (!sourceRunId) {
      queueMicrotask(() => {
        if (!disposed) setTrainingPreview(null);
      });
      return () => {
        disposed = true;
      };
    }
    void nexusFetch<TrainingPreview>(`/runs/${sourceRunId}/training-preview`)
      .then((preview) => {
        if (disposed) return;
        setTrainingPreview(preview);
        setSelectedFeatures(preview.feature_names);
      })
      .catch((reason: Error) => {
        if (disposed) return;
        setTrainingPreview(null);
        setError(reason.message);
      });
    return () => {
      disposed = true;
    };
  }, [sourceRunId]);

  const activeModel = useMemo(
    () => models.find((model) => model.active) ?? models[0] ?? null,
    [models],
  );
  useEffect(() => {
    let disposed = false;
    if (!activeModel) {
      queueMicrotask(() => {
        if (disposed) return;
        setFalsePositives(null);
        setFalseNegatives(null);
      });
      return () => {
        disposed = true;
      };
    }
    void Promise.all([
      nexusFetch<ModelReviewResult>(`/models/${activeModel.id}/review-errors?kind=false_positive&limit=12`),
      nexusFetch<ModelReviewResult>(`/models/${activeModel.id}/review-errors?kind=false_negative&limit=12`),
    ]).then(([positiveRows, negativeRows]) => {
      if (disposed) return;
      setFalsePositives(positiveRows);
      setFalseNegatives(negativeRows);
    }).catch((reason: Error) => {
      if (disposed) return;
      setFalsePositives(null);
      setFalseNegatives(null);
      setError(reason.message);
    });
    return () => {
      disposed = true;
    };
  }, [activeModel]);
  const activeTestMetrics = (
    activeModel?.metrics.test && typeof activeModel.metrics.test === "object"
      ? activeModel.metrics.test as Record<string, unknown>
      : {}
  );
  const confusionMatrix = (
    activeTestMetrics.confusion_matrix
    && typeof activeTestMetrics.confusion_matrix === "object"
      ? activeTestMetrics.confusion_matrix as Record<string, number>
      : {}
  );
  const perLabelSource = (
    activeModel?.metrics.per_label_source
    && typeof activeModel.metrics.per_label_source === "object"
      ? activeModel.metrics.per_label_source as Record<string, Record<string, unknown>>
      : {}
  );

  async function train(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const parsedHiddenLayers = hiddenLayers
        .split(",")
        .map((value) => Number(value.trim()))
        .filter((value) => Number.isInteger(value));
      if (!parsedHiddenLayers.length || parsedHiddenLayers.some((value) => value < 2)) {
        throw new Error("Enter one or more hidden-layer widths, for example 64,32.");
      }
      const run = await nexusFetch<RunRecord>("/runs/training", {
        method: "POST",
        body: JSON.stringify({
          source_run_id: sourceRunId,
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          seed,
          device,
          model_type: modelType,
          label_policy: labelPolicy,
          selected_features: selectedFeatures,
          hidden_layers: parsedHiddenLayers,
          dropout,
          early_stopping_patience: earlyStoppingPatience,
          threshold_strategy: thresholdStrategy,
          split_strategy: splitStrategy,
          calibration,
          tree_baseline: treeBaseline,
        }),
      });
      setRunId(run.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function activate(modelId: string) {
    await nexusFetch(`/models/${modelId}/activate`, { method: "POST" });
    await refresh();
  }

  async function resumeTraining(modelId: string) {
    setError("");
    try {
      const run = await nexusFetch<RunRecord>(`/models/${modelId}/resume`, {
        method: "POST",
      });
      setRunId(run.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function predict(event: FormEvent) {
    event.preventDefault();
    if (!activeModel) return;
    setPrediction(null);
    setPredictionEvidence([]);
    setError("");
    setPredicting(true);
    try {
      const result = await nexusFetch<{
        prediction: Record<string, string>;
        top_evidence: PredictionEvidence[];
      }>(
        `/models/${activeModel.id}/predict`,
        {
          method: "POST",
          body: JSON.stringify({
            source_run_id: sourceRunId,
            person_a: personA,
            person_b: personB,
          }),
        },
      );
      setPrediction(result.prediction);
      setPredictionEvidence(result.top_evidence ?? []);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setPredicting(false);
    }
  }

  async function predictBatch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeModel) return;
    setBatchResult(null);
    setError("");
    setBatchPredicting(true);
    const form = new FormData(event.currentTarget);
    form.set("source_run_id", sourceRunId);
    try {
      const result = await nexusFetch<BatchPredictionResult>(
        `/models/${activeModel.id}/predict-batch`,
        { method: "POST", body: form },
      );
      setBatchResult(result);
      event.currentTarget.reset();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBatchPredicting(false);
    }
  }

  function toggleFeature(feature: string) {
    setSelectedFeatures((current) => (
      current.includes(feature)
        ? current.filter((value) => value !== feature)
        : [...current, feature]
    ));
  }

  return (
    <NexusShell
      title="Neural Network Studio"
      eyebrow="TRAIN · EVALUATE · ACTIVATE · PREDICT"
      actions={<span className="model-badge"><BrainCircuit size={14} /> PyTorch MLP</span>}
    >
      <section className="model-overview">
        <div className="model-overview-copy">
          <span className="nexus-eyebrow">RELATIONSHIP CLASSIFICATION</span>
          <h2>Train directly from a versioned pipeline run.</h2>
          <p>CDR, IPDR, identity mapping, pair labels, feature schema, and provenance stay linked to the model record.</p>
        </div>
        <div className="model-principles">
          <span><ShieldCheck size={16} /> Inferred labels remain soft</span>
          <span><Check size={16} /> Canonical pairs cannot cross splits</span>
          <span><Gauge size={16} /> Validation macro-F1 threshold</span>
        </div>
      </section>
      <div className="two-column-workspace">
        <form className="nexus-panel config-panel" onSubmit={train}>
          <div className="panel-heading"><div><span className="nexus-eyebrow">TRAINING CONFIG</span><strong>Weighted relationship classifier</strong></div><BrainCircuit size={19} /></div>
          <label className="field full"><span>Training source run</span><select value={sourceRunId} onChange={(event) => setSourceRunId(event.target.value)} required>{runs.map((run) => <option value={run.id} key={run.id}>{run.kind} · {run.id}</option>)}</select><small>Files are resolved from the run lineage automatically.</small></label>
          {trainingPreview ? (
            <div className="training-snapshot-preview">
              <div>
                <span className="nexus-eyebrow">IMMUTABLE SNAPSHOT</span>
                <strong>{trainingPreview.rows.toLocaleString("en-IN")} canonical pairs</strong>
                <small>{trainingPreview.snapshot_file}</small>
              </div>
              <div className="snapshot-class-counts">
                <span><b>{trainingPreview.class_counts.positive.toLocaleString("en-IN")}</b> positive</span>
                <span><b>{trainingPreview.class_counts.negative.toLocaleString("en-IN")}</b> negative</span>
              </div>
              <div className="snapshot-sources">
                {Object.entries(trainingPreview.label_sources).slice(0, 6).map(([source, count]) => (
                  <span key={source}>{source.replaceAll("_", " ")} <b>{count.toLocaleString("en-IN")}</b></span>
                ))}
              </div>
              {trainingPreview.warnings.map((warning) => <small className="snapshot-warning" key={warning}>{warning}</small>)}
            </div>
          ) : null}
          <div className="field-grid">
            <label className="field"><span>Epochs</span><input type="number" min={1} max={5000} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} /></label>
            <label className="field"><span>Batch size</span><input type="number" min={1} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} /></label>
            <label className="field"><span>Learning rate</span><input type="number" min={0.000001} max={1} step={0.0001} value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} /></label>
            <label className="field"><span>Seed</span><input type="number" min={0} value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label>
            <label className="field"><span>Device</span><select value={device} onChange={(event) => setDevice(event.target.value)}><option value="auto">Auto</option><option value="cpu">CPU</option><option value="cuda">CUDA</option></select></label>
            <label className="field"><span>Evaluation</span><select value={modelType} onChange={(event) => setModelType(event.target.value)}><option value="compare">MLP + logistic baseline</option><option value="mlp">MLP only</option></select></label>
            <label className="field"><span>Label policy</span><select value={labelPolicy} onChange={(event) => setLabelPolicy(event.target.value)}><option value="all_weighted">All weighted evidence</option><option value="observed_only">Observed only</option><option value="confirmed_only">Analyst-confirmed only</option></select></label>
            <label className="field"><span>Hidden layers</span><input value={hiddenLayers} onChange={(event) => setHiddenLayers(event.target.value)} placeholder="64,32" /><small>Comma-separated widths</small></label>
            <label className="field"><span>Dropout</span><input type="number" min={0} max={0.89} step={0.05} value={dropout} onChange={(event) => setDropout(Number(event.target.value))} /></label>
            <label className="field"><span>Early-stop patience</span><input type="number" min={1} max={1000} value={earlyStoppingPatience} onChange={(event) => setEarlyStoppingPatience(Number(event.target.value))} /></label>
            <label className="field"><span>Threshold strategy</span><select value={thresholdStrategy} onChange={(event) => setThresholdStrategy(event.target.value)}><option value="macro_f1">Macro-F1 (balanced)</option><option value="balanced_accuracy">Balanced accuracy</option><option value="f1">Positive-class F1</option><option value="fixed_0_5">Fixed 0.5</option></select></label>
            <label className="field"><span>Evaluation split</span><select value={splitStrategy} onChange={(event) => setSplitStrategy(event.target.value)}><option value="group_disjoint">Subscriber group-disjoint</option><option value="temporal">Temporal holdout</option><option value="stratified">Standard stratified</option></select></label>
            <label className="field"><span>Calibration</span><select value={calibration} onChange={(event) => setCalibration(event.target.value)}><option value="platt">Platt scaling</option><option value="none">None</option></select></label>
          </div>
          {trainingPreview ? (
            <details className="feature-selector">
              <summary>{selectedFeatures.length} / {trainingPreview.feature_names.length} model features selected</summary>
              <div className="feature-selector-actions">
                <button type="button" onClick={() => setSelectedFeatures(trainingPreview.feature_names)}>Select all</button>
                <button type="button" onClick={() => setSelectedFeatures([])}>Clear</button>
              </div>
              <div className="feature-checkbox-grid">
                {trainingPreview.feature_names.map((feature) => (
                  <label key={feature}>
                    <input type="checkbox" checked={selectedFeatures.includes(feature)} onChange={() => toggleFeature(feature)} />
                    <span>{feature.replaceAll("_", " ")}</span>
                  </label>
                ))}
              </div>
            </details>
          ) : null}
          <label className="check-field"><input type="checkbox" checked={treeBaseline} onChange={(event) => setTreeBaseline(event.target.checked)} /><span><strong>Tree-based baseline</strong><small>Also evaluate histogram gradient boosting on the held-out split.</small></span></label>
          <div className="training-policy"><Sparkles size={16} /><div><strong>Weighted pseudo-label policy</strong><span>Observed labels receive the highest weight; IPDR- and CGAN-inferred examples remain lower-weight training signals.</span></div></div>
          {error ? <div className="inline-error">{error}</div> : null}
          <button className="button button-primary full" type="submit" disabled={!sourceRunId || !selectedFeatures.length}><Play size={16} /> Train model</button>
        </form>
        <div className="monitor-column">
          {runId ? <RunMonitor runId={runId} onUpdate={(run) => { if (run.status === "complete") refresh(); }} /> : (
            <div className="nexus-panel active-model-panel">
              <div className="panel-heading"><div><span className="nexus-eyebrow">ACTIVE MODEL</span><strong>{activeModel?.id ?? "No model registered"}</strong></div><span className={`status-dot ${activeModel?.status ?? "pending"}`} /></div>
              {activeModel ? (
                <>
                  <div className="metric-quads">
                    {["f1_macro", "balanced_accuracy", "roc_auc", "average_precision"].map((metric) => {
                      const value = findMetric(activeModel.metrics, metric);
                      return <div key={metric}><span>{metric.replaceAll("_", " ").toUpperCase()}</span><strong>{value === null ? "—" : value.toFixed(3)}</strong></div>;
                    })}
                  </div>
                  <small>Created {formatDateTime(activeModel.created_at)} · {activeModel.active ? "active selection" : "latest candidate"}</small>
                </>
              ) : <p>Train the first model to populate evaluation metrics and pair predictions.</p>}
            </div>
          )}
        </div>
      </div>
      {activeModel ? (
        <section className="model-evaluation-grid">
          <div className="nexus-panel evaluation-summary">
            <div className="panel-heading"><div><span className="nexus-eyebrow">HELD-OUT EVALUATION</span><strong>Confusion matrix and provenance slices</strong></div><Gauge size={19} /></div>
            <div className="confusion-matrix">
              <span />
              <b>Predicted −</b>
              <b>Predicted +</b>
              <b>Actual −</b>
              <strong className="correct">{Number(confusionMatrix.tn ?? 0).toLocaleString("en-IN")}<small>TN</small></strong>
              <strong className="error">{Number(confusionMatrix.fp ?? 0).toLocaleString("en-IN")}<small>FP</small></strong>
              <b>Actual +</b>
              <strong className="error">{Number(confusionMatrix.fn ?? 0).toLocaleString("en-IN")}<small>FN</small></strong>
              <strong className="correct">{Number(confusionMatrix.tp ?? 0).toLocaleString("en-IN")}<small>TP</small></strong>
            </div>
            <div className="source-performance">
              {Object.entries(perLabelSource).slice(0, 10).map(([source, metrics]) => (
                <div key={source}>
                  <span>{source.replaceAll("_", " ")}</span>
                  <b>{Number(metrics.rows ?? 0).toLocaleString("en-IN")} rows</b>
                  <strong>{typeof metrics.f1_macro === "number" ? `macro-F1 ${metrics.f1_macro.toFixed(3)}` : "single-class slice"}</strong>
                </div>
              ))}
              {!Object.keys(perLabelSource).length ? <small>Per-source metrics appear on models trained with the current evaluation pipeline.</small> : null}
            </div>
          </div>
          <div className="nexus-panel error-review-panel">
            <div className="panel-heading"><div><span className="nexus-eyebrow">ERROR REVIEW</span><strong>False-positive and false-negative samples</strong></div><ShieldCheck size={19} /></div>
            <div className="error-review-columns">
              {[falsePositives, falseNegatives].map((review, columnIndex) => (
                <div key={review?.kind ?? columnIndex}>
                  <span>{columnIndex === 0 ? "FALSE POSITIVES" : "FALSE NEGATIVES"} · {(review?.total ?? 0).toLocaleString("en-IN")}</span>
                  {(review?.rows ?? []).slice(0, 6).map((row, index) => (
                    <div key={`${row.person_a}-${row.person_b}-${index}`}>
                      <strong>{row.person_a} ↔ {row.person_b}</strong>
                      <b>{Number(row.predicted_probability ?? 0).toFixed(3)}</b>
                      <small>{(row.truth_source ?? "unspecified").replaceAll("_", " ")}</small>
                    </div>
                  ))}
                  {!review?.rows.length ? <small>No review rows in this model artifact.</small> : null}
                </div>
              ))}
            </div>
          </div>
        </section>
      ) : null}
      <section className="models-grid">
        <div className="nexus-panel registry-panel">
          <div className="panel-heading"><div><span className="nexus-eyebrow">MODEL REGISTRY</span><strong>{models.length} versioned model{models.length === 1 ? "" : "s"}</strong></div><Cpu size={19} /></div>
          <div className="registry-list">
            {models.map((model) => (
              <div className={model.active ? "active" : ""} key={model.id}>
                <span className={`status-dot ${model.status}`} />
                <div><strong>{model.id}</strong><small>{model.status} · {formatDateTime(model.created_at)}</small></div>
                {model.active ? <span className="status-badge complete">active</span> : model.status === "ready" ? <button type="button" className="button button-ghost small" onClick={() => activate(model.id)}>Set active</button> : ["failed", "cancelled"].includes(model.status) ? <button type="button" className="button button-ghost small" onClick={() => resumeTraining(model.id)}>Resume</button> : null}
              </div>
            ))}
            {!models.length ? <div className="empty-state">No models have been trained.</div> : null}
          </div>
        </div>
        <form className="nexus-panel prediction-panel" onSubmit={predict}>
          <div className="panel-heading"><div><span className="nexus-eyebrow">PAIR PREDICTION</span><strong>Score two identifiers</strong></div><Activity size={19} /></div>
          <label className="field full"><span>Person A</span><input value={personA} onChange={(event) => setPersonA(event.target.value)} placeholder="SUB_0000001 or MSISDN" required /></label>
          <label className="field full"><span>Person B</span><input value={personB} onChange={(event) => setPersonB(event.target.value)} placeholder="SUB_0000002 or MSISDN" required /></label>
          <button className="button button-ghost full" type="submit" disabled={!activeModel || !sourceRunId || predicting}>{predicting ? "Scoring pair…" : "Predict relationship"}</button>
          {prediction ? (
            <div className="prediction-result">
              <span>MODEL RESULT</span>
              <strong>{prediction.prediction ?? prediction.predicted_label ?? "scored"}</strong>
              <b>{Number(
                prediction.friend_or_interacting_probability
                ?? prediction.probability
                ?? prediction.friend_probability
                ?? 0,
              ).toFixed(3)}</b>
              <small>Statistical pattern match — not proof of friendship or intent.</small>
              {predictionEvidence.length ? (
                <div className="prediction-evidence">
                  <span>HIGHEST-WEIGHT NON-ZERO EVIDENCE</span>
                  {predictionEvidence.map((item) => (
                    <div key={item.feature}>
                      <strong>{item.feature.replaceAll("_", " ")}</strong>
                      <code>{item.value.toFixed(3)}</code>
                      <small>{(item.relative_input_weight * 100).toFixed(1)}% diagnostic input weight</small>
                    </div>
                  ))}
                  <small>Diagnostic weights are not causal explanations.</small>
                </div>
              ) : null}
            </div>
          ) : null}
        </form>
        <form className="nexus-panel batch-prediction-panel" onSubmit={predictBatch}>
          <div className="panel-heading"><div><span className="nexus-eyebrow">BATCH PREDICTION</span><strong>Score a CSV of canonical pairs</strong></div><FileSpreadsheet size={19} /></div>
          <label className="file-field">
            <FileSpreadsheet size={17} />
            <span><strong>Pair CSV</strong><small>Headers: person_a, person_b</small></span>
            <input type="file" name="pairs" accept=".csv,text/csv" required />
          </label>
          <button className="button button-ghost full" type="submit" disabled={!activeModel || !sourceRunId || batchPredicting}>{batchPredicting ? "Scoring batch…" : "Run batch prediction"}</button>
          {batchResult ? (
            <div className="batch-result">
              <div><span>COMPLETED</span><strong>{batchResult.row_count.toLocaleString("en-IN")} pairs scored</strong></div>
              <a className="button button-primary small" href={artifactDownloadUrl(batchResult.artifact.id)}><Download size={15} /> Download predictions</a>
              <div className="batch-preview">
                {batchResult.preview.slice(0, 5).map((row, index) => (
                  <div key={`${row.person_a}-${row.person_b}-${index}`}>
                    <span>{row.person_a} ↔ {row.person_b}</span>
                    <strong>{Number(row.friend_or_interacting_probability ?? 0).toFixed(3)}</strong>
                  </div>
                ))}
              </div>
              <small>Statistical evidence only — not proof of friendship or intent.</small>
            </div>
          ) : null}
        </form>
      </section>
    </NexusShell>
  );
}
