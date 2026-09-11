export function nexusApiUrl(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `/api${normalized}`;
}

export async function nexusFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(nexusApiUrl(path), {
    cache: "no-store",
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(
        typeof window !== "undefined"
          ? { "X-Nexus-Role": window.localStorage.getItem("nexus-role") ?? "analyst" }
          : {}
      ),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string; error?: string };
      message = payload.detail ?? payload.error ?? message;
    } catch {
      // Keep the status-based message for non-JSON errors.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export function artifactDownloadUrl(artifactId: string): string {
  return nexusApiUrl(`/artifacts/${artifactId}/download`);
}

export function formatIndianNumber(value: number | string | null | undefined) {
  const numeric = Number(value ?? 0);
  return new Intl.NumberFormat("en-IN").format(Number.isFinite(numeric) ? numeric : 0);
}

export function formatDateTime(value: string | null | undefined) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(parsed);
}

export type DatasetRecord = {
  id: string;
  name: string;
  kind: string;
  status: string;
  created_at: string;
  row_counts: Record<string, number>;
  quality: string[];
  metadata?: Record<string, unknown>;
  source_files?: Array<Record<string, unknown>>;
  artifact_count?: number;
};

export type IdentityValidation = {
  status?: string;
  all_records_scanned?: boolean;
  chunk_size?: number;
  sources?: Record<string, {
    present?: boolean;
    subject_column?: string | null;
    subscriber_column?: string | null;
    contact_column?: string | null;
    subject_count?: number;
    subject_identifiers?: string[];
    subject_counts?: Record<string, number>;
    date_range?: { start?: string | null; end?: string | null };
    row_count?: number;
    contact_count?: number;
    invalid_timestamp_rows?: number;
  }>;
  matching?: {
    cdr_subject_count?: number;
    ipdr_subscriber_count?: number;
    matched_count?: number;
    unmatched_cdr_count?: number;
    unmatched_ipdr_count?: number;
    matched_identifiers?: string[];
    unmatched_cdr_subjects?: string[];
    unmatched_ipdr_subscribers?: string[];
  };
  temporal?: {
    status?: string;
    overlaps?: boolean | null;
    explanation?: string;
  };
  identity_evidence?: {
    statement?: string;
    identifier_agreement_proves_personal_identity?: boolean;
  };
};

export type IdentityDecision = {
  allowed?: boolean;
  answer?: string | null;
  generation_mode?: string | null;
  decision_status?: string;
  blocking_reasons?: string[];
  warnings?: string[];
  available_operations?: string[];
  confirmation_is_not_an_override?: boolean;
};

export type ArtifactRecord = {
  id: string;
  kind: string;
  filename: string;
  size: number;
  created_at: string;
};

export type RunStage = {
  name: string;
  status: string;
  progress: number;
  input_count?: number | null;
  output_count?: number | null;
  error?: string | null;
};

export type RunRecord = {
  id: string;
  dataset_id?: string | null;
  parent_run_id?: string | null;
  kind: string;
  status: string;
  progress: number;
  stage: string;
  message: string;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
  error?: string | null;
  config: Record<string, unknown>;
  stages?: RunStage[];
  artifacts?: ArtifactRecord[];
  children?: RunRecord[];
  log?: string[];
  artifact_count?: number;
};

export type ModelRecord = {
  id: string;
  dataset_id?: string | null;
  run_id: string;
  status: string;
  active: boolean;
  created_at: string;
  metrics: Record<string, unknown>;
  config: Record<string, unknown>;
};
