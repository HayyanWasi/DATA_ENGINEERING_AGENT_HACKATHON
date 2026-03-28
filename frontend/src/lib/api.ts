const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ImportResponse {
  dataset_id: string;
  row_count: number;
  columns: string[];
}

export interface PipelineStatus {
  dataset_id: string;
  status: "pending" | "running" | "complete" | "failed";
  stages_completed: number;
  last_stage: string;
}

export interface ModelComparison {
  model: string;
  accuracy: number;
  f1: number;
  status: string;
}

export interface ResultsManifest {
  dataset_id: string;
  target_col: string;
  elapsed_seconds: number;
  pipeline_summary: {
    model_used: string;
    performance_rating: string;
    accuracy: number;
    f1: number;
    precision: number;
    recall: number;
    roc_auc: number;
    smote_applied: boolean;
    tuning_applied: boolean;
    best_params: Record<string, unknown>;
    model_comparison: ModelComparison[];
  };
  eda_charts: string[];
  dataset_stats: { rows: number; features: number };
  downloads: Array<{ file_type: string; filename: string; url: string }>;
  // backward-compat fields from old endpoint
  evaluation_summary?: Record<string, unknown>;
  available_downloads?: Array<{ file_type: string; url: string; path: string }>;
}

// ─── Upload ───────────────────────────────────────────────────────────────────

export async function importDataset(
  file: File,
): Promise<ImportResponse> {
  const form = new FormData();
  form.append("user_id", "frontend_user");
  form.append("dataset_name", file.name);
  form.append("import_type", "file");
  form.append("file", file);

  const res = await fetch(`${API_BASE}/api/import-data`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => String(res.status));
    throw new Error(`Import failed (${res.status}): ${text}`);
  }
  return res.json();
}

// ─── Pipeline ────────────────────────────────────────────────────────────────

/**
 * Fire-and-forget pipeline run.
 * Navigates away immediately; backend runs in background.
 */
export function startPipelineRun(datasetId: string, targetCol: string): void {
  fetch(`${API_BASE}/api/pipeline/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, target_col: targetCol }),
  }).catch((err) => console.error("[pipeline/run]", err));
}

export async function fetchPipelineStatus(
  datasetId: string,
): Promise<PipelineStatus> {
  const res = await fetch(`${API_BASE}/api/pipeline/status/${datasetId}`);
  if (!res.ok) throw new Error(`Status fetch failed: ${res.status}`);
  return res.json();
}

// ─── Results ─────────────────────────────────────────────────────────────────

export async function fetchResults(
  datasetId: string,
): Promise<ResultsManifest> {
  const res = await fetch(`${API_BASE}/api/results/${datasetId}`);
  if (!res.ok) throw new Error(`Results fetch failed: ${res.status}`);
  return res.json();
}

// ─── Charts & Downloads ───────────────────────────────────────────────────────

export function getChartUrl(datasetId: string, chartPath: string): string {
  const filename = chartPath.split("/").pop() ?? chartPath;
  return `${API_BASE}/api/charts/${datasetId}/${encodeURIComponent(filename)}`;
}

export function downloadFile(datasetId: string, fileType: string): void {
  window.open(`${API_BASE}/api/download/${datasetId}/${fileType}`, "_blank");
}
