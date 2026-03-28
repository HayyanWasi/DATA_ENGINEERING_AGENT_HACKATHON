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

export interface Metrics {
  accuracy: number;
  f1: number;
  precision: number;
  recall: number;
  roc_auc: number;
}

export interface ModelComparison {
  model: string;
  accuracy: number;
  f1: number;
  status: string;
}

export interface DatasetInfo {
  target_col: string;
  original_shape: [number, number];
  cleaned_shape: [number, number];
  engineered_features: number;
}

export interface ModelInfo {
  final_model_name: string;
  tuning_applied: boolean;
  smote_applied: boolean;
  performance_rating: string;
}

export interface PipelineSummary {
  dataset_info: DatasetInfo;
  model_info: ModelInfo;
  metrics: Metrics;
  model_comparison: ModelComparison[];
  balance_info: Record<string, unknown>;
  tuning_info: Record<string, unknown>;
}

export interface EdaSection {
  charts: string[];
  chart_count: number;
}

export interface ModelEvaluationSection {
  charts: string[];
  chart_count: number;
  metrics: Metrics;
  model_comparison: ModelComparison[];
  performance_rating: string;
}

export interface ResultsManifest {
  dataset_id: string;
  pipeline_summary: PipelineSummary;
  eda_section: EdaSection;
  model_evaluation_section: ModelEvaluationSection;
  downloads: Array<{ file_type: string; filename: string; url: string }>;
  /** Flat chart arrays written by build_results_manifest (also in sections above) */
  eda_charts?: string[];
  evaluation_charts?: string[];
  output_files?: {
    model_files: string[];
    data_files: string[];
    evaluation_charts: string[];
    eda_charts: string[];
    json_reports: string[];
  };
}

// ─── Upload ───────────────────────────────────────────────────────────────────

export async function importDataset(file: File): Promise<ImportResponse> {
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

// ─── Pipeline ─────────────────────────────────────────────────────────────────

/** Fire-and-forget — navigates away immediately; backend runs in background. */
export function startPipelineRun(datasetId: string, targetCol: string): void {
  fetch(`${API_BASE}/api/pipeline/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, target_col: targetCol }),
  }).catch((err) => console.error("[pipeline/run]", err));
}

export async function fetchPipelineStatus(datasetId: string): Promise<PipelineStatus> {
  const res = await fetch(`${API_BASE}/api/pipeline/status/${datasetId}`);
  if (!res.ok) throw new Error(`Status fetch failed: ${res.status}`);
  return res.json();
}

// ─── Results ──────────────────────────────────────────────────────────────────

export async function fetchResults(datasetId: string): Promise<ResultsManifest> {
  const res = await fetch(`${API_BASE}/api/results/${datasetId}`);
  if (!res.ok) throw new Error(`Results fetch failed (${res.status})`);
  return res.json();
}

// ─── Charts & Downloads ───────────────────────────────────────────────────────

/**
 * Build the URL for a chart image served by GET /api/charts/{datasetId}/{filename}.
 * chartPath can be a full path ("charts/foo.png") or just a filename ("foo.png").
 */
export function getChartUrl(datasetId: string, chartPath: string): string {
  const filename = chartPath.split("/").pop() ?? chartPath;
  return `${API_BASE}/api/charts/${datasetId}/${encodeURIComponent(filename)}`;
}

/** Open a pipeline output file download in a new tab. */
export function downloadFile(datasetId: string, fileType: string): void {
  window.open(`${API_BASE}/api/download/${datasetId}/${fileType}`, "_blank");
}
