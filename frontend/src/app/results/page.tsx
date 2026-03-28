"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  CheckCircle2,
  Database,
  TrendingUp,
  BarChart3,
  Layers,
  Download,
  Sparkles,
  AlertCircle,
  Loader2,
  Target,
  Cpu,
  ShieldCheck,
  Settings2,
  Activity,
  LineChart,
} from "lucide-react";
import { motion } from "motion/react";
import { Navbar } from "@/components/Navbar";
import { DownloadCard } from "@/components/DownloadCard";
import { StatTile } from "@/components/StatTile";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  fetchResults,
  getChartUrl,
  downloadFile,
  type ResultsManifest,
  type Metrics,
  type ModelComparison,
} from "@/lib/api";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function toChartTitle(path: string): string {
  const name = path.split("/").pop() ?? path;
  return name.replace(/\.[^.]+$/, "").replace(/_/g, " ");
}

// ─── Performance Badge ────────────────────────────────────────────────────────

const BADGE_STYLES: Record<string, string> = {
  Excellent: "bg-green-500/20 text-green-400 border-green-500/30",
  Good: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  Fair: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  Poor: "bg-red-500/20 text-red-400 border-red-500/30",
};

function PerformanceBadge({ rating }: { rating: string }) {
  const cls = BADGE_STYLES[rating] ?? BADGE_STYLES.Fair;
  return (
    <span
      className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold border ${cls}`}
    >
      {rating}
    </span>
  );
}

// ─── Metric Card ──────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  accent = "text-white",
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-4 text-center">
      <div className={`text-xl font-bold ${accent} mb-1 tabular-nums`}>{value}</div>
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
    </div>
  );
}

// ─── Chart Image ──────────────────────────────────────────────────────────────

function ChartImage({ url, title }: { url: string; title: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) return null;
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="block rounded-xl border border-zinc-700 bg-zinc-950/50 overflow-hidden hover:border-cyan-500/40 transition-colors"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={url}
        alt={title}
        onError={() => setFailed(true)}
        className="w-full h-44 object-contain bg-zinc-950/80"
      />
      <div className="px-3 py-2 border-t border-zinc-800">
        <p className="text-xs text-gray-400 capitalize truncate">{title}</p>
      </div>
    </a>
  );
}

// ─── Metrics Grid ─────────────────────────────────────────────────────────────

function MetricsGrid({ metrics }: { metrics: Metrics }) {
  return (
    <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
      <MetricCard label="Accuracy" value={formatPct(metrics.accuracy)} accent="text-green-400" />
      <MetricCard label="F1" value={metrics.f1.toFixed(3)} accent="text-cyan-400" />
      <MetricCard label="Precision" value={formatPct(metrics.precision)} />
      <MetricCard label="Recall" value={formatPct(metrics.recall)} />
      <MetricCard label="ROC-AUC" value={metrics.roc_auc.toFixed(3)} accent="text-purple-400" />
    </div>
  );
}

// ─── Model Comparison Table ───────────────────────────────────────────────────

function ComparisonTable({ rows }: { rows: ModelComparison[] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-zinc-800">
      <Table>
        <TableHeader>
          <TableRow className="border-zinc-800 bg-zinc-950/50 hover:bg-zinc-950/50">
            <TableHead className="text-gray-300 font-semibold">Model</TableHead>
            <TableHead className="text-gray-300 font-semibold">Accuracy</TableHead>
            <TableHead className="text-gray-300 font-semibold">F1 Score</TableHead>
            <TableHead className="text-gray-300 font-semibold">Status</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, i) => (
            <motion.tr
              key={i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.05 * i }}
              className={`border-zinc-800 transition-all duration-300 ${
                row.status === "Best"
                  ? "bg-gradient-to-r from-green-500/10 to-emerald-500/10 hover:from-green-500/20 hover:to-emerald-500/20"
                  : "hover:bg-zinc-900/50"
              }`}
            >
              <TableCell className="text-white font-medium">
                <div className="flex items-center gap-2">
                  {row.status === "Best" && (
                    <motion.div
                      animate={{ rotate: [0, 360] }}
                      transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                    >
                      <Sparkles className="w-4 h-4 text-green-400" />
                    </motion.div>
                  )}
                  {row.model}
                </div>
              </TableCell>
              <TableCell>
                <span className={row.status === "Best" ? "text-green-400 font-semibold" : "text-gray-300"}>
                  {formatPct(row.accuracy)}
                </span>
              </TableCell>
              <TableCell>
                <span className={row.status === "Best" ? "text-green-400 font-semibold" : "text-gray-300"}>
                  {row.f1.toFixed(3)}
                </span>
              </TableCell>
              <TableCell>
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.05 * i, type: "spring" }}
                  className={`inline-flex px-3 py-1 rounded-full text-xs font-semibold ${
                    row.status === "Best"
                      ? "bg-gradient-to-r from-green-500/20 to-emerald-500/20 text-green-400 border border-green-500/30"
                      : row.status === "Good"
                      ? "bg-gradient-to-r from-blue-500/20 to-cyan-500/20 text-blue-400 border border-blue-500/30"
                      : "bg-zinc-800 text-gray-400 border border-zinc-700"
                  }`}
                >
                  {row.status}
                </motion.span>
              </TableCell>
            </motion.tr>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

// ─── Results Content ──────────────────────────────────────────────────────────

function ResultsContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const datasetId   = searchParams.get("datasetId") ?? "";
  const fileName    = searchParams.get("fileName") ?? "dataset.csv";
  const targetParam = searchParams.get("targetColumn") ?? "";

  const [manifest, setManifest]     = useState<ResultsManifest | null>(null);
  const [loading, setLoading]       = useState(!!datasetId);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) return;
    fetchResults(datasetId)
      .then(setManifest)
      .catch((err) => setFetchError(String(err)))
      .finally(() => setLoading(false));
  }, [datasetId]);

  // ── Data extraction ────────────────────────────────────────────────────────
  const ps         = manifest?.pipeline_summary;
  const dsInfo     = ps?.dataset_info;
  const modelInfo  = ps?.model_info;
  const metrics    = ps?.metrics;
  const comparison = ps?.model_comparison ?? [];

  // Charts: prefer eda_section / model_evaluation_section, fall back to flat arrays
  const edaCharts  = manifest?.eda_section?.charts  ?? manifest?.eda_charts  ?? [];
  const evalCharts = manifest?.model_evaluation_section?.charts ?? manifest?.evaluation_charts ?? [];
  const evalRating = manifest?.model_evaluation_section?.performance_rating ?? modelInfo?.performance_rating ?? "";
  const evalComp   = manifest?.model_evaluation_section?.model_comparison ?? comparison;
  const evalMetrics = manifest?.model_evaluation_section?.metrics ?? metrics;

  const downloads  = manifest?.downloads ?? [];
  const targetCol  = dsInfo?.target_col || targetParam || "target";
  const rating     = modelInfo?.performance_rating ?? evalRating;

  // Split EDA charts by name pattern
  const distCharts = edaCharts.filter((c) => (c.split("/").pop() ?? "").startsWith("distribution_"));
  const compCharts = edaCharts.filter((c) => (c.split("/").pop() ?? "").startsWith("comparison_"));
  const otherCharts = edaCharts.filter((c) => {
    const name = c.split("/").pop() ?? "";
    return !name.startsWith("distribution_") && !name.startsWith("comparison_");
  });

  // No datasetId — user landed here without a valid pipeline session
  if (!datasetId) {
    return (
      <div className="min-h-screen bg-[#0D0F14] flex flex-col items-center justify-center gap-6 px-6">
        <AlertCircle className="w-12 h-12 text-red-400" />
        <div className="text-center">
          <p className="text-red-300 font-medium text-xl mb-2">No pipeline session found</p>
          <p className="text-gray-500 text-sm">
            The dataset upload may have failed. Please go back and try again.
          </p>
        </div>
        <Button
          onClick={() => router.push("/")}
          className="bg-gradient-to-r from-purple-600 to-cyan-600 text-white border-0"
        >
          Go Back
        </Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0D0F14] relative overflow-hidden">
      {/* Ambient blobs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          animate={{ scale: [1, 1.3, 1], x: [0, 100, 0], y: [0, 50, 0], opacity: [0.15, 0.25, 0.15] }}
          transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-full blur-3xl"
        />
        <motion.div
          animate={{ scale: [1.3, 1, 1.3], x: [0, -100, 0], y: [0, -50, 0], opacity: [0.15, 0.25, 0.15] }}
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
          className="absolute bottom-1/4 right-1/4 w-[600px] h-[600px] bg-gradient-to-br from-cyan-600/20 to-blue-600/20 rounded-full blur-3xl"
        />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(139,92,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(139,92,246,0.03)_1px,transparent_1px)] bg-[size:100px_100px]" />
      </div>

      <Navbar />

      <main className="mx-auto max-w-7xl px-6 py-8 relative z-10">

        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-32 gap-4">
            <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}>
              <Loader2 className="w-12 h-12 text-purple-400" />
            </motion.div>
            <p className="text-gray-400">Loading results…</p>
          </div>
        )}

        {/* Error */}
        {fetchError && !loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mb-8 p-5 rounded-2xl border border-red-500/30 bg-red-500/10 flex items-center gap-3"
          >
            <AlertCircle className="w-6 h-6 text-red-400 shrink-0" />
            <div>
              <p className="text-red-300 font-medium">Could not load results</p>
              <p className="text-red-400/70 text-sm mt-1">{fetchError}</p>
            </div>
          </motion.div>
        )}

        {/* ─────────────────────────────────────────────────────────────────── */}
        {/* SECTION 1 — Pipeline Summary                                        */}
        {/* ─────────────────────────────────────────────────────────────────── */}
        {!loading && manifest && (
          <>
            {/* Success banner */}
            <motion.div
              initial={{ opacity: 0, y: -20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
              className="bg-gradient-to-r from-green-500/10 via-emerald-500/10 to-green-500/10 border border-green-500/30 rounded-2xl p-6 mb-8 backdrop-blur-sm shadow-xl shadow-green-500/10 relative overflow-hidden"
            >
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-green-500/5 to-emerald-500/5"
                animate={{ x: ["-100%", "100%"] }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              />
              <div className="relative z-10 flex items-center gap-4 flex-wrap">
                <motion.div
                  animate={{ scale: [1, 1.1, 1], rotate: [0, 5, -5, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <CheckCircle2 className="w-8 h-8 text-green-400" />
                </motion.div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 flex-wrap mb-1">
                    <h2 className="text-green-400 font-semibold text-xl">Pipeline Complete</h2>
                    {rating && <PerformanceBadge rating={rating} />}
                  </div>
                  <p className="text-gray-300">
                    <span className="text-white font-medium">{fileName}</span>
                    {" · "}target:{" "}
                    <span className="text-purple-400 font-medium">{targetCol}</span>
                    {modelInfo?.final_model_name && (
                      <>
                        {" · "}Best model:{" "}
                        <span className="text-cyan-400 font-medium">{modelInfo.final_model_name}</span>
                      </>
                    )}
                  </p>
                </div>
                <motion.div
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                >
                  <Sparkles className="w-6 h-6 text-green-400" />
                </motion.div>
              </div>
            </motion.div>

            {/* Stat tiles */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
            >
              <StatTile
                label="Dataset Rows"
                value={dsInfo?.cleaned_shape?.[0]?.toLocaleString() ?? "—"}
                icon={<Database className="w-5 h-5" />}
                delay={0}
              />
              <StatTile
                label="Features"
                value={
                  dsInfo?.engineered_features
                    ? String(dsInfo.engineered_features)
                    : dsInfo?.cleaned_shape?.[1]
                    ? String(dsInfo.cleaned_shape[1])
                    : "—"
                }
                icon={<Layers className="w-5 h-5" />}
                delay={0.1}
              />
              <StatTile
                label="Best Accuracy"
                value={metrics?.accuracy ? formatPct(metrics.accuracy) : "—"}
                icon={<TrendingUp className="w-5 h-5" />}
                delay={0.2}
              />
              <StatTile
                label="F1 Score"
                value={metrics?.f1 ? metrics.f1.toFixed(3) : "—"}
                icon={<Activity className="w-5 h-5" />}
                delay={0.3}
              />
            </motion.div>

            {/* Pipeline details card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35 }}
              className="rounded-2xl border border-zinc-800 bg-zinc-900/90 backdrop-blur-xl p-8 mb-8 shadow-2xl"
            >
              <motion.h2
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
                className="text-2xl text-white mb-6 flex items-center gap-3"
              >
                <Cpu className="w-6 h-6 text-purple-400" />
                Pipeline Summary
              </motion.h2>

              {/* All 5 metrics */}
              {metrics && (
                <div className="mb-6">
                  <MetricsGrid metrics={metrics} />
                </div>
              )}

              {/* Dataset info + badges */}
              <div className="flex flex-wrap items-center gap-3 pt-5 border-t border-zinc-800">
                <div className="flex items-center gap-2 rounded-full border border-zinc-700 bg-zinc-800/50 px-3 py-1.5 text-sm">
                  <Target className="w-3.5 h-3.5 text-purple-400" />
                  <span className="text-gray-400">Target:</span>
                  <span className="text-purple-400 font-medium">{targetCol}</span>
                </div>

                {dsInfo?.original_shape && (
                  <div className="flex items-center gap-2 rounded-full border border-zinc-700 bg-zinc-800/50 px-3 py-1.5 text-sm">
                    <Database className="w-3.5 h-3.5 text-gray-400" />
                    <span className="text-gray-500">Raw:</span>
                    <span className="text-gray-300">
                      {dsInfo.original_shape[0].toLocaleString()} × {dsInfo.original_shape[1]}
                    </span>
                  </div>
                )}

                {dsInfo?.cleaned_shape && (
                  <div className="flex items-center gap-2 rounded-full border border-zinc-700 bg-zinc-800/50 px-3 py-1.5 text-sm">
                    <Database className="w-3.5 h-3.5 text-cyan-400" />
                    <span className="text-gray-500">Cleaned:</span>
                    <span className="text-gray-300">
                      {dsInfo.cleaned_shape[0].toLocaleString()} × {dsInfo.cleaned_shape[1]}
                    </span>
                  </div>
                )}

                {modelInfo?.smote_applied && (
                  <div className="flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5 text-sm">
                    <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                    <span className="text-cyan-400 font-medium">SMOTE Applied</span>
                  </div>
                )}

                {modelInfo?.tuning_applied && (
                  <div className="flex items-center gap-2 rounded-full border border-pink-500/30 bg-pink-500/10 px-3 py-1.5 text-sm">
                    <Settings2 className="w-3.5 h-3.5 text-pink-400" />
                    <span className="text-pink-400 font-medium">Hyperparameter Tuning</span>
                  </div>
                )}
              </div>
            </motion.div>

            {/* ───────────────────────────────────────────────────────────── */}
            {/* SECTION 2 — EDA Charts                                        */}
            {/* ───────────────────────────────────────────────────────────── */}
            {edaCharts.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.45 }}
                className="rounded-2xl border border-zinc-800 bg-zinc-900/90 backdrop-blur-xl p-8 mb-8 shadow-2xl"
              >
                <motion.h2
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 }}
                  className="text-2xl text-white mb-6 flex items-center gap-3"
                >
                  <BarChart3 className="w-6 h-6 text-cyan-400" />
                  Exploratory Data Analysis
                </motion.h2>

                {/* Feature Distributions */}
                {distCharts.length > 0 && (
                  <div className="mb-8">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                      Feature Distributions
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {distCharts.map((chart, i) => (
                        <motion.div
                          key={chart}
                          initial={{ opacity: 0, y: 15 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.52 + i * 0.06 }}
                          whileHover={{ y: -4 }}
                        >
                          <ChartImage
                            url={datasetId ? getChartUrl(datasetId, chart) : "#"}
                            title={toChartTitle(chart)}
                          />
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Before vs After Cleaning — side-by-side */}
                {compCharts.length > 0 && (
                  <div className="mb-8">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                      Before vs After Cleaning
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {compCharts.map((chart, i) => (
                        <motion.div
                          key={chart}
                          initial={{ opacity: 0, scale: 0.97 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.55 + i * 0.07 }}
                          whileHover={{ y: -4 }}
                        >
                          <ChartImage
                            url={datasetId ? getChartUrl(datasetId, chart) : "#"}
                            title={toChartTitle(chart)}
                          />
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Other EDA charts (heatmaps, target distribution, etc.) */}
                {otherCharts.length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                      {distCharts.length === 0 && compCharts.length === 0 ? "Charts" : "Overview"}
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {otherCharts.map((chart, i) => (
                        <motion.div
                          key={chart}
                          initial={{ opacity: 0, y: 15 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.58 + i * 0.06 }}
                          whileHover={{ y: -4 }}
                        >
                          <ChartImage
                            url={datasetId ? getChartUrl(datasetId, chart) : "#"}
                            title={toChartTitle(chart)}
                          />
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {/* ───────────────────────────────────────────────────────────── */}
            {/* SECTION 3 — Model Evaluation                                  */}
            {/* ───────────────────────────────────────────────────────────── */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="rounded-2xl border border-zinc-800 bg-zinc-900/90 backdrop-blur-xl p-8 mb-8 shadow-2xl"
            >
              <motion.h2
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.55 }}
                className="text-2xl text-white mb-6 flex items-center gap-3"
              >
                <LineChart className="w-6 h-6 text-pink-400" />
                Model Evaluation
              </motion.h2>

              {/* Metrics cards */}
              {evalMetrics && (
                <div className="mb-8">
                  <MetricsGrid metrics={evalMetrics} />
                </div>
              )}

              {/* Model comparison table */}
              {evalComp.length > 0 && (
                <div className="mb-8">
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                    Model Comparison
                  </p>
                  <ComparisonTable rows={evalComp} />
                </div>
              )}

              {/* Evaluation charts grid */}
              {evalCharts.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                    Evaluation Charts
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {evalCharts.map((chart, i) => (
                      <motion.div
                        key={chart}
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.6 + i * 0.07 }}
                        whileHover={{ y: -4 }}
                      >
                        <ChartImage
                          url={datasetId ? getChartUrl(datasetId, chart) : "#"}
                          title={toChartTitle(chart)}
                        />
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>

            {/* ───────────────────────────────────────────────────────────── */}
            {/* SECTION 4 — Downloads                                         */}
            {/* ───────────────────────────────────────────────────────────── */}
            {downloads.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="mb-8"
              >
                <motion.h2
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.65 }}
                  className="text-2xl text-white mb-6 flex items-center gap-3"
                >
                  <Download className="w-6 h-6 text-purple-400" />
                  Download Outputs
                </motion.h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {downloads.map((dl, i) => (
                    <motion.div
                      key={dl.file_type}
                      initial={{ opacity: 0, y: 30, scale: 0.9 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      transition={{ delay: 0.7 + i * 0.1, type: "spring", stiffness: 150 }}
                      whileHover={{ y: -10, scale: 1.03 }}
                    >
                      <DownloadCard
                        type={
                          dl.file_type === "model"
                            ? "pkl"
                            : dl.file_type === "evaluation"
                            ? "pdf"
                            : "csv"
                        }
                        fileName={dl.filename}
                        description={
                          dl.file_type === "model"
                            ? `Best model: ${modelInfo?.final_model_name ?? "—"} · Accuracy: ${
                                metrics?.accuracy ? formatPct(metrics.accuracy) : "—"
                              }`
                            : dl.file_type === "cleaned_data"
                            ? "Processed dataset — nulls handled, outliers removed"
                            : "Full evaluation metrics and model summaries"
                        }
                        onDownload={() => {
                          if (datasetId) downloadFile(datasetId, dl.file_type);
                        }}
                      />
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Run Another */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2 }}
              className="flex justify-center mb-8"
            >
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button
                  onClick={() => router.push("/")}
                  className="bg-gradient-to-r from-purple-600 via-pink-600 to-cyan-600 hover:from-purple-500 hover:via-pink-500 hover:to-cyan-500 text-white px-8 py-6 text-lg shadow-lg shadow-purple-500/30 border-0 relative overflow-hidden group"
                >
                  <motion.div className="absolute inset-0 bg-gradient-to-r from-cyan-600 to-purple-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  <span className="relative z-10 flex items-center gap-2">
                    <Sparkles className="w-5 h-5" />
                    Run Another Dataset
                  </span>
                </Button>
              </motion.div>
            </motion.div>
          </>
        )}
      </main>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#0D0F14]" />}>
      <ResultsContent />
    </Suspense>
  );
}
