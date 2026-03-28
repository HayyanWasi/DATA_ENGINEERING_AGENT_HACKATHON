"use client";

import { useEffect, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import {
  CheckCircle2,
  Loader2,
  AlertCircle,
  Sparkles,
  Zap,
} from "lucide-react";
import { motion } from "motion/react";
import { Navbar } from "@/components/Navbar";
import {
  fetchPipelineStatus,
  startPipelineRun,
  type PipelineStatus,
} from "@/lib/api";

const STAGES = [
  "Data Cleaning",
  "Exploratory Data Analysis",
  "Feature Engineering",
  "Feature Scaling",
  "Class Imbalance",
  "Model Training",
  "Model Selection",
  "Hyperparameter Tuning",
  "Model Evaluation",
  "Final Output",
];

export default function PipelinePage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();

  const datasetId = params.dataset_id as string;
  const fileName = searchParams.get("fileName") ?? "dataset.csv";
  const targetColumn = searchParams.get("targetColumn") ?? "target";

  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState(false);

  useEffect(() => {
    if (!datasetId || started) return;
    setStarted(true);

    // Fire-and-forget — backend runs all stages; we poll for progress
    startPipelineRun(datasetId, targetColumn);

    const poll = setInterval(async () => {
      try {
        const s = await fetchPipelineStatus(datasetId);
        setStatus(s);
        if (s.status === "complete") {
          clearInterval(poll);
          const q = new URLSearchParams({ datasetId, fileName, targetColumn }).toString();
          router.push(`/results?${q}`);
        } else if (s.status === "failed") {
          clearInterval(poll);
          setError("Pipeline failed. Check backend logs.");
        }
      } catch {
        // transient network error — keep polling
      }
    }, 2000);

    return () => clearInterval(poll);
  }, [datasetId, targetColumn, fileName, router, started]);

  const stagesCompleted = status?.stages_completed ?? 0;
  const progress = Math.round((stagesCompleted / STAGES.length) * 100);
  const currentStage = status?.last_stage ?? "Starting…";

  return (
    <div className="min-h-screen bg-[#0D0F14] relative overflow-hidden">
      {/* Ambient blobs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          animate={{ scale: [1, 1.3, 1], x: [0, 80, 0], y: [0, 40, 0], opacity: [0.1, 0.2, 0.1] }}
          transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-full blur-3xl"
        />
        <motion.div
          animate={{ scale: [1.3, 1, 1.3], x: [0, -80, 0], y: [0, -40, 0], opacity: [0.1, 0.2, 0.1] }}
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
          className="absolute bottom-1/4 right-1/4 w-[600px] h-[600px] bg-gradient-to-br from-cyan-600/20 to-blue-600/20 rounded-full blur-3xl"
        />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(139,92,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(139,92,246,0.03)_1px,transparent_1px)] bg-[size:100px_100px]" />
      </div>

      <Navbar />

      <main className="mx-auto max-w-2xl px-6 py-16 relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
            className="inline-block mb-4"
          >
            <Zap className="w-12 h-12 text-purple-400" />
          </motion.div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent mb-2">
            Running AutoML Pipeline
          </h1>
          <p className="text-gray-400">
            <span className="text-white font-medium">{fileName}</span> · target:{" "}
            <span className="text-purple-400 font-medium">{targetColumn}</span>
          </p>
        </motion.div>

        {/* Error state */}
        {error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-8 p-5 rounded-2xl border border-red-500/30 bg-red-500/10 flex items-center gap-3"
          >
            <AlertCircle className="w-6 h-6 text-red-400 shrink-0" />
            <p className="text-red-300">{error}</p>
          </motion.div>
        )}

        {/* Progress bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8 p-6 rounded-2xl border border-zinc-800 bg-zinc-900/90 backdrop-blur-xl"
        >
          <div className="flex justify-between text-sm mb-3">
            <span className="text-gray-400">
              {stagesCompleted === 0 ? "Initialising…" : currentStage}
            </span>
            <span className="text-purple-400 font-semibold">{progress}%</span>
          </div>
          <div className="h-3 rounded-full bg-zinc-800 overflow-hidden">
            <motion.div
              className="h-full rounded-full bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 relative overflow-hidden"
              initial={{ width: "0%" }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            >
              <motion.div
                animate={{ x: ["-100%", "200%"] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
              />
            </motion.div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {stagesCompleted} / {STAGES.length} stages complete
          </p>
        </motion.div>

        {/* Stage list */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="space-y-2"
        >
          {STAGES.map((stage, i) => {
            const done = i < stagesCompleted;
            const active = i === stagesCompleted && !error;
            return (
              <motion.div
                key={stage}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.35 + i * 0.04 }}
                className={`flex items-center gap-3 px-5 py-3 rounded-xl border transition-colors ${
                  done
                    ? "border-green-500/30 bg-green-500/10"
                    : active
                    ? "border-purple-500/40 bg-purple-500/10"
                    : "border-zinc-800 bg-zinc-900/50"
                }`}
              >
                <div className="shrink-0">
                  {done ? (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 300 }}
                    >
                      <CheckCircle2 className="w-5 h-5 text-green-400" />
                    </motion.div>
                  ) : active ? (
                    <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}>
                      <Loader2 className="w-5 h-5 text-purple-400" />
                    </motion.div>
                  ) : (
                    <div className="w-5 h-5 rounded-full border border-zinc-700 flex items-center justify-center">
                      <span className="text-[10px] text-zinc-600">{i + 1}</span>
                    </div>
                  )}
                </div>
                <span
                  className={`text-sm font-medium ${
                    done ? "text-green-300" : active ? "text-purple-300" : "text-zinc-500"
                  }`}
                >
                  {stage}
                </span>
                {active && (
                  <motion.span
                    animate={{ opacity: [1, 0.4, 1] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                    className="ml-auto text-xs text-purple-400"
                  >
                    running…
                  </motion.span>
                )}
                {done && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="ml-auto"
                  >
                    <Sparkles className="w-3 h-3 text-green-400" />
                  </motion.div>
                )}
              </motion.div>
            );
          })}
        </motion.div>
      </main>
    </div>
  );
}
