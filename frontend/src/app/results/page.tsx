"use client";

import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  CheckCircle2,
  Database,
  TrendingUp,
  Clock,
  Layers,
  Download,
  Sparkles,
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

const modelResults = [
  {
    model: "Random Forest",
    accuracy: "89.3%",
    f1Score: "0.88",
    status: "Good",
  },
  {
    model: "Logistic Regression",
    accuracy: "84.1%",
    f1Score: "0.82",
    status: "Fair",
  },
  {
    model: "XGBoost",
    accuracy: "94.2%",
    f1Score: "0.93",
    status: "Best",
  },
] as const;

function ResultsContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const fileName = searchParams.get("fileName") ?? "uploaded_dataset.csv";
  const targetColumn = searchParams.get("targetColumn") ?? "target";

  const handleRunAnother = () => {
    router.push("/");
  };

  return (
    <div className="min-h-screen bg-[#0D0F14] relative overflow-hidden">
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          animate={{
            scale: [1, 1.3, 1],
            x: [0, 100, 0],
            y: [0, 50, 0],
            opacity: [0.15, 0.25, 0.15],
          }}
          transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-full blur-3xl"
        />
        <motion.div
          animate={{
            scale: [1.3, 1, 1.3],
            x: [0, -100, 0],
            y: [0, -50, 0],
            opacity: [0.15, 0.25, 0.15],
          }}
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
          className="absolute bottom-1/4 right-1/4 w-[600px] h-[600px] bg-gradient-to-br from-cyan-600/20 to-blue-600/20 rounded-full blur-3xl"
        />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(139,92,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(139,92,246,0.03)_1px,transparent_1px)] bg-[size:100px_100px]" />
      </div>

      <Navbar />

      <main className="mx-auto max-w-7xl px-6 py-8 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: -20, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ type: "spring", stiffness: 200 }}
          className="bg-gradient-to-r from-green-500/10 via-emerald-500/10 to-green-500/10 border border-green-500/30 rounded-2xl p-6 mb-8 flex items-center gap-4 backdrop-blur-sm shadow-xl shadow-green-500/10 relative overflow-hidden"
        >
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-green-500/5 to-emerald-500/5"
            animate={{ x: ["-100%", "100%"] }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          />
          <motion.div
            animate={{ scale: [1, 1.1, 1], rotate: [0, 5, -5, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <CheckCircle2 className="w-8 h-8 text-green-400 relative z-10" />
          </motion.div>
          <div className="relative z-10">
            <h2 className="text-green-400 font-semibold text-xl mb-1">Pipeline Complete</h2>
            <p className="text-gray-300">Results ready for {fileName} with target column {targetColumn}</p>
          </div>
          <motion.div
            className="ml-auto"
            animate={{ rotate: [0, 360] }}
            transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
          >
            <Sparkles className="w-6 h-6 text-green-400 relative z-10" />
          </motion.div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8"
        >
          <StatTile label="Dataset Rows" value="9,995" icon={<Database className="w-5 h-5" />} delay={0} />
          <StatTile label="Features Used" value="27" icon={<Layers className="w-5 h-5" />} delay={0.1} />
          <StatTile label="Best Accuracy" value="94.2%" icon={<TrendingUp className="w-5 h-5" />} delay={0.2} />
          <StatTile label="Pipeline Duration" value="2m 14s" icon={<Clock className="w-5 h-5" />} delay={0.3} />
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mb-8"
        >
          <motion.h3
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="text-2xl text-white mb-6 flex items-center gap-3"
          >
            <Download className="w-6 h-6 text-purple-400" />
            Download Outputs
          </motion.h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                type: "csv" as const,
                fileName: "cleaned_data.csv",
                description: "Processed dataset, nulls handled, outliers removed",
                delay: 0,
              },
              {
                type: "pkl" as const,
                fileName: "final_model.pkl",
                description: "Best model: XGBoost (Accuracy: 94.2%)",
                delay: 0.1,
              },
              {
                type: "pdf" as const,
                fileName: "report.pdf",
                description: "Full EDA charts, evaluation metrics, agent explanations",
                delay: 0.2,
              },
            ].map((card, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ delay: 0.6 + card.delay, type: "spring", stiffness: 150 }}
                whileHover={{ y: -10, scale: 1.03 }}
              >
                <DownloadCard
                  type={card.type}
                  fileName={card.fileName}
                  description={card.description}
                />
              </motion.div>
            ))}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="bg-zinc-900/90 backdrop-blur-xl rounded-2xl border border-zinc-800 p-8 mb-8 shadow-2xl"
        >
          <motion.h3
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.9 }}
            className="text-2xl text-white mb-6 flex items-center gap-3"
          >
            <TrendingUp className="w-6 h-6 text-purple-400" />
            Model Comparison
          </motion.h3>
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
                {modelResults.map((result, index) => (
                  <motion.tr
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 1 + index * 0.1 }}
                    className={`border-zinc-800 transition-all duration-300 ${
                      result.status === "Best"
                        ? "bg-gradient-to-r from-green-500/10 to-emerald-500/10 hover:from-green-500/20 hover:to-emerald-500/20"
                        : "hover:bg-zinc-900/50"
                    }`}
                  >
                    <TableCell className="text-white font-medium">
                      <div className="flex items-center gap-2">
                        {result.status === "Best" && (
                          <motion.div
                            animate={{ rotate: [0, 360] }}
                            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                          >
                            <Sparkles className="w-4 h-4 text-green-400" />
                          </motion.div>
                        )}
                        {result.model}
                      </div>
                    </TableCell>
                    <TableCell className="text-gray-300">
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 1.2 + index * 0.1 }}
                        className={result.status === "Best" ? "text-green-400 font-semibold" : ""}
                      >
                        {result.accuracy}
                      </motion.span>
                    </TableCell>
                    <TableCell className="text-gray-300">
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 1.3 + index * 0.1 }}
                        className={result.status === "Best" ? "text-green-400 font-semibold" : ""}
                      >
                        {result.f1Score}
                      </motion.span>
                    </TableCell>
                    <TableCell>
                      <motion.span
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 1.4 + index * 0.1, type: "spring" }}
                        whileHover={{ scale: 1.1 }}
                        className={`inline-flex px-3 py-1 rounded-full text-xs font-semibold ${
                          result.status === "Best"
                            ? "bg-gradient-to-r from-green-500/20 to-emerald-500/20 text-green-400 border border-green-500/30"
                            : result.status === "Good"
                            ? "bg-gradient-to-r from-blue-500/20 to-cyan-500/20 text-blue-400 border border-blue-500/30"
                            : "bg-zinc-800 text-gray-400 border border-zinc-700"
                        }`}
                      >
                        {result.status}
                      </motion.span>
                    </TableCell>
                  </motion.tr>
                ))}
              </TableBody>
            </Table>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5 }}
          className="flex justify-center"
        >
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button
              onClick={handleRunAnother}
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
