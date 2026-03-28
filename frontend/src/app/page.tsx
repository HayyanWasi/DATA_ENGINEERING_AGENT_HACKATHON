"use client";

import { useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Database,
  Brain,
  FileText,
  Sparkles,
  CloudUpload,
  Zap,
  ArrowRight,
  FileSpreadsheet,
} from "lucide-react";
import { motion } from "motion/react";
import { Navbar } from "@/components/Navbar";
import { importDataset } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function Upload() {
  const router = useRouter();
  const [fileName, setFileName] = useState<string>("");
  const [sqlConnection, setSqlConnection] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isHovering, setIsHovering] = useState(false);
  const [datasetId, setDatasetId] = useState<string>("");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [apiKey, setApiKey] = useState<string>("");
  const [apiKeyError, setApiKeyError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = useCallback((file: File) => {
    setIsUploading(true);
    setUploadProgress(0);
    setUploadError(null);

    // Animate progress to 80% while upload is in flight
    const interval = setInterval(() => {
      setUploadProgress((prev) => (prev >= 80 ? prev : prev + 4));
    }, 80);

    importDataset(file)
      .then((res) => {
        clearInterval(interval);
        setUploadProgress(100);
        setTimeout(() => {
          setDatasetId(res.dataset_id);
          setFileName(file.name);
          setColumns(res.columns);
          setTargetColumn(res.columns[res.columns.length - 1] ?? "");
          setIsUploading(false);
          setUploadProgress(0);
        }, 300);
      })
      .catch((err) => {
        clearInterval(interval);
        setIsUploading(false);
        setUploadProgress(0);
        setUploadError(String(err));
        // Still allow manual filename entry as fallback
        setFileName(file.name);
      });
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith(".csv") || file.name.endsWith(".xlsx"))) {
      handleFileUpload(file);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleRunPipeline = () => {
    if (!fileName && !sqlConnection) return;
    setApiKeyError(null);
    if (!apiKey.trim()) {
      setApiKeyError("Please enter your LLM API key to run the pipeline.");
      return;
    }
    const sourceName = fileName || "database_connection";

    if (datasetId) {
      const query = new URLSearchParams({ fileName: sourceName, targetColumn, apiKey: apiKey.trim() }).toString();
      router.push(`/pipeline/${datasetId}?${query}`);
    } else {
      const query = new URLSearchParams({ fileName: sourceName }).toString();
      router.push(`/results?${query}`);
    }
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
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-full blur-3xl"
        />
        <motion.div
          animate={{
            scale: [1.3, 1, 1.3],
            x: [0, -100, 0],
            y: [0, -50, 0],
            opacity: [0.15, 0.25, 0.15],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          className="absolute bottom-1/4 right-1/4 w-[600px] h-[600px] bg-gradient-to-br from-cyan-600/20 to-blue-600/20 rounded-full blur-3xl"
        />

        <div className="absolute inset-0 bg-[linear-gradient(rgba(139,92,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(139,92,246,0.03)_1px,transparent_1px)] bg-[size:100px_100px]" />
      </div>

      <Navbar />

      <main className="mx-auto max-w-6xl px-6 py-16 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="inline-block mb-6"
          >
            <div className="relative">
              <motion.div
                animate={{
                  rotate: 360,
                }}
                transition={{
                  duration: 20,
                  repeat: Infinity,
                  ease: "linear",
                }}
                className="absolute inset-0 bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 rounded-full blur-2xl opacity-50"
              />
              <motion.div
                animate={{
                  scale: [1, 1.1, 1],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                }}
              >
                <Sparkles className="w-16 h-16 text-white relative z-10" />
              </motion.div>
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="text-6xl mb-4 font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent"
          >
            Turn raw data into a trained ML model
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.6 }}
            className="text-xl text-gray-400"
          >
            Created by Team Coloners — a multi-agent ML system
          </motion.p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          className="mb-12"
        >
          <motion.div
            onMouseEnter={() => setIsHovering(true)}
            onMouseLeave={() => setIsHovering(false)}
            className="relative"
          >
            <motion.div
              className="relative bg-gradient-to-br from-[#1A1D29] to-[#141720] backdrop-blur-2xl rounded-3xl border border-zinc-800/50 overflow-hidden"
              animate={{
                boxShadow: isHovering
                  ? "0 40px 100px rgba(139, 92, 246, 0.3), 0 0 80px rgba(236, 72, 153, 0.2)"
                  : "0 20px 60px rgba(0, 0, 0, 0.5)",
              }}
              transition={{ duration: 0.3 }}
            >
              <motion.div
                animate={{
                  background: [
                    "linear-gradient(135deg, transparent 0%, rgba(139, 92, 246, 0.1) 50%, transparent 100%)",
                    "linear-gradient(225deg, transparent 0%, rgba(236, 72, 153, 0.1) 50%, transparent 100%)",
                    "linear-gradient(315deg, transparent 0%, rgba(6, 182, 212, 0.1) 50%, transparent 100%)",
                  ],
                }}
                transition={{ duration: 5, repeat: Infinity }}
                className="absolute inset-0 pointer-events-none"
              />

              <div className="absolute inset-0 opacity-50">
                <motion.div
                  animate={{
                    rotate: 360,
                  }}
                  transition={{
                    duration: 10,
                    repeat: Infinity,
                    ease: "linear",
                  }}
                  className="absolute -inset-[2px] bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 rounded-3xl blur-sm"
                  style={{ zIndex: -1 }}
                />
              </div>

              <div className="relative p-12">
                <motion.div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  animate={{
                    scale: isDragging ? 1.02 : 1,
                  }}
                  className="relative mb-8"
                >
                  {isUploading ? (
                    <UploadingState fileName={fileName} progress={uploadProgress} />
                  ) : fileName ? (
                    <UploadedState fileName={fileName} onReplace={handleFileInput} fileInputRef={fileInputRef} />
                  ) : (
                    <EmptyState
                      isDragging={isDragging}
                      isHovering={isHovering}
                      onFileSelect={handleFileInput}
                      fileInputRef={fileInputRef}
                    />
                  )}
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  transition={{ delay: 0.6 }}
                  className="mb-8"
                >
                  <div className="flex items-center gap-3 mb-6">
                    <div className="h-px flex-1 bg-gradient-to-r from-transparent via-purple-500/30 to-transparent" />
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      className="px-4 py-2 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 backdrop-blur-xl rounded-full border border-purple-500/30"
                    >
                      <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold flex items-center gap-2">
                        <Database className="w-3 h-3" />
                        Or connect database
                      </span>
                    </motion.div>
                    <div className="h-px flex-1 bg-gradient-to-r from-transparent via-purple-500/30 to-transparent" />
                  </div>

                  <FrostedInput
                    id="sql-connection"
                    icon={<Database className="w-4 h-4 text-cyan-400" />}
                    placeholder="postgresql://user:password@host:port/database"
                    label="SQL Connection String"
                    value={sqlConnection}
                    onChange={(e) => setSqlConnection(e.target.value)}
                    delay={0.7}
                  />
                </motion.div>

                {/* API Key Input */}
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.65 }}
                  className="mb-6 p-4 rounded-xl border border-yellow-500/30 bg-yellow-500/5"
                >
                  <label className="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
                    <svg className="w-4 h-4 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                    </svg>
                    LLM API Key <span className="text-red-400">*</span>
                  </label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => { setApiKey(e.target.value); setApiKeyError(null); }}
                    placeholder="AIzaSy..."
                    className="w-full bg-zinc-900 border border-zinc-700 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-yellow-500 placeholder-zinc-600"
                  />
                  <p className="text-xs text-gray-500 mt-1.5">Enter your LLM API key to power the pipeline</p>
                </motion.div>

                {apiKeyError && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-4 p-3 rounded-xl border border-red-500/30 bg-red-500/10 flex items-center gap-2"
                  >
                    <svg className="w-4 h-4 text-red-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                    </svg>
                    <p className="text-red-300 text-sm">{apiKeyError}</p>
                  </motion.div>
                )}

                {columns.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-6 p-4 rounded-xl border border-purple-500/30 bg-purple-500/10"
                  >
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Target Column (what to predict)
                    </label>
                    <select
                      value={targetColumn}
                      onChange={(e) => setTargetColumn(e.target.value)}
                      className="w-full bg-zinc-900 border border-zinc-700 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-purple-500"
                    >
                      {columns.map((col) => (
                        <option key={col} value={col}>
                          {col.trim()}
                        </option>
                      ))}
                    </select>
                  </motion.div>
                )}

                {uploadError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-4 p-4 rounded-xl border border-red-500/30 bg-red-500/10 flex items-start gap-3"
                  >
                    <svg className="w-5 h-5 text-red-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                    </svg>
                    <div>
                      <p className="text-red-300 font-medium text-sm">Upload failed — backend may not be running</p>
                      <p className="text-red-400/70 text-xs mt-1">{uploadError}</p>
                    </div>
                  </motion.div>
                )}

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.8 }}
                  className="mt-8"
                >
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleRunPipeline}
                    disabled={!fileName && !sqlConnection}
                    type="button"
                    className="w-full bg-gradient-to-r from-purple-600 via-pink-600 to-cyan-600 hover:from-purple-500 hover:via-pink-500 hover:to-cyan-500 text-white py-7 text-lg font-bold rounded-2xl shadow-2xl shadow-purple-500/40 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-300 relative overflow-hidden group border-0 inline-flex items-center justify-center gap-2"
                  >
                      <motion.div
                        animate={{
                          x: ["-200%", "200%"],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          repeatDelay: 1,
                          ease: "easeInOut",
                        }}
                        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                      />

                      <span className="relative z-10 flex items-center justify-center gap-3">
                        <Zap className="w-5 h-5" />
                        Run AutoML Pipeline
                        <motion.span
                          animate={{ x: [0, 5, 0] }}
                          transition={{ duration: 1.5, repeat: Infinity }}
                        >
                          <ArrowRight className="w-5 h-5" />
                        </motion.span>
                      </span>
                    </motion.button>
                </motion.div>
              </div>
            </motion.div>

            <div
              className="absolute -inset-4 bg-gradient-to-br from-purple-600/5 to-cyan-600/5 rounded-3xl blur-2xl -z-10"
              style={{ transform: "translateZ(-100px)" }}
            />
          </motion.div>
        </motion.div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx"
          onChange={handleFileInput}
          className="hidden"
        />

        <FeaturePills />
      </main>
    </div>
  );
}

function EmptyState({
  isDragging,
  isHovering,
  onFileSelect,
  fileInputRef,
}: {
  isDragging: boolean;
  isHovering: boolean;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
}) {
  return (
    <motion.div
      animate={{
        backgroundColor: isDragging ? "rgba(168, 85, 247, 0.05)" : "rgba(0, 0, 0, 0)",
      }}
      className="relative rounded-2xl p-20 text-center group"
    >
      {isHovering && (
        <>
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0 }}
              animate={{
                opacity: [0, 1, 0],
                scale: [0, 1, 0],
                x: [0, Math.cos((i / 20) * Math.PI * 2) * 100],
                y: [0, Math.sin((i / 20) * Math.PI * 2) * 100],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: i * 0.1,
                ease: "easeOut",
              }}
              className="absolute top-1/2 left-1/2 w-1 h-1 bg-purple-400 rounded-full"
            />
          ))}
        </>
      )}

      <motion.div
        animate={{
          y: isDragging ? -10 : [0, -15, 0],
          scale: isDragging ? 1.1 : 1,
        }}
        transition={{
          y: { duration: 3, repeat: Infinity, ease: "easeInOut" },
          scale: { duration: 0.3 },
        }}
        className="relative inline-block mb-8"
      >
        <motion.div
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.3, 0.6, 0.3],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
          }}
          className="absolute inset-0 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-full blur-3xl"
        />

        <div className="relative w-24 h-24 flex items-center justify-center">
          <CloudUpload className="w-24 h-24 text-purple-400" strokeWidth={1.5} />
        </div>
      </motion.div>

      <motion.h3
        animate={{
          backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
        }}
        transition={{
          duration: 5,
          repeat: Infinity,
          ease: "linear",
        }}
        className="text-2xl font-bold mb-3 bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent"
        style={{ backgroundSize: "200% auto" }}
      >
        {isDragging ? "Drop it here!" : "Upload your dataset"}
      </motion.h3>

      <p className="text-gray-500 mb-8">
        Drag & drop your CSV or XLSX file here, or click to browse
        <br />
        <span className="text-sm text-gray-600">Maximum file size: 50MB</span>
      </p>

      <motion.div whileHover={{ scale: 1.05, y: -2 }} whileTap={{ scale: 0.95 }}>
        <Button 
          onClick={() => fileInputRef.current?.click()}
          className="cursor-pointer bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 text-white px-10 py-6 rounded-xl shadow-lg shadow-purple-500/30 border-0 text-base font-semibold relative overflow-hidden group"
        >
          <span className="relative z-10 flex items-center gap-2">
            <FileSpreadsheet className="w-5 h-5" />
            Browse Files
          </span>
        </Button>
      </motion.div>
    </motion.div>
  );
}

function UploadingState({ fileName, progress }: { fileName: string; progress: number }) {
  const circumference = 2 * Math.PI * 60;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center py-12 relative"
    >
      {[...Array(30)].map((_, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, scale: 0 }}
          animate={{
            opacity: [0, 1, 0],
            scale: [0, 1, 0],
            x: Math.cos((i / 30) * Math.PI * 2) * 150,
            y: Math.sin((i / 30) * Math.PI * 2) * 150,
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            delay: i * 0.03,
            ease: "easeOut",
          }}
          className="absolute w-1 h-1 rounded-full"
          style={{
            background: `linear-gradient(${(i / 30) * 360}deg, #a855f7, #ec4899, #06b6d4)`,
          }}
        />
      ))}

      <div className="relative mb-8">
        <svg className="w-40 h-40 transform -rotate-90">
          <circle
            cx="80"
            cy="80"
            r="60"
            stroke="rgba(139, 92, 246, 0.1)"
            strokeWidth="8"
            fill="none"
          />
          <motion.circle
            cx="80"
            cy="80"
            r="60"
            stroke="url(#progressGradient)"
            strokeWidth="8"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 0.3 }}
          />
          <defs>
            <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#a855f7" />
              <stop offset="50%" stopColor="#ec4899" />
              <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>
          </defs>
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            key={progress}
            initial={{ scale: 1.2, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="text-4xl font-bold text-white"
          >
            {progress}%
          </motion.span>
          <span className="text-xs text-gray-400 mt-1">Uploading</span>
        </div>
      </div>

      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="px-6 py-3 bg-gradient-to-r from-purple-500/20 to-cyan-500/20 backdrop-blur-xl rounded-full border border-purple-500/30"
      >
        <p className="text-sm text-white font-medium flex items-center gap-2">
          <FileSpreadsheet className="w-4 h-4" />
          {fileName || "Processing..."}
        </p>
      </motion.div>

      <div className="w-full max-w-md h-2 bg-zinc-900/50 rounded-full overflow-hidden mt-6">
        <motion.div
          className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 relative overflow-hidden"
          initial={{ width: "0%" }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3 }}
        >
          <motion.div
            animate={{
              x: ["-100%", "200%"],
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: "linear",
            }}
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent"
          />
        </motion.div>
      </div>
    </motion.div>
  );
}

function UploadedState({
  fileName,
  onReplace,
  fileInputRef,
}: {
  fileName: string;
  onReplace: (e: React.ChangeEvent<HTMLInputElement>) => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center py-12"
    >
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 200, delay: 0.2 }}
        className="relative mb-6"
      >
        <motion.div
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.5, 0.8, 0.5],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
          }}
          className="absolute inset-0 bg-gradient-to-br from-green-500 to-emerald-500 rounded-3xl blur-2xl"
        />

        <div className="relative w-24 h-24 rounded-2xl bg-gradient-to-br from-green-500/20 to-emerald-500/20 border border-green-500/30 flex items-center justify-center backdrop-blur-sm">
          <FileSpreadsheet className="w-12 h-12 text-green-400" />
        </div>

        <motion.div
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: "spring", stiffness: 200, delay: 0.3 }}
          className="absolute -top-2 -right-2 w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-500 rounded-full flex items-center justify-center shadow-lg"
        >
          <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
          </svg>
        </motion.div>
      </motion.div>

      <motion.h3
        initial={{ y: 10, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="text-2xl font-bold text-white mb-2"
      >
        {fileName}
      </motion.h3>

      <motion.div
        initial={{ y: 10, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="flex items-center gap-2 text-green-400 mb-6"
      >
        <motion.div
          animate={{ scale: [1, 1.3, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="w-2 h-2 bg-green-400 rounded-full"
        />
        <span className="text-sm font-medium">Ready to process</span>
      </motion.div>

      <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
        <Button
          onClick={() => fileInputRef.current?.click()}
          variant="ghost"
          className="text-gray-400 hover:text-white hover:bg-white/5 cursor-pointer border border-zinc-700 hover:border-zinc-600"
        >
          Replace file
        </Button>
      </motion.div>
    </motion.div>
  );
}

function FrostedInput({
  id,
  icon,
  label,
  placeholder,
  value,
  onChange,
  delay = 0,
}: {
  id: string;
  icon: React.ReactNode;
  label: string;
  placeholder: string;
  value?: string;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="group"
    >
      <Label htmlFor={id} className="text-gray-400 mb-3 flex items-center gap-2 text-sm font-medium">
        {icon}
        {label}
      </Label>

      <div className="relative">
        <motion.div whileFocus={{ scale: 1.01 }} className="relative">
          <div className="absolute left-4 top-1/2 -translate-y-1/2 z-10">{icon}</div>

          <Input
            id={id}
            placeholder={placeholder}
            value={value}
            onChange={onChange}
            className="bg-[#1A1D29]/60 backdrop-blur-xl border-zinc-700/50 text-white placeholder:text-gray-600 focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20 transition-all h-14 pl-12 rounded-xl text-base"
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          className="absolute inset-0 rounded-xl bg-gradient-to-r from-purple-500/10 to-cyan-500/10 pointer-events-none opacity-0 group-focus-within:opacity-100 transition-opacity blur-xl"
        />
      </div>
    </motion.div>
  );
}

function FeaturePills() {
  const features = [
    { icon: Brain, text: "3 Models Trained", gradient: "from-purple-500 to-pink-500", delay: 0 },
    { icon: FileText, text: "PDF Report", gradient: "from-orange-500 to-yellow-500", delay: 0.1 },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1 }}
      className="flex gap-4 justify-center flex-wrap"
    >
      {features.map((item, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, scale: 0.8, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ delay: 1.1 + item.delay, type: "spring", stiffness: 150 }}
          whileHover={{ scale: 1.1, y: -5 }}
          className="relative group cursor-pointer"
        >
          <motion.div
            whileHover={{
              boxShadow: "0 0 30px rgba(139, 92, 246, 0.4)",
            }}
            className="flex items-center gap-3 px-6 py-4 bg-[#1A1D29]/80 backdrop-blur-xl rounded-full border border-zinc-700/50 hover:border-zinc-600 transition-all"
          >
            <div
              className={`w-8 h-8 rounded-full bg-gradient-to-br ${item.gradient} opacity-20 flex items-center justify-center`}
            >
              <item.icon
                className={`w-5 h-5 bg-gradient-to-r ${item.gradient} bg-clip-text text-transparent`}
                strokeWidth={2.5}
              />
            </div>
            <span className="text-sm text-gray-300 group-hover:text-white transition-colors font-medium">
              {item.text}
            </span>
          </motion.div>
        </motion.div>
      ))}
    </motion.div>
  );
}
