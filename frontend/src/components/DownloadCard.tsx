"use client";

import { FileSpreadsheet, FileArchive, FileText, Download } from "lucide-react";
import { motion } from "motion/react";
import { Button } from "@/components/ui/button";

type DownloadType = "csv" | "pkl" | "pdf";

type DownloadCardProps = {
  type: DownloadType;
  fileName: string;
  description: string;
};

const styles: Record<
  DownloadType,
  { icon: typeof FileSpreadsheet; gradient: string; accent: string; buttonGradient: string; buttonGlow: string }
> = {
  csv: {
    icon: FileSpreadsheet,
    gradient: "from-emerald-500/20 to-green-500/20",
    accent: "text-emerald-400",
    buttonGradient: "from-indigo-500 via-blue-500 to-cyan-500",
    buttonGlow: "shadow-blue-500/30",
  },
  pkl: {
    icon: FileArchive,
    gradient: "from-purple-500/20 to-pink-500/20",
    accent: "text-purple-400",
    buttonGradient: "from-indigo-500 via-blue-500 to-cyan-500",
    buttonGlow: "shadow-blue-500/30",
  },
  pdf: {
    icon: FileText,
    gradient: "from-cyan-500/20 to-blue-500/20",
    accent: "text-cyan-400",
    buttonGradient: "from-indigo-500 via-blue-500 to-cyan-500",
    buttonGlow: "shadow-blue-500/30",
  },
};

export function DownloadCard({ type, fileName, description }: DownloadCardProps) {
  const style = styles[type];
  const Icon = style.icon;

  return (
    <div
      className="aspect-[11/5] rounded-2xl border border-zinc-800 bg-zinc-900/90 p-5 backdrop-blur-xl flex flex-col"
    >
      <div className="mb-4 flex items-center justify-between">
        <div className="rounded-lg border border-zinc-700 bg-zinc-900/80 p-2">
          <Icon className={`h-5 w-5 ${style.accent}`} />
        </div>
        <span className="rounded-full border border-zinc-700 px-2 py-1 text-xs uppercase tracking-wide text-zinc-400">
          {type}
        </span>
      </div>
      <p className="mb-2 text-base font-semibold text-white">{fileName}</p>
      <p className="mb-4 grow text-sm text-zinc-400">{description}</p>
      <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
        <Button
          className={`h-12 w-full rounded-xl border-0 bg-gradient-to-r ${style.buttonGradient} hover:from-indigo-400 hover:via-blue-400 hover:to-cyan-400 text-white font-semibold shadow-lg ${style.buttonGlow} relative overflow-hidden`}
        >
          <motion.div
            aria-hidden
            className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/25 to-transparent"
            animate={{ x: ["0%", "210%"] }}
            transition={{ duration: 1.5, repeat: Infinity, repeatDelay: 1.5, ease: "easeInOut" }}
          />
          <span className="relative z-10 flex items-center gap-2">
            <motion.span animate={{ y: [0, -1, 0] }} transition={{ duration: 1.4, repeat: Infinity }}>
              <Download className="h-4 w-4" />
            </motion.span>
            Download
          </span>
        </Button>
      </motion.div>
    </div>
  );
}
