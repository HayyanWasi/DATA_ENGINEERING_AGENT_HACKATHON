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

const styles: Record<DownloadType, { icon: typeof FileSpreadsheet; gradient: string; accent: string }> = {
  csv: {
    icon: FileSpreadsheet,
    gradient: "from-emerald-500/20 to-green-500/20",
    accent: "text-emerald-400",
  },
  pkl: {
    icon: FileArchive,
    gradient: "from-purple-500/20 to-pink-500/20",
    accent: "text-purple-400",
  },
  pdf: {
    icon: FileText,
    gradient: "from-cyan-500/20 to-blue-500/20",
    accent: "text-cyan-400",
  },
};

export function DownloadCard({ type, fileName, description }: DownloadCardProps) {
  const style = styles[type];
  const Icon = style.icon;

  return (
    <div className={`rounded-2xl border border-zinc-800 bg-gradient-to-br ${style.gradient} p-5 backdrop-blur-xl`}>
      <div className="mb-4 flex items-center justify-between">
        <div className="rounded-lg border border-zinc-700 bg-zinc-900/80 p-2">
          <Icon className={`h-5 w-5 ${style.accent}`} />
        </div>
        <span className="rounded-full border border-zinc-700 px-2 py-1 text-xs uppercase tracking-wide text-zinc-400">
          {type}
        </span>
      </div>
      <p className="mb-2 text-base font-semibold text-white">{fileName}</p>
      <p className="mb-4 text-sm text-zinc-400">{description}</p>
      <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
        <Button className="w-full border border-zinc-700 bg-zinc-900/80 text-zinc-100 hover:bg-zinc-800">
          <Download className="h-4 w-4" />
          Download
        </Button>
      </motion.div>
    </div>
  );
}
