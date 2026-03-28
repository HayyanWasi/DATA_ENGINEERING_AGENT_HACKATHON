"use client";

import { ReactNode } from "react";
import { motion } from "motion/react";

type StatTileProps = {
  label: string;
  value: string;
  icon: ReactNode;
  delay?: number;
};

export function StatTile({ label, value, icon, delay = 0 }: StatTileProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ delay: 0.3 + delay, type: "spring", stiffness: 200 }}
      whileHover={{ scale: 1.05, y: -5 }}
      className="group rounded-xl border border-zinc-800 bg-zinc-900/90 p-5 shadow-lg backdrop-blur-sm transition-all duration-300 hover:border-purple-500/30"
    >
      <motion.div
        className="mb-2 text-purple-400 transition-colors group-hover:text-pink-400"
        animate={{ rotate: [0, 5, -5, 0] }}
        transition={{ duration: 3, repeat: Infinity, delay }}
      >
        {icon}
      </motion.div>
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.5 + delay }}
        className="mb-1 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-3xl text-transparent"
      >
        {value}
      </motion.div>
      <div className="text-sm text-gray-400">{label}</div>
    </motion.div>
  );
}
