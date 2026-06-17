"use client";

import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";

export function MetricCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
  delay = 0,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  sub?: string;
  color: string;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      whileHover={{ y: -4 }}
      className="card p-5"
    >
      <div className="flex items-center gap-2" style={{ color }}>
        <Icon className="h-4 w-4" />
        <span className="text-[0.72rem] font-bold uppercase tracking-wider text-muted">{label}</span>
      </div>
      <div className="mt-2 text-3xl font-extrabold tracking-tight" style={{ color }}>
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-muted">{sub}</div>}
    </motion.div>
  );
}
