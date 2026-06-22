"use client";

import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { TrendingDown, TrendingUp } from "lucide-react";

export function MetricCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
  delay = 0,
  trend,
  trendLabel,
  trendUp = true,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  sub?: string;
  color: string;
  delay?: number;
  trend?: string;
  trendLabel?: string;
  trendUp?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      whileHover={{ y: -4, scale: 1.01 }}
      className="card p-5 transition-shadow duration-200 hover:shadow-lg hover:shadow-primary/5"
    >
      <div className="flex items-center gap-2" style={{ color }}>
        <Icon className="h-4 w-4" />
        <span className="text-[0.72rem] font-bold uppercase tracking-wider text-muted">{label}</span>
      </div>
      <div className="mt-2 text-3xl font-extrabold tracking-tight" style={{ color }}>
        {value}
      </div>
      <div className="mt-1 flex items-center gap-2">
        {sub && <div className="text-xs text-muted">{sub}</div>}
        {trend && (
          <span
            className={`inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[0.65rem] font-bold ${
              trendUp ? "text-success" : "text-danger"
            } ${trendUp ? "bg-success/10" : "bg-danger/10"}`}
          >
            {trendUp ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
            {trend}
          </span>
        )}
      </div>
      {trendLabel && <div className="mt-0.5 text-[0.62rem] text-muted/60">{trendLabel}</div>}
    </motion.div>
  );
}
