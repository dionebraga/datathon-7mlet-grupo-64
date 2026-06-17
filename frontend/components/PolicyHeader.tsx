"use client";

import { motion } from "framer-motion";
import { Activity, Database, Radio, Satellite } from "lucide-react";
import type { Health, Policy } from "@/lib/types";

function Status({ on, label }: { on: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-bold ${
        on ? "border-success/30 bg-success/10 text-success" : "border-border bg-white/5 text-muted"
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${on ? "bg-success" : "bg-muted"}`} />
      {label}
    </span>
  );
}

export function PolicyHeader({ health, policy }: { health?: Health; policy?: Policy }) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="card flex flex-col gap-3 px-6 py-5 sm:flex-row sm:items-center sm:justify-between"
    >
      <div className="flex items-center gap-3">
        <div className="grid h-11 w-11 place-items-center rounded-xl bg-primary/15 text-primary-soft">
          <Satellite className="h-5 w-5" />
        </div>
        <div>
          <h1 className="text-xl font-extrabold tracking-tight">Adaptive Offers — Decision Console</h1>
          <p className="text-sm text-muted">
            Multi-armed bandit · política ativa{" "}
            <b className="text-text">
              {policy ? `${policy.name}@${policy.version}` : "—"}
            </b>
          </p>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Status on={!!health?.policy_loaded} label="Política" />
        <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-white/5 px-3 py-1 text-xs font-bold text-muted">
          <Database className="h-3.5 w-3.5" /> Feature Store {health?.feature_store_materialized ? "✓" : "—"}
        </span>
        <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-white/5 px-3 py-1 text-xs font-bold text-muted">
          {health?.status === "ok" ? <Radio className="h-3.5 w-3.5 text-success" /> : <Activity className="h-3.5 w-3.5" />}
          API {health?.status ?? "?"}
        </span>
      </div>
    </motion.header>
  );
}
