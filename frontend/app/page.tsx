"use client";

import { Activity, Banknote, Gauge, Search, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { MetricCard } from "@/components/MetricCard";
import { PolicyHeader } from "@/components/PolicyHeader";
import { OffersGrid } from "@/components/OffersGrid";
import { DecisionExplorer } from "@/components/DecisionExplorer";
import { api } from "@/lib/api";
import { brl, pct } from "@/lib/utils";
import type { Health, Offer, Policy } from "@/lib/types";

export default function Page() {
  const [health, setHealth] = useState<Health>();
  const [policy, setPolicy] = useState<Policy>();
  const [offers, setOffers] = useState<Offer[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [h, p, o] = await Promise.all([api.health(), api.policy(), api.offers()]);
        setHealth(h);
        setPolicy(p);
        setOffers(o);
      } catch (e) {
        setErr(e instanceof Error ? e.message : "Falha ao conectar na API");
      }
    })();
  }, []);

  const m = (policy?.metrics ?? {}) as Record<string, number>;

  return (
    <main className="mx-auto max-w-7xl space-y-5 px-4 py-6 lg:px-8">
      <PolicyHeader health={health} policy={policy} />

      {err && (
        <div className="card flex items-center gap-2 border-danger/40 p-4 text-danger">
          <TriangleAlert className="h-5 w-5" />
          <span>
            {err} — inicie a API: <code className="rounded bg-white/5 px-1.5 py-0.5">adaptive-offers serve</code>
          </span>
        </div>
      )}

      <section className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard icon={Banknote} label="Reward / 1k" value={policy ? brl(m.reward_per_1k ?? 0) : "—"}
          sub="valor por 1k impressões" color="#a5b4fc" delay={0.05} />
        <MetricCard icon={Gauge} label="Regret ratio" value={policy ? pct(m.regret_ratio ?? 0) : "—"}
          sub="distância do ótimo" color="#fb7185" delay={0.1} />
        <MetricCard icon={Activity} label="Conversão" value={policy ? pct(m.conversion_rate ?? 0) : "—"}
          sub="conversões / impressões" color="#2dd4bf" delay={0.15} />
        <MetricCard icon={Search} label="Exploração" value={policy ? pct(m.exploration_rate ?? 0) : "—"}
          sub="decisões exploratórias" color="#34d399" delay={0.2} />
      </section>

      <DecisionExplorer offers={offers} />
      <OffersGrid offers={offers} />

      <footer className="pt-2 text-center text-xs text-muted">
        <b className="text-text">Adaptive Offers Platform</b> · © 2026{" "}
        <b className="text-primary-soft">Dione Braga</b> — Grupo 64 · FIAP Pós-Tech 7MLET · Licença MIT
      </footer>
    </main>
  );
}
