"use client";

import { AnimatePresence, motion } from "framer-motion";
import { BarChart, Bot, Compass, Gift, Loader2, Percent, Sparkles, Target } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart as RechartBar, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Slider } from "@/components/ui/Slider";
import { Select } from "@/components/ui/Select";
import { ValueBreakdown, type OfferValue } from "@/components/ValueBreakdown";
import { api } from "@/lib/api";
import { brl, pct } from "@/lib/utils";
import type { AssistantAnswer, ContextInput, Decision, Offer } from "@/lib/types";

export function DecisionExplorer({ offers }: { offers: Offer[] }) {
  const [ctx, setCtx] = useState<ContextInput>({
    age: 66,
    contact: "cellular",
    poutcome: "success",
    euribor3m: 0.8,
    default: "no",
    loan: "no",
    previously_contacted: 1,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [explain, setExplain] = useState<AssistantAnswer | null>(null);

  const margins = Object.fromEntries(offers.map((o) => [o.offer_id, o]));

  async function decide() {
    setLoading(true);
    setError(null);
    try {
      const body = { ...ctx, previously_contacted: ctx.poutcome === "nonexistent" ? 0 : 1 };
      const d = await api.decide(body);
      const ex = await api.explain(body, "Por que esta oferta foi escolhida?");
      setDecision(d);
      setExplain(ex);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Falha ao decidir (API rodando?)");
    } finally {
      setLoading(false);
    }
  }

  const breakdown: OfferValue[] = decision
    ? decision.eligible_arms
        .map((id) => ({
          name: margins[id]?.name ?? id,
          value: Math.max(0, decision.estimates[id] ?? 0) * (margins[id]?.margin ?? 0),
          chosen: id === decision.arm_id,
        }))
        .sort((a, b) => b.value - a.value)
    : [];

  return (
    <Card>
      <CardTitle icon={<Compass className="h-4 w-4 text-primary-soft" />}>Explorador de decisão</CardTitle>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Slider label="Idade" value={ctx.age} min={18} max={95} onChange={(age) => setCtx({ ...ctx, age })} />
        <Select
          label="Resultado anterior"
          value={ctx.poutcome}
          options={["nonexistent", "failure", "success"]}
          onChange={(poutcome) => setCtx({ ...ctx, poutcome: poutcome as ContextInput["poutcome"] })}
        />
        <Select
          label="Em default?"
          value={ctx.default}
          options={["no", "yes", "unknown"]}
          onChange={(v) => setCtx({ ...ctx, default: v as ContextInput["default"] })}
        />
        <Select
          label="Canal"
          value={ctx.contact}
          options={["cellular", "telephone"]}
          onChange={(v) => setCtx({ ...ctx, contact: v as ContextInput["contact"] })}
        />
        <Slider
          label="Euribor 3m (juros)"
          value={ctx.euribor3m}
          min={0.6}
          max={5.1}
          step={0.1}
          onChange={(euribor3m) => setCtx({ ...ctx, euribor3m })}
        />
        <Select
          label="Tem empréstimo?"
          value={ctx.loan}
          options={["no", "yes", "unknown"]}
          onChange={(v) => setCtx({ ...ctx, loan: v as ContextInput["loan"] })}
        />
      </div>

      <button
        onClick={decide}
        disabled={loading}
        className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-primary py-3 font-bold text-white shadow-lg shadow-primary/30 transition hover:brightness-110 disabled:opacity-60"
      >
        {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
        {loading ? "Decidindo…" : "Decidir oferta"}
      </button>

      {error && <p className="mt-3 text-sm text-danger">⚠️ {error}</p>}

      <AnimatePresence>
        {decision && (
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.45, ease: "easeOut" }}
            className="mt-5 space-y-4"
          >
            {/* headline */}
            <div className="card flex items-center justify-between p-5">
              <div className="flex items-center gap-3">
                <Gift className="h-7 w-7 text-primary-soft" />
                <div>
                  <div className="text-xl font-extrabold text-primary-soft">{decision.arm_name}</div>
                  <div className="mt-0.5 text-xs text-muted">
                    {decision.explored ? "🔍 exploração" : "🎯 explotação"} · política{" "}
                    <b className="text-text">
                      {decision.policy_name}@{decision.policy_version}
                    </b>{" "}
                    · {decision.eligible_arms.length} de {offers.length} elegíveis
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-extrabold text-success">{brl(decision.expected_reward)}</div>
                <div className="text-[0.72rem] text-muted">valor esperado</div>
              </div>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {decision.reason_codes.map((c) => (
                <Badge key={c}>{c}</Badge>
              ))}
            </div>

            {/* value breakdown + probability + reasons */}
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-6">
              <div className="card p-4 lg:col-span-2">
                <CardTitle icon={<Target className="h-4 w-4 text-primary-soft" />}>
                  Valor esperado por oferta
                </CardTitle>
                <ValueBreakdown data={breakdown} />
              </div>
              <div className="card p-4 lg:col-span-2">
                <CardTitle icon={<Percent className="h-4 w-4 text-primary-soft" />}>
                  P(conversão) por oferta
                </CardTitle>
                <ResponsiveContainer width="100%" height={Math.max(180, breakdown.length * 42)}>
                  <RechartBar data={breakdown} layout="vertical" margin={{ left: 8, right: 28, top: 4, bottom: 4 }}>
                    <XAxis type="number" hide />
                    <YAxis type="category" dataKey="name" width={150} tick={{ fill: "#a1a1aa", fontSize: 12 }} axisLine={false} tickLine={false} />
                    <Tooltip
                      cursor={{ fill: "rgba(255,255,255,0.04)" }}
                      contentStyle={{ background: "#0a0a0a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, color: "#ededed" }}
                      formatter={(v: number) => [pct(v / 100), "probabilidade"]}
                    />
                    <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={22}>
                      {breakdown.map((d, i) => (
                        <Cell key={i} fill={d.chosen ? "#34d399" : "#0070f3"} fillOpacity={d.chosen ? 1 : 0.6} />
                      ))}
                    </Bar>
                  </RechartBar>
                </ResponsiveContainer>
              </div>
              <div className="card p-4 lg:col-span-2">
                <CardTitle icon={<Sparkles className="h-4 w-4 text-primary-soft" />}>Por que esta decisão</CardTitle>
                <ul className="space-y-2">
                  {decision.reasons.map((r) => (
                    <li key={r.code} className="text-sm">
                      <b className="text-primary-soft">{r.code}</b>
                      <span className="text-muted"> — {r.description}</span>
                    </li>
                  ))}
                </ul>
                {/* decomposition visual */}
                {decision.estimates && decision.arm_id && (
                  <div className="mt-4 rounded-xl border border-border bg-surface2 p-3 text-center text-xs">
                    <div className="mb-2 text-muted">Valor = P(conv) × Margem</div>
                    <div className="flex items-center justify-around gap-1">
                      <div>
                        <div className="text-lg font-extrabold text-primary-soft">
                          {((decision.estimates[decision.arm_id] ?? 0) / 100).toFixed(1)}%
                        </div>
                        <div className="text-muted">P(conv)</div>
                      </div>
                      <span className="text-lg text-muted">×</span>
                      <div>
                        <div className="text-lg font-extrabold text-success">
                          {brl(decision.expected_reward / Math.max(0.01, (decision.estimates[decision.arm_id] ?? 1) / 100))}
                        </div>
                        <div className="text-muted">Margem</div>
                      </div>
                      <span className="text-lg text-muted">=</span>
                      <div>
                        <div className="text-lg font-extrabold text-success">{brl(decision.expected_reward)}</div>
                        <div className="text-muted">Valor</div>
                      </div>
                    </div>
                  </div>
                )}
                {/* explore/exploit indicator */}
                <div className={`mt-3 rounded-xl border p-3 text-center ${
                  decision.explored ? "border-amber/30 bg-amber/10" : "border-success/30 bg-success/10"
                }`}>
                  <div className={`text-base font-extrabold ${decision.explored ? "text-amber" : "text-success"}`}>
                    {decision.explored ? "🔍 Exploração" : "🎯 Explotação"}
                  </div>
                  <div className="mt-0.5 text-[0.7rem] text-muted">
                    {decision.explored
                      ? "Testando alternativa promissora para aprender"
                      : "Usando a melhor estimativa atual"}
                  </div>
                </div>
              </div>
            </div>

            {/* assistant */}
            {explain && (
              <div className="card p-5">
                <CardTitle icon={<Bot className="h-4 w-4 text-primary-soft" />}>
                  Assistente (LLM + RAG) · {explain.provider}
                </CardTitle>
                <p className="whitespace-pre-line text-sm leading-relaxed text-text">{explain.answer}</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}
