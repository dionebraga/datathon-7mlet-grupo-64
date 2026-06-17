// Thin client for the Adaptive Offers FastAPI. Requests go to /api/* which
// next.config.mjs proxies to the Python backend (no CORS, backend untouched).
import type { AssistantAnswer, ContextInput, Decision, Health, Offer, Policy } from "./types";

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json();
}

export const api = {
  health: () => get<Health>("/health"),
  policy: () => get<Policy>("/policy"),
  offers: () => get<Offer[]>("/offers"),
  decide: (ctx: ContextInput) => post<Decision>("/decide", ctx),
  explain: (ctx: ContextInput, question: string) =>
    post<AssistantAnswer>(`/assistant/explain?question=${encodeURIComponent(question)}`, ctx),
};
