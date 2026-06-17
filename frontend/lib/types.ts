// Mirrors the FastAPI contracts (src/adaptive_offers/api/schemas.py).

export interface Health {
  status: string;
  policy_loaded: boolean;
  feature_store_materialized: boolean;
  version: string | null;
}

export interface Policy {
  name: string;
  version: string;
  trained_on: string;
  metrics: Record<string, number | string>;
}

export interface Offer {
  offer_id: string;
  name: string;
  category: string;
  margin: number;
  suitability_tier: string;
}

export interface Reason {
  code: string;
  description: string;
}

export interface Decision {
  decision_id: string;
  ts: string;
  client_event_id: string | null;
  arm_id: string;
  arm_name: string;
  score: number;
  expected_reward: number;
  explored: boolean;
  policy_name: string;
  policy_version: string;
  eligible_arms: string[];
  reason_codes: string[];
  reasons: Reason[];
  estimates: Record<string, number>;
}

export interface AssistantAnswer {
  answer: string;
  provider: string;
  citations: { source: string; score: number; text: string }[];
}

export interface ContextInput {
  age: number;
  contact: "cellular" | "telephone";
  poutcome: "nonexistent" | "failure" | "success";
  euribor3m: number;
  default: "no" | "yes" | "unknown";
  loan: "no" | "yes" | "unknown";
  previously_contacted: number;
}
