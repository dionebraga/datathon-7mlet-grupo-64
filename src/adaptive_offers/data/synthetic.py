"""Synthetic adaptive-experimentation layer (Stage 2).

On top of the *factual* Bank Marketing base we build the synthetic layer that the
bandits actually learn from. It is **physically separated** from the Kaggle data
(written under ``data/synthetic_enrichment/``) and fully reproducible from seeds.

Three artifacts are produced:

* ``offer_catalog``  — the arms: financial offers with eligibility rules, profit
  margins and a **documented latent reward model** (weights over context).
* ``offer_events``   — logged impressions under a logging policy, with the
  decision context, an immediate click signal and the (possibly delayed) reward.
* ``delayed_rewards``— conversions that materialise *after* a delay, so online
  learning must cope with rewards that are unknown at decision time.

The same latent model powers the offline simulator (Stage 3), which lets us
compute **regret against an oracle** that knows the true expected reward.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_offers.config import get_settings
from adaptive_offers.logging_utils import get_logger, log_event

logger = get_logger("data.synthetic")

# Context feature order shared by the latent model, the bandits and the API.
# Index 0 is a bias term; the rest are bounded/normalised signals derived from
# the processed (leakage-free) row. NO protected attribute is used directly.
CONTEXT_FEATURES: list[str] = [
    "bias",
    "prev_success",        # poutcome == success
    "cellular",            # contact channel is cellular
    "age_norm",            # (age - 40) / 15, clipped
    "previously_contacted",
    "low_rates",           # euribor3m below the run's median
    "young",               # age < 30
    "senior",              # age > 60
]

EVENTS_FILE = "offer_events.parquet"
CATALOG_FILE = "offer_catalog.parquet"
DELAYED_FILE = "delayed_rewards.parquet"
SCHEMA_FILE = "schema.json"

# Temporal horizon for delayed-reward maturation (business-day proxy).
REWARD_HORIZON_DAYS = 30


@dataclass(frozen=True)
class OfferArm:
    """A synthetic offer (bandit arm) with its latent reward parameters."""

    offer_id: str
    name: str
    category: str
    margin: float                      # profit if converted (synthetic R$)
    base_logit: float                  # baseline log-odds of conversion
    context_weights: dict[str, float]  # weight per CONTEXT_FEATURES[1:]
    requires_no_default: bool = False
    requires_no_loan: bool = False
    min_age: int = 0
    suitability_tier: str = "standard"  # standard | restricted

    def weight_vector(self) -> np.ndarray:
        """Dense weight vector aligned to CONTEXT_FEATURES (bias weight = 0)."""
        w = [0.0] + [self.context_weights.get(f, 0.0) for f in CONTEXT_FEATURES[1:]]
        return np.asarray(w, dtype=float)


def offer_catalog() -> list[OfferArm]:
    """Deterministic catalog of synthetic offers (arms).

    Offers deliberately have **different best segments** so a *contextual* policy
    (LinUCB) can outperform a context-free one. ``no_offer`` is the control arm.
    """
    return [
        OfferArm(
            offer_id="OFF_CC_CASHBACK", name="Cartão Cashback", category="card",
            margin=120.0, base_logit=-2.7,
            context_weights={"young": 0.9, "cellular": 0.5, "age_norm": -0.3},
            requires_no_default=True, suitability_tier="standard",
        ),
        OfferArm(
            offer_id="OFF_LOAN_PREAPP", name="Empréstimo Pré-aprovado", category="credit",
            margin=300.0, base_logit=-3.1,
            context_weights={"previously_contacted": 0.7, "low_rates": 0.6, "young": 0.3},
            requires_no_default=True, requires_no_loan=True, suitability_tier="restricted",
        ),
        OfferArm(
            offer_id="OFF_TD_PREMIUM", name="Depósito a Prazo Premium", category="deposit",
            margin=200.0, base_logit=-2.6,
            context_weights={"prev_success": 1.6, "senior": 0.8, "low_rates": -0.5},
            suitability_tier="standard",
        ),
        OfferArm(
            offer_id="OFF_FUND_INTRO", name="Fundo de Investimento Intro", category="invest",
            margin=180.0, base_logit=-2.9,
            context_weights={"senior": 0.7, "prev_success": 0.9, "age_norm": 0.4},
            min_age=25, suitability_tier="restricted",
        ),
        OfferArm(
            offer_id="OFF_INSURANCE", name="Seguro Bundle", category="insurance",
            margin=90.0, base_logit=-2.4,
            context_weights={"previously_contacted": 0.3, "cellular": 0.2},
            suitability_tier="standard",
        ),
        OfferArm(
            offer_id="OFF_NONE", name="Sem Oferta (controle)", category="control",
            margin=0.0, base_logit=-99.0, context_weights={},
            suitability_tier="standard",
        ),
    ]


# --------------------------------------------------------------------------- #
# Context construction and latent reward model
# --------------------------------------------------------------------------- #
def build_context_vector(row: pd.Series | dict, rate_median: float) -> np.ndarray:
    """Map a processed row to the numeric context vector (CONTEXT_FEATURES)."""
    get = row.get if isinstance(row, dict) else row.__getitem__
    age = float(get("age"))
    euribor = float(get("euribor3m"))
    return np.array([
        1.0,
        1.0 if str(get("poutcome")).lower() == "success" else 0.0,
        1.0 if str(get("contact")).lower() == "cellular" else 0.0,
        float(np.clip((age - 40.0) / 15.0, -2.0, 3.0)),
        float(get("previously_contacted")),
        1.0 if euribor < rate_median else 0.0,
        1.0 if age < 30 else 0.0,
        1.0 if age > 60 else 0.0,
    ], dtype=float)


def latent_conversion_prob(arm: OfferArm, context: np.ndarray) -> float:
    """True P(convert | arm, context) under the synthetic generative model."""
    logit = arm.base_logit + float(arm.weight_vector() @ context)
    return float(1.0 / (1.0 + np.exp(-logit)))


def expected_reward(arm: OfferArm, context: np.ndarray) -> float:
    """Margin-weighted expected reward = P(convert) * margin."""
    return latent_conversion_prob(arm, context) * arm.margin


def is_eligible(arm: OfferArm, row: pd.Series | dict) -> bool:
    """Eligibility / suitability gate applied before any arm can be offered."""
    get = row.get if isinstance(row, dict) else row.__getitem__
    if arm.requires_no_default and str(get("default")).lower() == "yes":
        return False
    if arm.requires_no_loan and str(get("loan")).lower() == "yes":
        return False
    if float(get("age")) < arm.min_age:
        return False
    return True


def eligible_arms(row: pd.Series | dict, arms: list[OfferArm]) -> list[OfferArm]:
    """Subset of arms a client is eligible for (``no_offer`` always included)."""
    return [a for a in arms if a.offer_id == "OFF_NONE" or is_eligible(a, row)]


# --------------------------------------------------------------------------- #
# Event generation
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticBundle:
    """In-memory view of the synthetic layer (used by the simulator)."""

    catalog: list[OfferArm]
    events: pd.DataFrame
    delayed: pd.DataFrame
    rate_median: float
    seed: int
    contexts: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))


def _logging_policy_probs(n_arms: int, epsilon: float = 0.85) -> np.ndarray:
    """Soft logging policy: mostly uniform exploration (realistic logged data)."""
    p = np.full(n_arms, epsilon / n_arms)
    p[0] += 1.0 - epsilon  # slight bias to the first eligible arm
    return p / p.sum()


def generate_events(
    processed: pd.DataFrame,
    arms: list[OfferArm],
    seed: int,
    delayed_fraction: float = 0.4,
    horizon_days: int = REWARD_HORIZON_DAYS,
) -> SyntheticBundle:
    """Generate logged impressions, immediate clicks and (delayed) rewards."""
    rng = np.random.default_rng(seed)
    rate_median = float(processed["euribor3m"].median())

    rows, delayed_rows, ctx_list = [], [], []
    base_day = np.datetime64("2025-01-01")

    for i, (_, row) in enumerate(processed.iterrows()):
        elig = eligible_arms(row, arms)
        probs = _logging_policy_probs(len(elig))
        idx = int(rng.choice(len(elig), p=probs))
        arm = elig[idx]
        propensity = float(probs[idx])

        ctx = build_context_vector(row, rate_median)
        ctx_list.append(ctx)

        p_conv = latent_conversion_prob(arm, ctx)
        converted = int(rng.random() < p_conv)
        # immediate click is a noisy precursor of conversion
        clicked = int(converted or (rng.random() < 0.12 + 0.4 * p_conv))
        reward = arm.margin if converted else 0.0

        is_delayed = bool(converted and rng.random() < delayed_fraction)
        delay_days = int(rng.integers(1, horizon_days)) if is_delayed else 0
        impression_day = base_day + np.timedelta64(int(rng.integers(0, 120)), "D")

        rows.append({
            "event_id": f"evt_{i:07d}",
            "client_event_id": row["client_event_id"],
            "impression_ts": str(impression_day),
            "offer_id": arm.offer_id,
            "propensity": round(propensity, 5),
            "n_eligible": len(elig),
            "clicked": clicked,
            "converted": converted,
            "reward": float(reward),
            "reward_observed_immediately": int(converted and not is_delayed),
            "reward_is_delayed": int(is_delayed),
            "reward_delay_days": delay_days,
        })
        if is_delayed:
            delayed_rows.append({
                "event_id": f"evt_{i:07d}",
                "client_event_id": row["client_event_id"],
                "offer_id": arm.offer_id,
                "conversion_ts": str(impression_day + np.timedelta64(delay_days, "D")),
                "reward_delay_days": delay_days,
                "realized_reward": float(reward),
            })

    events = pd.DataFrame(rows)
    delayed = pd.DataFrame(delayed_rows)
    return SyntheticBundle(
        catalog=arms, events=events, delayed=delayed,
        rate_median=rate_median, seed=seed, contexts=np.asarray(ctx_list),
    )


def _catalog_frame(arms: list[OfferArm]) -> pd.DataFrame:
    records = []
    for a in arms:
        d = asdict(a)
        d["context_weights"] = json.dumps(d["context_weights"], ensure_ascii=False)
        records.append(d)
    return pd.DataFrame(records)


def _write_schema(out_dir: Path, bundle: SyntheticBundle) -> None:
    schema = {
        "description": "Synthetic adaptive-experimentation layer over Bank Marketing.",
        "separated_from": "data/processed (Kaggle-derived) — physically distinct.",
        "seed": bundle.seed,
        "reward_horizon_days": REWARD_HORIZON_DAYS,
        "context_features": CONTEXT_FEATURES,
        "tables": {
            CATALOG_FILE: {
                "grain": "one row per offer (arm)",
                "columns": {
                    "offer_id": "string PK", "name": "string", "category": "string",
                    "margin": "float — profit if converted (synthetic R$)",
                    "base_logit": "float — baseline conversion log-odds",
                    "context_weights": "json — latent weights over context features",
                    "requires_no_default": "bool eligibility", "requires_no_loan": "bool eligibility",
                    "min_age": "int eligibility", "suitability_tier": "standard|restricted",
                },
            },
            EVENTS_FILE: {
                "grain": "one row per logged impression",
                "join_key": "client_event_id -> data/processed",
                "columns": {
                    "event_id": "string PK", "client_event_id": "FK to processed base",
                    "impression_ts": "ISO date of impression", "offer_id": "FK to catalog",
                    "propensity": "float — logging-policy prob of the shown arm",
                    "n_eligible": "int — eligible arms at decision time",
                    "clicked": "0/1 immediate signal", "converted": "0/1 outcome",
                    "reward": "float — margin if converted else 0",
                    "reward_observed_immediately": "0/1",
                    "reward_is_delayed": "0/1", "reward_delay_days": "int 0..horizon",
                },
            },
            DELAYED_FILE: {
                "grain": "one row per conversion that matured after a delay",
                "columns": {
                    "event_id": "FK to offer_events", "client_event_id": "FK to processed",
                    "offer_id": "FK to catalog", "conversion_ts": "ISO date reward matured",
                    "reward_delay_days": "int", "realized_reward": "float",
                },
            },
        },
    }
    (out_dir / SCHEMA_FILE).write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def generate(
    processed: pd.DataFrame | None = None,
    seed: int | None = None,
    out_dir: Path | None = None,
) -> SyntheticBundle:
    """Stage 2 orchestrator: build the synthetic layer and persist it + schema."""
    settings = get_settings()
    seed = settings.random_seed if seed is None else seed
    out_dir = out_dir or settings.paths.synthetic
    out_dir.mkdir(parents=True, exist_ok=True)

    if processed is None:
        from adaptive_offers.data.preprocessing import load_processed
        processed = load_processed()

    arms = offer_catalog()
    bundle = generate_events(processed, arms, seed=seed)

    _catalog_frame(arms).to_parquet(out_dir / CATALOG_FILE, index=False)
    bundle.events.to_parquet(out_dir / EVENTS_FILE, index=False)
    if not bundle.delayed.empty:
        bundle.delayed.to_parquet(out_dir / DELAYED_FILE, index=False)
    _write_schema(out_dir, bundle)

    log_event(
        logger, "synthetic_generated",
        seed=seed, n_arms=len(arms), n_events=int(len(bundle.events)),
        n_delayed=int(len(bundle.delayed)),
        conversion_rate=round(float(bundle.events["converted"].mean()), 4),
        delayed_share=round(
            float(bundle.events["reward_is_delayed"].mean()), 4
        ),
        out_dir=str(out_dir),
    )
    return bundle
