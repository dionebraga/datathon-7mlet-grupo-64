"""BI dashboard for the Adaptive Offers Platform (Streamlit).

Course: ML Hands-On — "BI com análise". Provides a business view over the bandit
experiment: policy comparison, regret curves, offer mix, golden-set quality and
an interactive decision explorer with reason codes and an LLM/RAG explanation.

Run: ``streamlit run dashboard/app.py``  (or ``make dashboard``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_offers.bandits.registry import build_policy  # noqa: E402
from adaptive_offers.data.preprocessing import build_processed, load_processed  # noqa: E402
from adaptive_offers.data.synthetic import CONTEXT_FEATURES, generate  # noqa: E402
from adaptive_offers.simulation.environment import build_arms, run_simulation  # noqa: E402
from adaptive_offers.simulation.metrics import compare_results, regret_curve  # noqa: E402

st.set_page_config(page_title="Adaptive Offers — BI", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Carregando dados e simulando políticas...")
def load_experiment(horizon: int = 8000, seed: int = 123):
    proc_path = ROOT / "data" / "processed" / "bank_marketing_processed.parquet"
    if not proc_path.exists():
        build_processed(n_rows=20_000, seed=42)
    processed = load_processed()
    bundle = generate(processed=processed, seed=42)
    arms = build_arms(bundle.catalog)
    results = {}
    for name in ("baseline", "thompson", "nilos_ucb", "linucb"):
        pol = build_policy(name, arms, context_dim=len(CONTEXT_FEATURES), seed=seed)
        results[name] = run_simulation(pol, processed, bundle, horizon=horizon, seed=seed)
    return processed, bundle, results


st.title("📊 Adaptive Offers Platform — BI & Análise")
st.caption("FIAP 7MLET · Grupo 64 · Multi-armed bandit para ofertas financeiras")

horizon = st.sidebar.slider("Horizonte de simulação", 2000, 20000, 8000, step=2000)
seed = st.sidebar.number_input("Seed", value=123, step=1)
processed, bundle, results = load_experiment(horizon=horizon, seed=int(seed))
summary = compare_results(list(results.values()))
sdf = pd.DataFrame(summary)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏆 Comparação", "📉 Regret", "🎯 Mix de ofertas", "🧪 Explorador de decisão"]
)

with tab1:
    best = summary[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Melhor política", best["policy"])
    c2.metric("Reward acumulado", f"{best['cumulative_reward']:,.0f}")
    c3.metric("Regret ratio", f"{best['regret_ratio']:.1%}")
    c4.metric("Lift vs baseline", f"{best.get('lift_vs_baseline_pct', 0):.1f}%")
    st.subheader("Valor capturado por política")
    st.bar_chart(sdf.set_index("policy")["cumulative_reward"])
    st.dataframe(
        sdf[["policy", "cumulative_reward", "regret_ratio", "conversion_rate",
             "exploration_rate", "lift_vs_baseline_pct"]],
        use_container_width=True, hide_index=True,
    )
    st.info("Observação: o baseline tem alta conversão mas baixo valor — trava em "
            "ofertas de baixa margem. O KPI correto é valor (margem×conversão).")

with tab2:
    st.subheader("Curva de regret acumulado (menor é melhor)")
    curves = {}
    for name, res in results.items():
        idx, cum = regret_curve(res, points=60)
        curves[name] = pd.Series(cum, index=idx)
    st.line_chart(pd.DataFrame(curves))
    st.caption("LinUCB (contextual) mantém o menor regret ao longo do tempo.")

with tab3:
    st.subheader("Mix de ofertas escolhidas")
    policy_pick = st.selectbox("Política", list(results.keys()), index=3)
    pulls = pd.Series(results[policy_pick].arm_pulls).sort_values(ascending=False)
    st.bar_chart(pulls)
    st.caption("Personalização: políticas contextuais distribuem ofertas conforme o contexto.")

with tab4:
    st.subheader("Explorador de decisão (com reason codes + assistente)")
    col = st.columns(3)
    age = col[0].slider("Idade", 18, 95, 66)
    contact = col[0].selectbox("Canal", ["cellular", "telephone"])
    poutcome = col[1].selectbox("Resultado anterior", ["nonexistent", "failure", "success"])
    euribor = col[1].slider("Euribor 3m", 0.6, 5.1, 0.8)
    default = col[2].selectbox("Em default?", ["no", "yes", "unknown"])
    loan = col[2].selectbox("Tem empréstimo?", ["no", "yes", "unknown"])
    prev = 1 if poutcome != "nonexistent" else 0

    if st.button("Decidir", type="primary"):
        from adaptive_offers.assistant import Assistant
        from adaptive_offers.bootstrap import ensure_service

        ctx = {"age": age, "contact": contact, "poutcome": poutcome, "euribor3m": euribor,
               "default": default, "loan": loan, "previously_contacted": prev}
        svc = ensure_service(train_if_missing=True)
        rec = svc.decide(context=ctx, log=False)
        st.success(f"Oferta: **{rec.arm_name}** ({rec.arm_id}) · "
                   f"valor esperado {rec.expected_reward:.1f} · "
                   f"{'exploração' if rec.explored else 'explotação'}")
        st.write("**Reason codes:**", ", ".join(rec.reason_codes))
        st.write("**Elegíveis:**", ", ".join(rec.eligible_arms))
        exp = Assistant().explain_decision(rec.to_dict())
        st.markdown("**Assistente (RAG):**")
        st.write(exp["answer"])
        with st.expander("Citações de política (RAG)"):
            for c in exp["citations"]:
                st.write(f"- `{c['source']}` (score {c.get('score', 0)})")
