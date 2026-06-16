"""BI dashboard for the Adaptive Offers Platform (Streamlit + Plotly).

Course: ML Hands-On — "BI com análise". A modern, business-facing view over the
bandit experiment: KPI cards, policy comparison, regret curves, offer mix and an
interactive decision explorer with reason codes and an LLM/RAG explanation.

Run (PowerShell):  streamlit run dashboard\app.py   ->  http://localhost:8501
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_offers.bandits.registry import build_policy  # noqa: E402
from adaptive_offers.data.preprocessing import build_processed, load_processed  # noqa: E402
from adaptive_offers.data.synthetic import CONTEXT_FEATURES, generate  # noqa: E402
from adaptive_offers.simulation.environment import build_arms, run_simulation  # noqa: E402
from adaptive_offers.simulation.metrics import compare_results, regret_curve  # noqa: E402

# --------------------------------------------------------------------------- #
# Page + theme
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Adaptive Offers · BI", page_icon="🎯", layout="wide")

ACCENT = "#6C5CE7"
POLICY_COLORS = {
    "linucb": "#6C5CE7", "thompson": "#00B894",
    "nilos_ucb": "#0984E3", "baseline": "#B2BEC3",
}
POLICY_LABEL = {
    "linucb": "LinUCB (contextual)", "thompson": "Thompson Sampling",
    "nilos_ucb": "Nilos-UCB (UCB-V)", "baseline": "Baseline (controle)",
}

st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1250px;}
      .hero {
        background: linear-gradient(110deg, #6C5CE7 0%, #8E7BFF 45%, #00B894 130%);
        border-radius: 18px; padding: 26px 30px; color: white;
        box-shadow: 0 10px 30px rgba(108,92,231,.28); margin-bottom: 18px;
      }
      .hero h1 {margin: 0; font-size: 1.9rem; font-weight: 800;}
      .hero p {margin: 6px 0 0; opacity: .92; font-size: .98rem;}
      .kpi {
        background: white; border: 1px solid #ECECF5; border-radius: 16px;
        padding: 16px 18px; box-shadow: 0 4px 18px rgba(22,21,39,.05); height: 100%;
      }
      .kpi .label {font-size: .78rem; color: #8A89A6; text-transform: uppercase;
        letter-spacing: .04em; font-weight: 600;}
      .kpi .value {font-size: 1.7rem; font-weight: 800; color: #161527; line-height: 1.1;}
      .kpi .sub {font-size: .82rem; color: #00B894; font-weight: 600;}
      .pill {display:inline-block; padding:3px 10px; border-radius:999px;
        background:#F0EEFF; color:#6C5CE7; font-size:.78rem; font-weight:600; margin:2px 4px 2px 0;}
      .result {
        background: linear-gradient(135deg,#FFFFFF,#F6F5FF); border:1px solid #E6E3FB;
        border-radius:16px; padding:18px 22px; box-shadow:0 6px 22px rgba(108,92,231,.10);
      }
      .result .arm {font-size:1.4rem;font-weight:800;color:#6C5CE7;}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="sans-serif", color="#161527"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --------------------------------------------------------------------------- #
# Experiment (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Treinando políticas e simulando o experimento…")
def load_experiment(horizon: int, seed: int):
    proc_path = ROOT / "data" / "processed" / "bank_marketing_processed.parquet"
    # Ensure a full-size base: tests may have left a tiny (1.5k) processed layer,
    # which would make the demo unrepresentative. Rebuild to 20k if needed.
    needs_build = not proc_path.exists()
    if not needs_build:
        try:
            needs_build = len(pd.read_parquet(proc_path, columns=["client_event_id"])) < 20_000
        except Exception:
            needs_build = True
    if needs_build:
        build_processed(n_rows=20_000, seed=42)
    processed = load_processed()
    bundle = generate(processed=processed, seed=42)
    arms = build_arms(bundle.catalog)
    results = {}
    for name in ("baseline", "thompson", "nilos_ucb", "linucb"):
        pol = build_policy(name, arms, context_dim=len(CONTEXT_FEATURES), seed=seed)
        results[name] = run_simulation(pol, processed, bundle, horizon=horizon, seed=seed)
    return processed, bundle, results


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f"### 🎯 Adaptive Offers")
    st.caption("FIAP 7MLET · Grupo 64")
    horizon = st.select_slider("Horizonte de simulação",
                               options=[2000, 4000, 6000, 8000, 12000, 20000], value=6000)
    seed = st.number_input("Seed", value=123, step=1)
    st.divider()
    st.caption("Multi-armed bandit para decisão de ofertas financeiras. "
               "Dados sintéticos sobre a base Bank Marketing (Kaggle).")

processed, bundle, results = load_experiment(int(horizon), int(seed))
summary = compare_results(list(results.values()))
sdf = pd.DataFrame(summary)
best = summary[0]

# --------------------------------------------------------------------------- #
# Hero + KPIs
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="hero"><h1>🎯 Adaptive Offers Platform</h1>'
    '<p>Decisão de oferta em canais digitais com multi-armed bandits · '
    'exploração vs explotação · recompensa ponderada por margem</p></div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Melhor política", POLICY_LABEL.get(best["policy"], best["policy"]), "por valor capturado")
kpi(c2, "Reward acumulado", f"R$ {best['cumulative_reward']:,.0f}", f"{best['reward_per_1k']:,.0f}/1k impr.")
kpi(c3, "Regret ratio", f"{best['regret_ratio']:.1%}", "distância do ótimo")
kpi(c4, "Lift vs baseline", f"+{best.get('lift_vs_baseline_pct', 0):.1f}%", "valor adicional")

st.write("")

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
t1, t2, t3, t4 = st.tabs(["🏆 Comparação", "📉 Regret", "🎯 Mix de ofertas", "🧪 Explorador de decisão"])

with t1:
    left, right = st.columns([3, 2])
    order = sdf.sort_values("cumulative_reward")
    fig = go.Figure(go.Bar(
        x=order["cumulative_reward"], y=[POLICY_LABEL.get(p, p) for p in order["policy"]],
        orientation="h", marker_color=[POLICY_COLORS.get(p, ACCENT) for p in order["policy"]],
        text=[f"R$ {v:,.0f}" for v in order["cumulative_reward"]], textposition="outside",
    ))
    fig.update_layout(title="Valor capturado por política")
    left.plotly_chart(style_fig(fig), use_container_width=True)

    radar_cols = ["conversion_rate", "exploration_rate", "regret_ratio"]
    show = sdf[["policy", "cumulative_reward", "regret_ratio", "conversion_rate",
                "exploration_rate", "lift_vs_baseline_pct"]].copy()
    show.columns = ["Política", "Reward", "Regret", "Conversão", "Exploração", "Lift %"]
    right.dataframe(show, hide_index=True, use_container_width=True)
    right.info("💡 O baseline tem **alta conversão mas baixo valor** — trava em ofertas de "
               "baixa margem. O KPI correto é **valor (margem×conversão)**.")

with t2:
    fig = go.Figure()
    for name, res in results.items():
        idx, cum = regret_curve(res, points=80)
        fig.add_trace(go.Scatter(x=idx, y=cum, mode="lines", name=POLICY_LABEL.get(name, name),
                                 line=dict(width=3, color=POLICY_COLORS.get(name, ACCENT))))
    fig.update_layout(title="Regret acumulado ao longo do tempo (menor é melhor)",
                      xaxis_title="Rounds", yaxis_title="Regret acumulado")
    st.plotly_chart(style_fig(fig, height=440), use_container_width=True)
    st.caption("A política contextual (LinUCB) mantém o menor regret — usa o contexto "
               "para rotear a oferta certa ao cliente certo.")

with t3:
    pick = st.selectbox("Política", list(results.keys()),
                        format_func=lambda p: POLICY_LABEL.get(p, p), index=3)
    pulls = pd.Series(results[pick].arm_pulls)
    pulls = pulls[pulls > 0].sort_values(ascending=False)
    fig = go.Figure(go.Pie(labels=pulls.index, values=pulls.values, hole=.55,
                           marker=dict(colors=px.colors.sequential.Purp[::-1])))
    fig.update_layout(title=f"Mix de ofertas — {POLICY_LABEL.get(pick, pick)}")
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    st.caption("Personalização: políticas contextuais distribuem ofertas conforme o contexto.")

with t4:
    st.markdown("##### Simule uma decisão e veja a explicação do assistente")
    col = st.columns(3)
    age = col[0].slider("Idade", 18, 95, 66)
    contact = col[0].selectbox("Canal", ["cellular", "telephone"])
    poutcome = col[1].selectbox("Resultado anterior", ["nonexistent", "failure", "success"], index=2)
    euribor = col[1].slider("Euribor 3m (juros)", 0.6, 5.1, 0.8)
    default = col[2].selectbox("Em default?", ["no", "yes", "unknown"])
    loan = col[2].selectbox("Tem empréstimo?", ["no", "yes", "unknown"])
    prev = 1 if poutcome != "nonexistent" else 0

    if st.button("🚀 Decidir oferta", type="primary", use_container_width=True):
        with st.spinner("Decidindo…"):
            from adaptive_offers.assistant import Assistant
            from adaptive_offers.bootstrap import ensure_service

            ctx = {"age": age, "contact": contact, "poutcome": poutcome, "euribor3m": euribor,
                   "default": default, "loan": loan, "previously_contacted": prev}
            svc = ensure_service(train_if_missing=True)
            rec = svc.decide(context=ctx, log=False)
            exp = Assistant().explain_decision(rec.to_dict())

        pills = " ".join(f'<span class="pill">{c}</span>' for c in rec.reason_codes)
        st.markdown(
            f'<div class="result"><div class="arm">🎁 {rec.arm_name}</div>'
            f'<div style="color:#8A89A6;margin:2px 0 10px">valor esperado '
            f'<b>R$ {rec.expected_reward:.1f}</b> · '
            f'{"🔍 exploração" if rec.explored else "🎯 explotação"} · '
            f'política <b>{rec.policy_name}@{rec.policy_version}</b></div>{pills}</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        cc = st.columns(2)
        cc[0].markdown("**Ofertas elegíveis**")
        cc[0].write(", ".join(rec.eligible_arms))
        cc[1].markdown("**🤖 Assistente (RAG)**")
        cc[1].write(exp["answer"])
        with st.expander("Citações de política (RAG)"):
            for c in exp["citations"]:
                st.write(f"- `{c['source']}` · relevância {c.get('score', 0)}")

st.divider()
st.caption("Grupo 64 · FIAP Pós-Tech 7MLET · github.com/dionebraga/datathon-7mlet-grupo-64")
