"""BI dashboard for the Adaptive Offers Platform (Streamlit + Plotly).

Course: ML Hands-On — "BI com análise". A modern, business-facing view over the
bandit experiment: KPI cards, policy comparison, regret curves, offer mix, a
dataset glance and an interactive decision explorer with reason codes and an
LLM/RAG explanation.

Run (PowerShell):  streamlit run dashboard\app.py   ->  http://localhost:8501
"""

from __future__ import annotations

import socket
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
from adaptive_offers.data.quality import quality_report, target_rate_by  # noqa: E402
from adaptive_offers.data.synthetic import CONTEXT_FEATURES, generate  # noqa: E402
from adaptive_offers.simulation.environment import build_arms, run_simulation  # noqa: E402
from adaptive_offers.simulation.metrics import compare_results, regret_curve  # noqa: E402

# --------------------------------------------------------------------------- #
# Compatibility: avoid the `use_container_width` deprecation on Streamlit >=1.49
# while still working on older versions (project requires >=1.30).
# --------------------------------------------------------------------------- #
_ST_VER = tuple(int(p) for p in (st.__version__.split(".") + ["0", "0"])[:2])


def fill() -> dict:
    return {"width": "stretch"} if _ST_VER >= (1, 49) else {"use_container_width": True}


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


# --------------------------------------------------------------------------- #
# Page + theme
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Adaptive Offers · BI", page_icon="🎯", layout="wide")

ACCENT = "#6C5CE7"
GREEN = "#00B894"
POLICY_COLORS = {"linucb": "#6C5CE7", "thompson": "#00B894",
                 "nilos_ucb": "#0984E3", "baseline": "#B2BEC3"}
POLICY_LABEL = {"linucb": "LinUCB (contextual)", "thompson": "Thompson Sampling",
                "nilos_ucb": "Nilos-UCB (UCB-V)", "baseline": "Baseline (controle)"}

st.markdown(
    """
    <style>
      #MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px;}
      html, body, [class*="css"] {font-family: 'Inter','Segoe UI',sans-serif;}
      .hero {
        background: linear-gradient(115deg,#5A4BD6 0%,#6C5CE7 38%,#8E7BFF 70%,#00B894 140%);
        border-radius: 20px; padding: 28px 32px; color: white;
        box-shadow: 0 14px 38px rgba(108,92,231,.30); margin-bottom: 20px;
      }
      .hero h1 {margin: 0; font-size: 2rem; font-weight: 800; letter-spacing:-.01em;}
      .hero p {margin: 8px 0 0; opacity: .94; font-size: 1rem; max-width: 760px;}
      .hero .tags {margin-top: 14px;}
      .htag {display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25);
        padding:4px 12px; border-radius:999px; font-size:.78rem; font-weight:600; margin-right:6px;}
      .kpi {
        background: white; border: 1px solid #ECECF5; border-radius: 16px;
        padding: 18px 20px; box-shadow: 0 4px 20px rgba(22,21,39,.06); height: 100%;
        transition: transform .15s ease, box-shadow .15s ease;
      }
      .kpi:hover {transform: translateY(-3px); box-shadow: 0 10px 28px rgba(108,92,231,.16);}
      .kpi .ic {font-size: 1.2rem;}
      .kpi .label {font-size: .74rem; color: #8A89A6; text-transform: uppercase;
        letter-spacing: .05em; font-weight: 700; margin-top:4px;}
      .kpi .value {font-size: 1.75rem; font-weight: 800; color: #161527; line-height: 1.15;}
      .kpi .sub {font-size: .82rem; color: #00B894; font-weight: 600;}
      .pill {display:inline-block; padding:4px 11px; border-radius:999px;
        background:#F0EEFF; color:#6C5CE7; font-size:.76rem; font-weight:600; margin:2px 5px 2px 0;}
      .result {
        background: linear-gradient(135deg,#FFFFFF,#F4F2FF); border:1px solid #E6E3FB;
        border-radius:18px; padding:22px 26px; box-shadow:0 8px 26px rgba(108,92,231,.12);
      }
      .result .arm {font-size:1.6rem;font-weight:800;color:#6C5CE7;}
      .svc {font-size:.86rem; padding:6px 0;}
      .dot-on{color:#00B894;font-weight:700} .dot-off{color:#B2BEC3;font-weight:700}
      div[data-baseweb="tab-list"] {gap: 6px;}
      button[data-baseweb="tab"] {border-radius: 10px 10px 0 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(col, icon: str, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="ic">{icon}</div><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin={"l": 10, "r": 10, "t": 40, "b": 10},
        font={"family": "Inter, sans-serif", "color": "#161527"},
        title_font={"size": 16},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel={"font_size": 13},
    )
    fig.update_xaxes(gridcolor="#EFEFF6", zeroline=False)
    fig.update_yaxes(gridcolor="#EFEFF6", zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# Experiment (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Treinando políticas e simulando o experimento…")
def load_experiment(horizon: int, seed: int):
    proc_path = ROOT / "data" / "processed" / "bank_marketing_processed.parquet"
    # Ensure a full-size base: tests may leave a tiny (1.5k) layer that would make
    # the demo unrepresentative. Rebuild to 20k if missing or too small.
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
    st.markdown("## 🎯 Adaptive Offers")
    st.caption("FIAP 7MLET · Grupo 64 · Datathon")
    st.divider()
    st.markdown("##### ⚙️ Parâmetros")
    horizon = st.select_slider("Horizonte de simulação",
                               options=[2000, 4000, 6000, 8000, 12000, 20000], value=6000)
    seed = st.number_input("Seed", value=123, step=1)
    st.divider()
    st.markdown("##### 🔌 Serviços relacionados")

    def status_badge(up: bool) -> str:
        return ('<span class="dot-on">● online</span>' if up
                else '<span class="dot-off">● offline</span>')

    api_up, mlf_up = port_open(8000), port_open(5000)
    st.markdown(
        '<div class="svc">API REST · ' + status_badge(api_up)
        + '<br/><code>adaptive-offers serve</code> → '
        '<a href="http://localhost:8000/docs" target="_blank">:8000/docs</a></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="svc">MLflow · ' + status_badge(mlf_up)
        + '<br/><code>mlflow ui</code> → '
        '<a href="http://localhost:5000" target="_blank">:5000</a></div>',
        unsafe_allow_html=True)
    st.info("API e MLflow são **processos separados**. Abra **outro terminal** "
            "para iniciá-los — o dashboard funciona sozinho.", icon="ℹ️")

processed, bundle, results = load_experiment(int(horizon), int(seed))
summary = compare_results(list(results.values()))
sdf = pd.DataFrame(summary)
best = summary[0]
qrep = quality_report(processed)

# --------------------------------------------------------------------------- #
# Hero + KPIs
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="hero"><h1>🎯 Adaptive Offers Platform</h1>'
    '<p>Decisão de oferta em canais digitais com <b>multi-armed bandits</b> — '
    'equilibra exploração e explotação e otimiza recompensa ponderada por margem.</p>'
    '<div class="tags"><span class="htag">Thompson</span><span class="htag">Nilos-UCB</span>'
    '<span class="htag">LinUCB</span><span class="htag">Feature Store</span>'
    '<span class="htag">LLM + RAG</span><span class="htag">MLOps</span></div></div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "🏆", "Melhor política", POLICY_LABEL.get(best["policy"], best["policy"]), "por valor capturado")
kpi(c2, "💰", "Reward acumulado", f"R$ {best['cumulative_reward']:,.0f}", f"{best['reward_per_1k']:,.0f}/1k impressões")
kpi(c3, "🎯", "Regret ratio", f"{best['regret_ratio']:.1%}", "distância do ótimo")
kpi(c4, "📈", "Lift vs baseline", f"+{best.get('lift_vs_baseline_pct', 0):.1f}%", "valor adicional")
st.write("")

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
t1, t2, t3, t4, t5 = st.tabs(
    ["🏆 Comparação", "📉 Regret", "🎯 Mix de ofertas", "📊 Dados", "🧪 Explorador de decisão"]
)

with t1:
    left, right = st.columns([3, 2])
    order = sdf.sort_values("cumulative_reward")
    fig = go.Figure(go.Bar(
        x=order["cumulative_reward"], y=[POLICY_LABEL.get(p, p) for p in order["policy"]],
        orientation="h", marker_color=[POLICY_COLORS.get(p, ACCENT) for p in order["policy"]],
        text=[f"R$ {v:,.0f}" for v in order["cumulative_reward"]], textposition="outside",
        hovertemplate="%{y}<br>R$ %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(title="💰 Valor capturado por política", xaxis_title=None, yaxis_title=None)
    left.plotly_chart(style_fig(fig), **fill())
    show = sdf[["policy", "cumulative_reward", "regret_ratio", "conversion_rate",
                "exploration_rate", "lift_vs_baseline_pct"]].copy()
    show.columns = ["Política", "Reward", "Regret", "Conversão", "Exploração", "Lift %"]
    right.dataframe(show, hide_index=True, **fill())
    right.success("💡 O baseline tem **alta conversão mas baixo valor** — trava em ofertas de "
                  "baixa margem. O KPI correto é **valor = margem × conversão**.", icon="💡")

with t2:
    fig = go.Figure()
    for name, res in results.items():
        idx, cum = regret_curve(res, points=80)
        fig.add_trace(go.Scatter(
            x=idx, y=cum, mode="lines", name=POLICY_LABEL.get(name, name),
            line={"width": 3, "color": POLICY_COLORS.get(name, ACCENT)},
            hovertemplate="round %{x}<br>regret %{y:,.0f}<extra></extra>"))
    fig.update_layout(title="📉 Regret acumulado ao longo do tempo (menor é melhor)",
                      xaxis_title="Rounds", yaxis_title="Regret acumulado")
    st.plotly_chart(style_fig(fig, height=460), **fill())
    st.caption("A política contextual (LinUCB) mantém o menor regret — usa o contexto para "
               "rotear a oferta certa ao cliente certo.")

with t3:
    pick = st.selectbox("Política", list(results.keys()),
                        format_func=lambda p: POLICY_LABEL.get(p, p), index=3)
    pulls = pd.Series(results[pick].arm_pulls)
    pulls = pulls[pulls > 0].sort_values(ascending=False)
    ca, cb = st.columns([3, 2])
    fig = go.Figure(go.Pie(labels=pulls.index, values=pulls.values, hole=.58,
                           marker={"colors": px.colors.sequential.Purp[::-1]},
                           textinfo="percent+label"))
    fig.update_layout(title=f"🎯 Mix de ofertas — {POLICY_LABEL.get(pick, pick)}", showlegend=False)
    ca.plotly_chart(style_fig(fig, height=420), **fill())
    mix = pulls.rename("impressões").to_frame()
    mix["%"] = (mix["impressões"] / mix["impressões"].sum() * 100).round(1)
    cb.dataframe(mix, **fill())
    cb.caption("Personalização: políticas contextuais distribuem ofertas conforme o contexto.")

with t4:
    st.markdown("##### 📊 Base factual (Bank Marketing, sem vazamento)")
    d1, d2, d3, d4 = st.columns(4)
    kpi(d1, "🧾", "Registros", f"{qrep['n_rows']:,}", "linhas processadas")
    kpi(d2, "🎯", "Taxa de conversão", f"{qrep['target']['positive_rate']:.1%}", "alvo `subscribed`")
    kpi(d3, "⚖️", "Desbalanceamento", f"{qrep['target']['imbalance_ratio']:.0f}:1", "neg:pos")
    kpi(d4, "🧬", "Duplicatas", f"{qrep['n_duplicates']}", "qualidade")
    st.write("")
    g1, g2 = st.columns(2)
    by_p = target_rate_by(processed, "poutcome").reset_index()
    fig = go.Figure(go.Bar(x=by_p["poutcome"], y=by_p["subscription_rate"], marker_color=ACCENT,
                           text=[f"{v:.1%}" for v in by_p["subscription_rate"]], textposition="outside"))
    fig.update_layout(title="Conversão por resultado anterior (poutcome)")
    g1.plotly_chart(style_fig(fig), **fill())
    by_c = target_rate_by(processed, "contact").reset_index()
    fig = go.Figure(go.Bar(x=by_c["contact"], y=by_c["subscription_rate"], marker_color=GREEN,
                           text=[f"{v:.1%}" for v in by_c["subscription_rate"]], textposition="outside"))
    fig.update_layout(title="Conversão por canal de contato")
    g2.plotly_chart(style_fig(fig), **fill())
    st.caption("Esses sinais (sucesso prévio, canal, juros, idade) compõem o **contexto** do bandit.")

with t5:
    st.markdown("##### 🧪 Simule uma decisão e veja a explicação do assistente")
    col = st.columns(3)
    age = col[0].slider("Idade", 18, 95, 66)
    contact = col[0].selectbox("Canal", ["cellular", "telephone"])
    poutcome = col[1].selectbox("Resultado anterior", ["nonexistent", "failure", "success"], index=2)
    euribor = col[1].slider("Euribor 3m (juros)", 0.6, 5.1, 0.8)
    default = col[2].selectbox("Em default?", ["no", "yes", "unknown"])
    loan = col[2].selectbox("Tem empréstimo?", ["no", "yes", "unknown"])
    prev = 1 if poutcome != "nonexistent" else 0

    if st.button("🚀 Decidir oferta", type="primary", **fill()):
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
            f'<div style="color:#8A89A6;margin:4px 0 12px">valor esperado '
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
        with st.expander("📄 Citações de política (RAG)"):
            for c in exp["citations"]:
                st.write(f"- `{c['source']}` · relevância {c.get('score', 0)}")

st.divider()
st.caption("Grupo 64 · FIAP Pós-Tech 7MLET · github.com/dionebraga/datathon-7mlet-grupo-64")
