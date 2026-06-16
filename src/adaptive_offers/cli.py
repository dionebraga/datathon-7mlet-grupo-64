"""``adaptive-offers`` CLI — single entry point for the whole pipeline.

    adaptive-offers data build           # Stage 1
    adaptive-offers synth generate       # Stage 2
    adaptive-offers train                # Stage 3 (+ MLflow tracking, registers policy)
    adaptive-offers evaluate             # Stage 4 (golden set + metrics + fairness)
    adaptive-offers pipeline             # Stages 1-4 end-to-end
    adaptive-offers decide --context ... # Stage 5 (single auditable decision)
    adaptive-offers serve                # Stage 5 (FastAPI service)
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from adaptive_offers.config import get_settings
from adaptive_offers.logging_utils import get_logger

logger = get_logger("cli")


@click.group()
@click.version_option(package_name="adaptive-offers")
def cli() -> None:
    """Adaptive Offers Platform — FIAP 7MLET Grupo 64."""


# --------------------------------------------------------------------------- #
# Stage 1 — data
# --------------------------------------------------------------------------- #
@cli.group()
def data() -> None:
    """Data layer commands (Stage 1)."""


@data.command("build")
@click.option("--rows", default=20_000, show_default=True, help="Rows for the facsimile.")
@click.option("--seed", default=None, type=int, help="Random seed.")
def data_build(rows: int, seed: int | None) -> None:
    """Build the processed, leakage-free base + provenance."""
    from adaptive_offers.data.preprocessing import build_processed

    df, prov, path = build_processed(n_rows=rows, seed=seed)
    click.echo(f"[ok] processed {df.shape} ({prov.source}) -> {path}")


# --------------------------------------------------------------------------- #
# Stage 2 — synth
# --------------------------------------------------------------------------- #
@cli.group()
def synth() -> None:
    """Synthetic enrichment commands (Stage 2)."""


@synth.command("generate")
@click.option("--seed", default=None, type=int, help="Random seed.")
def synth_generate(seed: int | None) -> None:
    """Generate offer catalog, events and delayed rewards."""
    from adaptive_offers.bootstrap import ensure_bundle

    bundle = ensure_bundle(seed=seed)
    click.echo(f"[ok] synthetic: {len(bundle.events)} events, "
               f"{len(bundle.delayed)} delayed, {len(bundle.catalog)} arms")


# --------------------------------------------------------------------------- #
# Stage 3 — train
# --------------------------------------------------------------------------- #
@cli.command("train")
@click.option("--policy", "policy_name", default="linucb",
              type=click.Choice(["baseline", "thompson", "nilos_ucb", "linucb"]))
@click.option("--version", default="v1", show_default=True)
@click.option("--horizon", default=12_000, show_default=True)
@click.option("--seed", default=None, type=int)
@click.option("--compare/--no-compare", default=True, help="Also compare all policies.")
def train(policy_name: str, version: str, horizon: int, seed: int | None, compare: bool) -> None:
    """Train a policy, register it as active and track in MLflow."""
    from adaptive_offers.bootstrap import ensure_bundle, ensure_data, train_and_register
    from adaptive_offers.evaluation.offline_eval import metrics_matrix
    from adaptive_offers.tracking import log_metrics, log_params, mlflow_run

    meta = train_and_register(policy_name=policy_name, version=version, horizon=horizon, seed=seed)
    click.echo(f"[ok] trained & registered {meta.name}@{meta.version}")
    click.echo(json.dumps(meta.metrics, indent=2, ensure_ascii=False))

    with mlflow_run(run_name=f"{policy_name}-{version}", tags={"stage": "3", "policy": policy_name}):
        log_params({"policy": policy_name, "version": version, "horizon": horizon})
        log_metrics(meta.metrics)

    if compare:
        proc = ensure_data(seed=seed)
        bundle = ensure_bundle(proc, seed=seed)
        rows = metrics_matrix(proc, bundle, horizon=horizon)
        click.echo("\n=== policy comparison ===")
        for r in rows:
            click.echo(f"  {r['policy']:10} reward={r['cumulative_reward']:>10} "
                       f"regret_ratio={r['regret_ratio']} lift={r.get('lift_vs_baseline_pct', 0)}%")


# --------------------------------------------------------------------------- #
# Stage 4 — evaluate
# --------------------------------------------------------------------------- #
@cli.command("evaluate")
@click.option("--horizon", default=12_000, show_default=True)
@click.option("--seed", default=None, type=int)
def evaluate(horizon: int, seed: int | None) -> None:
    """Run golden set + metrics matrix + fairness; write a JSON report."""
    from adaptive_offers.bootstrap import ensure_bundle, ensure_data
    from adaptive_offers.evaluation import (
        evaluate_golden,
        exposure_report,
        load_cases,
        metrics_matrix,
        train_frozen_policy,
    )

    settings = get_settings()
    proc = ensure_data(seed=seed)
    bundle = ensure_bundle(proc, seed=seed)
    cases = load_cases(settings.paths.golden_set / "evaluation_cases.jsonl")

    lin = train_frozen_policy("linucb", proc, bundle, horizon=horizon, seed=seed)
    golden = evaluate_golden(cases, lin, bundle.catalog, bundle.rate_median)
    matrix = metrics_matrix(proc, bundle, horizon=horizon)
    fairness = exposure_report(lin, proc.head(6000), bundle.catalog, bundle.rate_median)

    report = {
        "golden_set": {k: golden[k] for k in ("pass_rate", "n_cases", "by_category")},
        "metrics_matrix": matrix,
        "fairness": {"max_exposure_disparity": fairness["max_exposure_disparity"],
                     "flag": fairness["fairness_flag"]},
    }
    out = settings.paths.artifacts / "evaluation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"[ok] golden pass_rate={golden['pass_rate']} | "
               f"best={matrix[0]['policy']} | fairness={fairness['fairness_flag']}")
    click.echo(f"[ok] report -> {out}")


# --------------------------------------------------------------------------- #
# Stages 1-4 — pipeline
# --------------------------------------------------------------------------- #
@cli.command("pipeline")
@click.option("--seed", default=None, type=int)
@click.option("--horizon", default=12_000, show_default=True)
@click.pass_context
def pipeline(ctx: click.Context, seed: int | None, horizon: int) -> None:
    """Run data -> synth -> train -> evaluate end-to-end."""
    ctx.invoke(data_build, rows=20_000, seed=seed)
    ctx.invoke(synth_generate, seed=seed)
    ctx.invoke(train, policy_name="linucb", version="v1", horizon=horizon, seed=seed, compare=True)
    ctx.invoke(evaluate, horizon=horizon, seed=seed)
    click.echo("\n[ok] pipeline complete.")


# --------------------------------------------------------------------------- #
# Stage 5 — decide / serve
# --------------------------------------------------------------------------- #
@cli.command("decide")
@click.option("--context", "context_file", type=click.Path(exists=True, path_type=Path),
              help="JSON file with a decision context.")
@click.option("--client-event-id", default=None, help="Resolve features from the feature store.")
@click.option("--explain/--no-explain", default=False, help="Add an LLM/RAG explanation.")
def decide(context_file: Path | None, client_event_id: str | None, explain: bool) -> None:
    """Produce a single auditable decision."""
    from adaptive_offers.bootstrap import ensure_service

    svc = ensure_service(train_if_missing=True)
    if context_file:
        context = json.loads(context_file.read_text(encoding="utf-8"))
        record = svc.decide(context=context)
    elif client_event_id:
        record = svc.decide(client_event_id=client_event_id)
    else:
        raise click.UsageError("provide --context or --client-event-id")

    click.echo(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
    if explain:
        from adaptive_offers.assistant import Assistant

        result = Assistant().explain_decision(record.to_dict())
        click.echo("\n=== assistant ===")
        click.echo(result["answer"])


@cli.command("serve")
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
def serve(host: str | None, port: int | None) -> None:
    """Run the FastAPI decision service."""
    import uvicorn

    settings = get_settings()
    uvicorn.run("adaptive_offers.api.main:app",
                host=host or settings.api_host, port=port or settings.api_port)


if __name__ == "__main__":
    cli()
