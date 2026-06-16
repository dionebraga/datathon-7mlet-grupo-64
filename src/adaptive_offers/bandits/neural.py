"""Neural contextual bandit (PyTorch) with MC-dropout exploration.

A small MLP predicts ``P(conversion | context, arm)`` from the context vector
concatenated with a one-hot arm encoding. Exploration follows the **dropout-as-
Bayesian-approximation** idea (Gal & Ghahramani, 2016; Riquelme et al., 2018):
a stochastic forward pass with dropout *on* yields a sampled estimate — a
Thompson-style draw — while a deterministic pass gives the greedy estimate.

The model trains online from a replay buffer (mini-batch SGD every ``train_every``
updates), so it copes with cold-start and delayed rewards like the other policies.
Ranking is margin-weighted (``estimate × margin``), keeping the same business
objective. Optional dependency: ``pip install "adaptive-offers[deep]"``.
"""

from __future__ import annotations

import numpy as np

from adaptive_offers.bandits.base import Arm, Decision, Policy

try:  # torch is an optional extra
    import torch
    from torch import nn

    _TORCH_OK = True
except Exception:  # pragma: no cover - exercised only when torch is absent
    _TORCH_OK = False


if _TORCH_OK:

    class _MLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class NeuralBandit(Policy):
    name = "neural"

    def __init__(
        self,
        arms: list[Arm],
        dim: int,
        seed: int = 42,
        hidden: int = 32,
        dropout: float = 0.1,
        lr: float = 0.01,
        train_every: int = 32,
        batch_size: int = 64,
        warmup: int = 64,
    ) -> None:
        if not _TORCH_OK:
            raise ImportError(
                "NeuralBandit requires PyTorch. Install it with "
                "`pip install \"adaptive-offers[deep]\"` (or `pip install torch`)."
            )
        super().__init__(arms, seed=seed)
        torch.manual_seed(seed)
        self.dim = dim
        self.n_arms = len(self.arm_ids)
        self.arm_index = {a: i for i, a in enumerate(self.arm_ids)}
        self.in_dim = dim + self.n_arms
        self.model = _MLP(self.in_dim, hidden, dropout)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.buffer: list[tuple[np.ndarray, float]] = []
        self.train_every = train_every
        self.batch_size = batch_size
        self.warmup = warmup
        self._since_train = 0

    def _features(self, context: np.ndarray, arm_id: str) -> np.ndarray:
        onehot = np.zeros(self.n_arms, dtype=np.float32)
        onehot[self.arm_index[arm_id]] = 1.0
        return np.concatenate([np.asarray(context, dtype=np.float32), onehot])

    def _probs(self, x: np.ndarray, stochastic: bool) -> np.ndarray:
        xt = torch.from_numpy(x)
        self.model.train(stochastic)  # dropout on iff stochastic (MC-dropout)
        with torch.no_grad():
            logits = self.model(xt).squeeze(-1).numpy()
        return 1.0 / (1.0 + np.exp(-logits))

    def select(self, context: np.ndarray, eligible: list[str]) -> Decision:
        elig = self._check_eligible(eligible)
        if np.asarray(context).shape[0] != self.dim:
            raise ValueError(f"context dim {np.asarray(context).shape[0]} != policy dim {self.dim}")
        self.t += 1
        x = np.vstack([self._features(context, a) for a in elig])
        sampled = self._probs(x, stochastic=True)   # Thompson-style draw (dropout)
        greedy_p = self._probs(x, stochastic=False)  # deterministic estimate
        scores = {a: float(sampled[i]) * self.margin(a) for i, a in enumerate(elig)}
        estimates = {a: float(greedy_p[i]) for i, a in enumerate(elig)}
        best = max(scores, key=scores.get)
        greedy = max(elig, key=lambda a: estimates[a] * self.margin(a))
        explored = best != greedy
        return Decision(
            arm_id=best, score=scores[best], explored=explored,
            reason_codes=["NEURAL", "CONTEXTUAL", "MC_DROPOUT",
                          "EXPLORATION" if explored else "EXPLOITATION"],
            scores=scores, estimates=estimates,
        )

    def update(self, arm_id: str, reward: float, context: np.ndarray | None = None) -> None:
        if context is None:
            raise ValueError("NeuralBandit.update requires the decision context")
        self.buffer.append((self._features(context, arm_id), float(np.clip(reward, 0.0, 1.0))))
        self._since_train += 1
        if len(self.buffer) >= self.warmup and self._since_train >= self.train_every:
            self._train_step()
            self._since_train = 0

    def _train_step(self) -> None:
        idx = self.rng.integers(0, len(self.buffer), min(self.batch_size, len(self.buffer)))
        X = torch.from_numpy(np.vstack([self.buffer[i][0] for i in idx]))
        y = torch.tensor([[self.buffer[i][1]] for i in idx], dtype=torch.float32)
        self.model.train()
        self.opt.zero_grad()
        loss = self.loss_fn(self.model(X), y)
        loss.backward()
        self.opt.step()

    def state_dict(self) -> dict:
        return {**super().state_dict(), "dim": self.dim, "in_dim": self.in_dim,
                "buffer_size": len(self.buffer)}
