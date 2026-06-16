"""Reward & conversion health monitoring.

Tracks the realised reward/conversion stream against a reference baseline using a
simple control-chart rule (z-score on a rolling window). A sustained drop is a
signal to roll back the active policy or trigger human review (Stage 7 / System
Card guardrails such as reward-hacking detection).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RewardMonitor:
    """Online monitor over a rolling window of per-decision rewards."""

    reference_mean: float
    reference_std: float
    window: int = 500
    z_alert: float = 3.0
    _buf: deque[float] = field(default_factory=lambda: deque(maxlen=500))

    def __post_init__(self) -> None:
        self._buf = deque(maxlen=self.window)

    def observe(self, reward: float) -> None:
        self._buf.append(float(reward))

    def status(self) -> dict[str, Any]:
        if len(self._buf) < max(30, self.window // 10):
            return {"ready": False, "n": len(self._buf)}
        cur_mean = float(np.mean(self._buf))
        se = self.reference_std / np.sqrt(len(self._buf)) if self.reference_std else 1e-9
        z = (cur_mean - self.reference_mean) / se
        alert = z < -self.z_alert  # sustained drop
        return {
            "ready": True,
            "n": len(self._buf),
            "current_mean": round(cur_mean, 3),
            "reference_mean": round(self.reference_mean, 3),
            "z_score": round(float(z), 2),
            "alert": bool(alert),
            "action": "rollback/review" if alert else "ok",
        }


def reward_health(rewards: np.ndarray, reference_mean: float, reference_std: float,
                  window: int = 500) -> dict[str, Any]:
    """Convenience: replay a reward array through a monitor and return status."""
    mon = RewardMonitor(reference_mean=reference_mean, reference_std=reference_std, window=window)
    for r in rewards:
        mon.observe(r)
    return mon.status()
