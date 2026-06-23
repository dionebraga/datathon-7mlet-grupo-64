"""Multi-channel orchestration + contact policy.

The challenge is explicit: a fintech decides **across different channels**. The
bandit picks *what* to offer; this module decides *where and whether* to deliver
it. It adds the two things channel optimisation in a regulated fintech actually
needs:

* a **channel catalog** (app push, SMS, e-mail, voice call) with real cost,
  latency, richness and sync/async properties; and
* a **contact policy** — frequency capping (don't over-contact), quiet hours
  (no intrusive contact at night), and channel selection by customer preference,
  message richness and cost.

The policy is applied as a guardrail *after* arm selection, emitting auditable
reason codes (``FREQUENCY_CAPPED`` / ``QUIET_HOURS`` / ``CHANNEL_SELECTED`` /
``CONTACT_SUPPRESSED``) — exactly like the eligibility/suitability gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Channel:
    """A delivery channel with the properties that drive selection."""

    channel_id: str
    label: str
    cost: float       # R$ per delivered message
    latency_s: float  # typical delivery latency
    rich: bool        # supports rich creative (images / long copy)
    sync: bool        # real-time/intrusive (voice) vs async (email/push)


CHANNELS: list[Channel] = [
    Channel("app_push", "App Push", 0.02, 1.0, rich=False, sync=False),
    Channel("email",    "E-mail",   0.01, 30.0, rich=True,  sync=False),
    Channel("sms",      "SMS",      0.08, 3.0,  rich=False, sync=False),
    Channel("call",     "Ligação",  2.50, 0.0,  rich=True,  sync=True),
]
_BY_ID = {c.channel_id: c for c in CHANNELS}


def channel_label(channel_id: str) -> str:
    c = _BY_ID.get(channel_id)
    return c.label if c else (channel_id or "—")


def _get(ctx: Any, key: str, default: Any) -> Any:
    try:
        val = ctx.get(key, default) if hasattr(ctx, "get") else ctx[key]
    except Exception:
        return default
    return default if val is None else val


@dataclass(frozen=True)
class ContactPolicy:
    """Contact governance: frequency cap, quiet hours and channel preference."""

    frequency_cap: int = 3   # max contacts inside the rolling window
    window_days: int = 7
    quiet_start: int = 21    # quiet hours start (inclusive), 24h clock
    quiet_end: int = 8       # quiet hours end (exclusive)
    prefer_low_cost: bool = True

    def in_quiet_hours(self, hour: int) -> bool:
        h = int(hour) % 24
        if self.quiet_start <= self.quiet_end:
            return self.quiet_start <= h < self.quiet_end
        return h >= self.quiet_start or h < self.quiet_end

    def select(self, context: Any, *, offer_rich: bool = False) -> tuple[Channel | None, list[str]]:
        """Pick a channel (or suppress) for this contact, with reason codes.

        ``context`` may carry ``recent_contacts`` (int) and ``hour`` (0–23); both
        default to safe values so the policy works on partial contexts.
        """
        codes: list[str] = []
        recent = int(_get(context, "recent_contacts", 0) or 0)
        hour = int(_get(context, "hour", 12) or 12)

        # 1) Frequency cap — never over-contact.
        if recent >= self.frequency_cap:
            return None, ["FREQUENCY_CAPPED", "CONTACT_SUPPRESSED"]

        # 2) Customer-preferred channel family (from the contact feature).
        contact = str(_get(context, "contact", "cellular")).lower()
        order = (["app_push", "sms", "email", "call"] if contact == "cellular"
                 else ["call", "email", "sms", "app_push"])
        candidates = [_BY_ID[c] for c in order]

        # 3) Quiet hours — drop intrusive channels (voice/SMS), keep async.
        if self.in_quiet_hours(hour):
            codes.append("QUIET_HOURS")
            async_only = [c for c in candidates if not c.sync and c.channel_id != "sms"]
            candidates = async_only or [c for c in candidates if not c.sync] or candidates

        # 4) Rank: rich offers prefer rich channels; otherwise minimise cost.
        if offer_rich:
            candidates.sort(key=lambda c: (not c.rich, c.cost))
        elif self.prefer_low_cost:
            candidates.sort(key=lambda c: c.cost)

        chosen = candidates[0] if candidates else None
        if chosen is not None:
            codes.append("CHANNEL_SELECTED")
        return chosen, codes


DEFAULT_CONTACT_POLICY = ContactPolicy()
