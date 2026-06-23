"""Contact policy — channel selection is deterministic, governed and auditable."""

from __future__ import annotations

from adaptive_offers.channels import (
    CHANNELS,
    DEFAULT_CONTACT_POLICY,
    ContactPolicy,
    channel_label,
)


def test_frequency_cap_suppresses_contact():
    """At or above the cap the policy suppresses contact (no channel)."""
    pol = ContactPolicy(frequency_cap=3)
    chan, codes = pol.select({"recent_contacts": 3})
    assert chan is None
    assert "FREQUENCY_CAPPED" in codes and "CONTACT_SUPPRESSED" in codes


def test_quiet_hours_drops_intrusive_channels():
    """At night, voice (sync) and SMS are dropped; an async channel is chosen."""
    pol = ContactPolicy()
    chan, codes = pol.select({"recent_contacts": 0, "hour": 23, "contact": "telephone"})
    assert "QUIET_HOURS" in codes
    assert chan is not None and chan.channel_id not in {"call", "sms"}


def test_low_cost_preference_for_plain_offer():
    """A plain offer minimises cost → cheapest channel wins (e-mail)."""
    pol = ContactPolicy()
    chan, codes = pol.select({"recent_contacts": 0, "hour": 12}, offer_rich=False)
    assert "CHANNEL_SELECTED" in codes
    assert chan is not None
    assert chan.cost == min(c.cost for c in CHANNELS)


def test_rich_offer_prefers_rich_channel():
    """A rich offer prefers a rich-creative channel among the candidates."""
    pol = ContactPolicy()
    chan, _ = pol.select({"recent_contacts": 0, "hour": 12}, offer_rich=True)
    assert chan is not None and chan.rich is True


def test_default_policy_and_labels():
    assert DEFAULT_CONTACT_POLICY.frequency_cap == 3
    assert channel_label("app_push") == "App Push"
    assert channel_label("nope") == "nope"


def test_in_quiet_hours_wraps_midnight():
    pol = ContactPolicy(quiet_start=21, quiet_end=8)
    assert pol.in_quiet_hours(22) is True
    assert pol.in_quiet_hours(2) is True
    assert pol.in_quiet_hours(12) is False
