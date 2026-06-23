"""Next-Best-Action — message + next step are deterministic and persona-aware."""

from __future__ import annotations

from adaptive_offers.nba import next_best_action


def test_offer_maps_to_action_and_cta():
    nba = next_best_action("OFF_LOAN_PREAPP", "seg_massa", "email")
    assert nba.action_code == "SIMULATE_LOAN"
    assert nba.cta == "Ver valor pré-aprovado"
    assert nba.headline


def test_persona_changes_tone():
    """The same offer reads differently for different personas."""
    senior = next_best_action("OFF_TD_PREMIUM", "seg_senior_conserv", "email")
    jovem = next_best_action("OFF_TD_PREMIUM", "seg_jovem_digital", "app_push")
    assert senior.message != jovem.message
    assert senior.action_code == jovem.action_code  # same next step, different copy


def test_channel_hint_is_appended():
    push = next_best_action("OFF_CC_CASHBACK", "seg_jovem_digital", "app_push")
    email = next_best_action("OFF_CC_CASHBACK", "seg_jovem_digital", "email")
    assert "app" in push.message.lower()
    assert "e-mail" in email.message.lower()


def test_control_arm_has_no_action():
    nba = next_best_action("OFF_NONE")
    assert nba.action_code == "NO_ACTION"
    assert nba.cta == ""


def test_unknown_ids_fall_back_safely():
    nba = next_best_action("OFF_DOES_NOT_EXIST", "seg_unknown", "weird_channel")
    assert nba.action_code == "NO_ACTION"


def test_deterministic():
    a = next_best_action("OFF_FUND_INTRO", "seg_recorrente", "call")
    b = next_best_action("OFF_FUND_INTRO", "seg_recorrente", "call")
    assert a == b
