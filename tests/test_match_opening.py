from __future__ import annotations

from rescue_vision.app import MatchPreflight, MatchState
from rescue_vision.config import (
    MatchOpeningStraight,
    MatchOpeningTurn,
    load_runtime_config,
)
from noncontact_support import MotionPlant
from test_match import make_sequence, runtime_config


def test_formal_match_config_has_its_own_opening_actions() -> None:
    config = load_runtime_config("configs/runtime.match.yaml")

    assert config.match.opening_actions == (
        MatchOpeningTurn(-0.75, 3.0, 0.2),
        MatchOpeningStraight(2.5, 1.5, 0.1),
    )
    assert config.match.opening_actions != config.match.nb_opening_actions


def test_formal_opening_runs_its_configured_actions_in_order() -> None:
    sequence = make_sequence(
        config=runtime_config(
            opening_actions=(
                MatchOpeningTurn(-0.20, 0.50, 0.0),
                MatchOpeningStraight(0.10, 0.50, 0.0),
            ),
        )
    )
    assert sequence.preflight(
        0,
        MatchPreflight(True, True, True, True, True, True),
    ).state is MatchState.PREFLIGHT
    sequence.start(1)
    plant = MotionPlant(sequence, heading=-1.5707963267948966, dt=0.01)
    plant.until(lambda decision: decision.state is MatchState.SEARCH_CLUSTER, seconds=8)

    reasons = [decision.reason for decision in plant.records]
    assert any(reason.startswith("noncontact=opening_action_1") for reason in reasons)
    assert any(reason.startswith("noncontact=opening_action_2") for reason in reasons)
    assert "opening_sequence_complete" in reasons
    assert reasons.index("opening_sequence_complete") > next(
        index
        for index, reason in enumerate(reasons)
        if reason.startswith("noncontact=opening_action_2")
    )
