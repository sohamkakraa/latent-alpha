"""Tests for persistent multi-fidelity search configuration."""

from experiments.successive_halving import (
    ARCHITECTURE_STAGES,
    HYPERPARAMETER_STAGES,
    _candidate_score,
    hyperparameter_candidates,
)


def test_stage_fidelity_increases_and_survivors_decrease():
    for stages in (ARCHITECTURE_STAGES, HYPERPARAMETER_STAGES):
        assert [stage.timesteps for stage in stages] == sorted(
            stage.timesteps for stage in stages
        )
        assert [stage.keep for stage in stages] == sorted(
            (stage.keep for stage in stages),
            reverse=True,
        )
        assert len(stages[-1].seeds) >= 3
        assert len(stages[-1].folds) >= 7


def test_hyperparameter_candidates_are_distinct():
    candidates = hyperparameter_candidates()
    assert len(candidates) >= 10
    serialized = {repr(candidate) for candidate in candidates.values()}
    assert len(serialized) == len(candidates)


def test_candidate_score_penalizes_instability():
    stable = [
        {"robust_score": 1.0},
        {"robust_score": 1.0},
        {"robust_score": 1.0},
    ]
    unstable = [
        {"robust_score": 0.0},
        {"robust_score": 1.0},
        {"robust_score": 2.0},
    ]
    assert _candidate_score(stable) > _candidate_score(unstable)
