import numpy as np
import pytest

from lzero.worker.muzero_evaluator import (
    balanced_episode_targets,
    evaluation_diversity_metrics,
    observation_checksum,
    update_action_checksum,
)


@pytest.mark.parametrize(
    'env_num,n_episode,expected',
    [
        (3, 3, [1, 1, 1]),
        (3, 5, [2, 2, 1]),
        (3, 8, [3, 3, 2]),
    ],
)
def test_balanced_episode_targets_match_vector_eval_monitor(env_num, n_episode, expected):
    targets = balanced_episode_targets(env_num, n_episode)

    np.testing.assert_array_equal(targets, expected)
    assert targets.sum() == n_episode


def test_balanced_episode_targets_reject_too_few_episodes():
    with pytest.raises(ValueError, match='at least env_num'):
        balanced_episode_targets(env_num=3, n_episode=2)


def test_evaluation_diversity_metrics_detect_duplicate_trajectories():
    signatures = [
        (11, 101, 500, 870.0),
        (22, 202, 420, 690.0),
        (11, 101, 500, 870.0),
    ]
    metrics = evaluation_diversity_metrics(signatures)
    assert metrics['eval/unique_trajectory_ratio'] == pytest.approx(2 / 3)
    assert metrics['eval/duplicate_trajectory_count'] == 1.0
    assert metrics['eval/unique_reward_length_ratio'] == pytest.approx(2 / 3)


def test_evaluation_checksums_are_deterministic_and_order_sensitive():
    observation = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    assert observation_checksum(observation) == observation_checksum(observation.copy())
    forward = update_action_checksum(update_action_checksum(0, 1), 2)
    reverse = update_action_checksum(update_action_checksum(0, 2), 1)
    assert forward != reverse
