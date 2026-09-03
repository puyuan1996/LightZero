import pytest
import numpy as np
import gym

from zoo.atari.envs.atari_lightzero_env import AtariEnvLightZero, atari_episode_diagnostics
from zoo.atari.envs.atari_wrappers import RawRewardInfoWrapper
from easydict import EasyDict

config = EasyDict(dict(
    collector_env_num=8,
    evaluator_env_num=3,
    n_evaluator_episode=3,
    env_id='PongNoFrameskip-v4',
    env_type='Atari',
    obs_shape=(4, 96, 96),
    collect_max_episode_steps=int(1.08e5),
    eval_max_episode_steps=int(1.08e5),
    gray_scale=True,
    frame_skip=4,
    episode_life=True,
    clip_rewards=True,
    channel_last=False,
    render_mode_human=False,
    scale=True,
    warp_frame=True,
    save_video=False,
    transform2string=False,
    game_wrapper=True,
    manager=dict(shared_memory=False, ),
    stop_value=int(1e6),
))

config.max_episode_steps = config.eval_max_episode_steps


def test_atari_episode_diagnostics_reports_termination_and_configured_metadata():
    cfg = EasyDict(episode_diagnostic_info_keys=['lives', 'room', 'ignored'])
    metrics = atari_episode_diagnostics(
        {'TimeLimit.truncated': True, 'lives': 2, 'room': 7, 'ignored': 'text'},
        cfg,
        episode_length=123,
    )
    assert metrics == {
        'episode/terminated_by_time_limit': 1.0,
        'episode/natural_termination': 0.0,
        'episode/final_length': 123.0,
        'atari/lives': 2.0,
        'atari/room': 7.0,
    }


def test_raw_reward_info_wrapper_preserves_reward_and_records_raw_value():
    class DummyEnv(gym.Env):
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def step(self, action):
            return np.zeros(1), 7.5, False, {'existing': 1}

        def reset(self, **kwargs):
            return np.zeros(1)

    wrapped = RawRewardInfoWrapper(DummyEnv())
    observation, reward, done, info = wrapped.step(0)

    assert reward == 7.5
    assert done is False
    assert info['existing'] == 1
    assert np.asarray(info['raw_reward']).item() == 7.5

@pytest.mark.envtest
class TestAtariEnvLightZero:
    def test_naive(self):
        env = AtariEnvLightZero(config)
        env.reset()
        while True:
            action = env.random_action()
            obs, reward, done, info = env.step(action)
            if done:
                print(info)
                break
