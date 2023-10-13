import os

import numpy as np
import pytest
from dizoo.minigrid.envs import MiniGridEnv


@pytest.mark.envtest
class TestMiniGridEnv:

    def test_naive(self):
        env = MiniGridEnv(MiniGridEnv.default_config())
        env.seed(314)
        path = './video'
        if not os.path.exists(path):
            os.mkdir(path)
        env.enable_save_replay(path)
        assert env._seed == 314
        obs = env.reset()
        print(f'env_id: {env._env_id}')
        print(f'max_step: {env._max_step}')
        for i in range(env._max_step):
            print(f'step: {i}')
            random_action = np.array(env.action_space.sample())
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (2835,)
            assert timestep.reward.shape == (1,)
            assert timestep.reward >= env.reward_space.low[0]
            assert timestep.reward <= env.reward_space.high[0]
            if timestep.done:
                break
                # env.reset()
        env.close()
