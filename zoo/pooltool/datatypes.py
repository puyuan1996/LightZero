"""This would be a great place to create PoolToolEnv and observation/action vector building"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List

import attrs
import numpy as np
from ding.envs import BaseEnv
from easydict import EasyDict
from gym import spaces
from numpy.typing import NDArray

import pooltool as pt

ObservationDict = Dict[str, Any]


@attrs.define
class Spaces:
    observation: spaces.Space
    action: spaces.Space
    reward: spaces.Space


@attrs.define
class PoolToolGym(pt.State):
    spaces: Spaces

    def observation(self) -> ObservationDict:
        return dict(
            observation=self.observation_array(),
            action_mask=None,
            to_play=-1,
        )

    def scale_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        """Scale the action from [-1, 1] to the given range [low, high]"""
        low = self.spaces.action.low  # type: ignore
        high = self.spaces.action.high  # type: ignore
        assert np.all(action >= -1) and np.all(action <= 1), f"{action=}"
        scaled_action = low + (0.5 * (action + 1.0) * (high - low))
        return np.clip(scaled_action, low, high)

    def simulate(self) -> None:
        """Simulate the system"""
        pt.simulate(self.system, inplace=True, max_events=200)
        self.game.process_shot(self.system)
        self.game.advance(self.system)

    def seed(self, seed_value: int) -> None:
        random.seed(seed_value)
        np.random.seed(seed_value)

    def observation_array(self) -> Any:
        raise NotImplementedError("Inheriting classes must define this")

    def set_action(self, scaled_action: NDArray[np.float32]) -> None:
        raise NotImplementedError("Inheriting classes must define this")


class PoolToolEnv(BaseEnv):
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def manage_seeds(self) -> None:
        if (
            hasattr(self, "_seed")
            and hasattr(self, "_dynamic_seed")
            and self._dynamic_seed
        ):
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, "_seed"):
            self._env.seed(self._seed)

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + "Dict"
        return cfg

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop("collector_env_num")
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop("evaluator_env_num")
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(evaluator_env_num)]
