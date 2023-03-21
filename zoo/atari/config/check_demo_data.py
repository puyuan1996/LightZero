from typing import Callable
from collections import namedtuple
import numpy as np
import torch.distributed as dist

from ding.envs import BaseEnvManager
from ding.torch_utils import to_tensor, to_ndarray
from ding.utils import build_logger, EasyTimer
from ding.utils import get_world_size, get_rank
from ding.worker.collector.base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor
import pickle


from typing import Union, Optional, Tuple
import os
import torch
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.utils.data import DataLoader

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed


class InteractionSerialEvaluator(ISerialEvaluator):

    config = dict(
        # Evaluate every "eval_freq" training iterations.
        eval_freq=1000,
        render=dict(
            # tensorboard video render is disabled by default
            render_freq=-1,
            mode='train_iter',
        )
    )

    def __init__(
            self,
            cfg: dict,
            real_dataset,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'evaluator',
    ) -> None:
        self._cfg = cfg
        self._exp_name = exp_name
        self._instance_name = instance_name
        self.real_dataset = real_dataset

        # Logger (Monitor will be initialized in policy setter)
        # Only rank == 0 learner needs monitor and tb_logger, others only need text_logger to display terminal output.
        if get_rank() == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name
                )
        else:
            self._logger, _ = build_logger(
                './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False
            )
            self._tb_logger = None
        self.reset(policy, env)

        self._timer = EasyTimer()
        self._default_n_episode = cfg.n_episode
        self._stop_value = cfg.stop_value
        # only one freq
        self._render = cfg.render
        assert self._render.mode in ('envstep', 'train_iter'), 'mode should be envstep or train_iter'

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)
        self._max_eval_reward = float("-inf")
        self._last_eval_iter = -1
        self._end_flag = False
        self._last_render_iter = -1

    def close(self) -> None:
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self):
        self.close()

    def should_eval(self, train_iter: int) -> bool:
        if train_iter == self._last_eval_iter:
            return False
        if (train_iter - self._last_eval_iter) < self._cfg.eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def _should_render(self, envstep, train_iter):
        if self._render.render_freq == -1:
            return False
        iter = envstep if self._render.mode == 'envstep' else train_iter
        if (iter - self._last_render_iter) < self._render.render_freq:
            return False
        self._last_render_iter = iter
        return True

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            force_render: bool = False,
    ) -> Tuple[bool, dict]:
        if get_world_size() > 1:
            # sum up envstep to rank0
            envstep_tensor = torch.tensor(envstep).cuda()
            dist.reduce(envstep_tensor, dst=0)
            envstep = envstep_tensor.item()

        # evaluator only work on rank0
        stop_flag, return_info = False, []
        if get_rank() == 0:
            if n_episode is None:
                n_episode = self._default_n_episode
            assert n_episode is not None, "please indicate eval n_episode"
            envstep_count = 0
            info = {}
            eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
            self._env.reset()
            self._policy.reset()
            cnt = 0

            # force_render overwrite frequency constraint
            render = force_render or self._should_render(envstep, train_iter)

            with self._timer:
                while not eval_monitor.is_finished():
                    obs = self._env.ready_obs
                    obs = to_tensor(obs, dtype=torch.float32)
                    try:
                        print(f"cnt: {cnt}")
                        print('shape:', obs[0]['observation'].shape, self.real_dataset[cnt]['obs'].shape)
                        print("Obs distance: " + str(torch.sum(obs[0]['observation'] != self.real_dataset[cnt]['obs']).item()))
                    except Exception as error:
                        print(error, "no action")
                        pass
                    if cnt == 444:
                        print('here')
                    # update videos
                    if render:
                        eval_monitor.update_video(self._env.ready_imgs)
                    try:
                        actions = to_ndarray({0: self.real_dataset[cnt]['action']})
                    except:
                        actions = to_ndarray({0: 0})

                    timesteps = self._env.step(actions)
                    cnt += 1
                    timesteps = to_tensor(timesteps, dtype=torch.float32)
                    for env_id, t in timesteps.items():
                        if t.info.get('abnormal', False):
                            # If there is an abnormal timestep, reset all the related variables(including this env).
                            self._policy.reset([env_id])
                            continue
                        if t.done:
                            # Env reset is done by env_manager automatically.
                            self._policy.reset([env_id])
                            reward = t.info['final_eval_reward']
                            # print(reward)
                            if 'episode_info' in t.info:
                                eval_monitor.update_info(env_id, t.info['episode_info'])
                            eval_monitor.update_reward(env_id, reward)
                            return_info.append(t.info)
                            self._logger.info(
                                "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                    env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                                )
                            )
                        envstep_count += 1
            duration = self._timer.value
            episode_reward = eval_monitor.get_episode_reward()
            info = {
                'train_iter': train_iter,
                'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
                'episode_count': n_episode,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / n_episode,
                'evaluate_time': duration,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_time_per_episode': n_episode / duration,
                'reward_mean': np.mean(episode_reward),
                'reward_std': np.std(episode_reward),
                'reward_max': np.max(episode_reward),
                'reward_min': np.min(episode_reward),
                # 'each_reward': episode_reward,
            }
            episode_info = eval_monitor.get_episode_info()
            if episode_info is not None:
                info.update(episode_info)
            self._logger.info(self._logger.get_tabulate_vars_hor(info))
            # self._logger.info(self._logger.get_tabulate_vars(info))
            for k, v in info.items():
                if k in ['train_iter', 'ckpt_name', 'each_reward']:
                    continue
                if not np.isscalar(v):
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)

            if render:
                video_title = '{}_{}/'.format(self._instance_name, self._render.mode)
                videos = eval_monitor.get_video()
                render_iter = envstep if self._render.mode == 'envstep' else train_iter
                from ding.utils import fps
                self._tb_logger.add_video(video_title, videos, render_iter, fps(self._env))

            eval_reward = np.mean(episode_reward)
            if eval_reward > self._max_eval_reward:
                if save_ckpt_fn:
                    save_ckpt_fn('ckpt_best.pth.tar')
                self._max_eval_reward = eval_reward
            stop_flag = eval_reward >= self._stop_value and train_iter > 0
            if stop_flag:
                self._logger.info(
                    "[DI-engine serial pipeline] " +
                    "Current eval_reward: {} is greater than stop_value: {}".format(eval_reward, self._stop_value) +
                    ", so your RL agent is converged, you can refer to " +
                    "'log/evaluator/evaluator_logger.txt' for details."
                )

        if get_world_size() > 1:
            objects = [stop_flag, return_info]
            dist.broadcast_object_list(objects, src=0)
            stop_flag, return_info = objects

        return stop_flag, return_info


def test_main(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int,
        dataset: list,
        model: Optional[torch.nn.Module] = None,
        max_iter=int(1e6),
) -> Union['Policy', bool]:  # noqa
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Env, Policy
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    # Random seed
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    if cfg.policy.device == 'cuda' and torch.cuda.is_available():
        cfg.policy.cuda = True
    else:
        cfg.policy.cuda = False
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'eval'])

    # Main components
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    print(f'len of dataset: {len(dataset)}')
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, dataset, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    learner.call_hook('before_run')
    stop, reward = evaluator.eval(learner.save_checkpoint, 1)
    learner.call_hook('after_run')
    print('final reward is: {}'.format(reward))
    return policy, stop


def get_dataset(path):
    with open(path, 'rb') as f:
        p = pickle.load(f)
    game_blocks = p[0]
    dataset = []
    for game_block in game_blocks:
        if game_block.obs_history.shape[0] == 209:
            game_block_action_length = 200
        else:
            # the last game_block
            game_block_action_length = game_block.obs_history.shape[0]-4

        for i in range(game_block_action_length):
            dataset.append({
                # 'obs': torch.tensor(h.obs_history[i + 9]).permute(2, 0, 1).float() / 255,
                'obs': torch.tensor(game_block.obs_history[i + 3]),
                'action': game_block.action_history[i],
                'reward': game_block.reward_history[i],

            })

    return_ = 0
    for i in range(len(dataset)):
        return_ += dataset[i]['reward']
    print(f'the return of episodes in dataset is: {return_}')

    return dataset


if __name__ == "__main__":
    from atari_efficientzero_collect_demo_config import main_config, create_config

    expert_path = 'ez_breakout_seed0_1eps.pkl'

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    real_dataset = get_dataset(expert_path)
    test_main([main_config, create_config], args.seed, real_dataset)


