import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_name = 'BreakoutNoFrameskip-v4'

if env_name == 'PongNoFrameskip-v4':
    action_space_size = 6
    average_episode_length_when_converge = 2000
elif env_name == 'QbertNoFrameskip-v4':
    action_space_size = 6
    average_episode_length_when_converge = 2000
elif env_name == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
    average_episode_length_when_converge = 500
elif env_name == 'SpaceInvadersNoFrameskip-v4':
    action_space_size = 6
    average_episode_length_when_converge = 1000
elif env_name == 'BreakoutNoFrameskip-v4':
    action_space_size = 4
    average_episode_length_when_converge = 800


# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 50
update_per_collect = 1000
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/{env_name[:-14]}_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
        obs_shape=(4, 96, 96),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        max_episode_steps=int(1.08e5),
        channel_last=True,
        scale=True,
        render_mode_human=False,
        # render_mode_human=True,
        # episode_life=True,
        clip_rewards=False,
        episode_life=False,
        collect_max_episode_steps=int(5e3),
    ),
    policy=dict(
        # collect_demo related
        # mcts_ctree=False,
        mcts_ctree=True,

        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='not_board_games',
        game_block_length=200,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        manual_temperature_decay=False,
        fixed_temperature_value=0.25,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_piecewise_constant_decay=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
        ),
        model=dict(
            observation_shape=(4, 96, 96),
            action_space_size=action_space_size,
            representation_network_type='conv_res_blocks',
        ),
        collect=dict(n_episode=n_episode, ),
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
    ),
)
atari_efficientzero_config = EasyDict(atari_efficientzero_config)
main_config = atari_efficientzero_config

atari_efficientzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero_collect_demo',
        import_names=['lzero.policy.efficientzero_collect_demo'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector_collect_demo'],
    )
)
atari_efficientzero_create_config = EasyDict(atari_efficientzero_create_config)
create_config = atari_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import collect_muzero
    # collect_demo related
    # the pretrained model path.
    # Users should add their own model path here. Model path should lead to a model.
    # Absolute path is recommended.
    # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
    model_path = '/Users/puyuan/code/LightZero/zoo/atari/tb/Breakout_efficientzero_ns50_upc1000_rr0.0_seed0_ms1e4/ckpt/ckpt_best.pth.tar'

    data_path = 'ez_breakout_seed0_1eps_return422_ctree_epslife-f_ms5e3.pkl'
    # TODO：
    # data_path = 'ez_breakout_seed0_1eps_return422_ctree_epslife-f_leaf-node_leaf-hidden-state_search-path_ms5e3.pkl'

    collect_muzero([main_config, create_config], seed=0, max_env_step=max_env_step, model_path=model_path, data_path=data_path)
