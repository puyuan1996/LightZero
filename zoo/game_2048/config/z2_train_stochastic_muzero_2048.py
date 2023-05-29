from easydict import EasyDict

env_name = 'game_2048'
action_space_size = 4
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 3
# num_simulations = 50  # TODO(pu):100
# update_per_collect = 200
# batch_size = 256
# max_env_step = int(1e6)
# reanalyze_ratio = 0.

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 100
batch_size = 256
max_env_step = int(1e8)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

game_2048_stochastic_muzero_config = dict(
    exp_name=f'data_mz_ctree/game2_2048_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs{batch_size}_sslw2_rew-morm-true_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
        obs_shape=(16, 4, 4),
        obs_type='dict_observation',
        reward_scale=100,
        reward_normalize=False,
        max_tile=int(2**16),  # 2**11=2048, 2**16=65536
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(16, 4, 4),
            action_space_size=action_space_size,
            image_channel=16,
            # NOTE: whether to use the self_supervised_learning_loss. default is False
            self_supervised_learning_loss=True,
        ),
        mcts_ctree=True,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        td_steps=10,
        discount_factor=0.999,
        manual_temperature_decay=True,
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,  # init lr for manually decay schedule
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,  # default is 0
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
game_2048_stochastic_muzero_config = EasyDict(game_2048_stochastic_muzero_config)
main_config = game_2048_stochastic_muzero_config

game_2048_stochastic_muzero_create_config = dict(
    env=dict(
        type='game_2048',
        import_names=['zoo.game_2048.envs.game_2048_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero',
        import_names=['lzero.policy.stochastic_muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
game_2048_stochastic_muzero_create_config = EasyDict(game_2048_stochastic_muzero_create_config)
create_config = game_2048_stochastic_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
