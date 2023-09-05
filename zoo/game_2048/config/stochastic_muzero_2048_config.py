from easydict import EasyDict


# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
env_name = 'game_2048'
action_space_size = 4
use_ture_chance_label_in_chance_encoder = True  # option: {True, False}
# use_ture_chance_label_in_chance_encoder = False  # option: {True, False}

num_of_possible_chance_tile = 2
chance_space_size = 16 * num_of_possible_chance_tile

# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 3
# num_simulations = 100
# update_per_collect = 200
# batch_size = 512
# max_env_step = int(1e9)
# reanalyze_ratio = 0.


# debug config
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 5
update_per_collect = 2
batch_size = 10240
max_env_step = int(1e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

game_2048_stochastic_muzero_config = dict(
    exp_name=f'data_stochastic_mz_ctree/game_2048_npct-{num_of_possible_chance_tile}_stochastic_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs{batch_size}_chance-{use_ture_chance_label_in_chance_encoder}_sslw2_seed0',
    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
        obs_shape=(16, 4, 4),
        obs_type='dict_observation',
        raw_reward_type='raw',  # 'merged_tiles_plus_log_max_tile_num'
        reward_normalize=False,
        reward_scale=100,
        max_tile=int(2**16),  # 2**11=2048, 2**16=65536
        num_of_possible_chance_tile=num_of_possible_chance_tile,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path='/Users/puyuan/code/LightZero/zoo/game_2048/tb/game_2048_nct-2_stochastic_muzero_ns100_upc200_rr0.0_bs512_chance-True-32_sslw2_rbs1e6_seed0/ckpt/ckpt_best.pth.tar',
        model=dict(
            observation_shape=(16, 4, 4),
            action_space_size=action_space_size,
            chance_space_size=chance_space_size,
            image_channel=16,
            # NOTE: whether to use the self_supervised_learning_loss. default is False
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        use_ture_chance_label_in_chance_encoder=use_ture_chance_label_in_chance_encoder,
        mcts_ctree=True,
        gumbel_algo=False,
        cuda=True,
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        td_steps=10,
        discount_factor=0.999,
        manual_temperature_decay=True,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        weight_decay=1e-4,
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
)
game_2048_stochastic_muzero_create_config = EasyDict(game_2048_stochastic_muzero_create_config)
create_config = game_2048_stochastic_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], model_path=main_config.policy.model_path, seed=0, max_env_step=max_env_step)
