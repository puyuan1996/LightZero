from easydict import EasyDict
import torch
torch.cuda.set_device(0)
# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_name = 'PongNoFrameskip-v4'

if env_name == 'PongNoFrameskip-v4':
    action_space_size = 6
elif env_name == 'QbertNoFrameskip-v4':
    action_space_size = 6
elif env_name == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
elif env_name == 'SpaceInvadersNoFrameskip-v4':
    action_space_size = 6
elif env_name == 'BreakoutNoFrameskip-v4':
    action_space_size = 4

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50

# update_per_collect = 1000

update_per_collect = None
model_update_ratio = 0.25
# model_update_ratio = 1


# num_simulations = 1 # TODO: only for debug
# update_per_collect = 1


batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.5
# reanalyze_ratio = 1

eps_greedy_exploration_in_collect = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    # mcts_ctree, muzero_collector: empty_cache
    # exp_name=f'data_mz_ctree/{env_name[:-14]}_muzero_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_new-rr{reanalyze_ratio}_46464_collect-orig_tep025_gsl50_noprio_target100_start2000_adamw1e-4_wd1e-4_seed0',

    exp_name=f'data_mz_ctree/{env_name[:-14]}_muzero_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_new-rr{reanalyze_ratio}_46464_train-per-collect-one-segment_tep025_gsl50_noprio_target100_start2000_adamw1e-4_wd1e-4_seed0',

    # exp_name=f'data_mz_ctree/{env_name[:-14]}_muzero_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_46464_train-per-collect-one-segment_temdecy-50k_seed0',
    # exp_name=f'data_mz_ctree_debug/{env_name[:-14]}_muzero_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_46464_collect-orig_seed0',
    
    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
        # obs_shape=(4, 96, 96),
        observation_shape=(4, 64, 64),
        frame_stack_num=4,
        gray_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: debug
        # collect_max_episode_steps=int(100),
        # eval_max_episode_steps=int(100),
    ),
    policy=dict(
        model=dict(
            # observation_shape=(4, 96, 96),
            observation_shape=(4, 64, 64),
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        cuda=True,
        env_type='not_board_games',
        # game_segment_length=400,
        game_segment_length=50,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # need to dynamically adjust the number of decay steps 
            # according to the characteristics of the environment and the algorithm
            type='linear',
            start=1.,
            end=0.05,
            decay=int(1e5),
        ),
        use_augmentation=True,
        model_update_ratio = model_update_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,

        # optim_type='SGD',
        # lr_piecewise_constant_decay=True,
        # learning_rate=0.2,

        # optim_type='SGD',
        # lr_piecewise_constant_decay=False,
        # learning_rate=1e-4,
        # weight_decay=1e-4,


        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=1e-4,
        weight_decay=1e-4,
        # weight_decay=0.1,


        # manual_temperature_decay=True,
        # threshold_training_steps_for_final_temperature=int(5e4),
        target_update_freq=100,
        use_priority=False,


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
atari_muzero_config = EasyDict(atari_muzero_config)
main_config = atari_muzero_config

atari_muzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)