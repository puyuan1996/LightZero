import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
# update_per_collect determines the number of training steps after each collection of a batch of data.
# For different env, we have different episode_length,
# we usually set update_per_collect = collector_env_num * episode_length * reuse_factor
update_per_collect = 1000
batch_size = 256
max_env_step = int(1e6)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

mspacman_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/mspacman_efficientzero_ns{num_simulations}_upc{update_per_collect}_ic1_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_name='MsPacmanNoFrameskip-v4',
        frame_skip=4,
        frame_stack_num=4,
        gray_scale=True,
        obs_shape=(4, 96, 96),
        manager=dict(shared_memory=False, ),
        stop_value=int(1e6),
    ),
    policy=dict(
        # the pretrained model path.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        model_path=None,
        env_name='MsPacmanNoFrameskip-v4',
        # whether to use cuda for network.
        cuda=True,
        model=dict(
            # ==============================================================
            # We use the default large size model, please refer to the
            # default init config in EfficientZeroModel class or EfficientZero
            # original paper for details.
            # ==============================================================
            image_channel=1,
            frame_stack_num=4,
            downsample=True,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 96, 96, 3] -> [4*3, 96, 96]
            # observation_shape=(12, 96, 96),  # if frame_stack_num=4, gray_scale=False
            # observation_shape=(3, 96, 96),  # if frame_stack_num=1, gray_scale=False
            observation_shape=(4, 96, 96),  # if frame_stack_num=4, gray_scale=True
            # NOTE: the action_space_size=9 for mspacman env.
            action_space_size=9,
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            categorical_distribution=True,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='SGD',
            lr_manually=True,
            learning_rate=0.2,  # init lr for manually decay schedule
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # Get "n_episode" episodes per collect.
            n_episode=n_episode,
        ),
        # If the eval cost is expensive, we could set eval_freq larger.
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        # command_mode config
        other=dict(
            # NOTE: the replay_buffer_size is ineffective, we specify it in game config
            replay_buffer=dict(type='game_buffer_efficientzero')
        ),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='not_board_games',
        battle_mode='play_with_bot_mode',
        game_history_length=400,

        ## observation
        # the key difference setting between image-input and vector input
        image_based=True,
        use_augmentation=True,

        ## reward
        clip_reward=True,

        ## learn
        num_simulations=num_simulations,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=2,
        # the size/capacity of replay_buffer
        replay_buffer_size=int(1e5),
        # ``max_training_steps`` is only used for adjusting temperature manually.
        max_training_steps=int(1e5),

        ## reanalyze
        reanalyze_ratio=0.3,
        reanalyze_outdated=True,
        # whether to use root value in reanalyzing part
        use_root_value=False,
        mini_infer_size=256,

        ## priority
        use_priority=True,
        use_max_priority_for_new_data=True,
        # ==============================================================
        # end of additional game_config
        # ==============================================================
    ),
)
mspacman_efficientzero_config = EasyDict(mspacman_efficientzero_config)
main_config = mspacman_efficientzero_config

mspacman_efficientzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_efficientzero',
        get_train_sample=True,
        import_names=['lzero.worker.efficientzero_collector'],
    )
)
mspacman_efficientzero_create_config = EasyDict(mspacman_efficientzero_create_config)
create_config = mspacman_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import serial_pipeline_efficientzero
    serial_pipeline_efficientzero([main_config, create_config], seed=0, max_env_step=max_env_step)
