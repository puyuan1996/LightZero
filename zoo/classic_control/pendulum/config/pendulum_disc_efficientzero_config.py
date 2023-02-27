import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 3
# num_simulations = 50
# # update_per_collect determines the number of training steps after each collection of a batch of data.
# # For different env, we have different episode_length,
# # we usually set update_per_collect = collector_env_num * episode_length * reuse_factor
# update_per_collect = 200
# batch_size = 256
# max_env_step = int(1e6)

## debug config
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 5
update_per_collect = 2
batch_size = 4
max_env_step = int(1e4)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pendulum_disc_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/pendulum_disc_efficientzero_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        continuous=False,
        norm_obs=dict(use_norm=False, ),
        manager=dict(shared_memory=False, ),
        stop_value=int(1e6),
    ),
    policy=dict(
        # the pretrained model path.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        model_path=None,
        env_name='pendulum_disc',
        # whether to use cuda for network.
        cuda=True,
        model=dict(
            # ==============================================================
            # We use the small size model for pendulum.
            # ==============================================================
            # NOTE: the key difference setting between image-input and vector input.
            image_channel=1,
            frame_stack_num=1,
            downsample=False,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [1, 3, 1, 1] -> [1*1, 3, 1]
            # observation_shape=(4, 3, 1),  # if frame_stack_num=4
            observation_shape=(1, 3, 1),  # if frame_stack_num=1
            action_space_size=11,
            num_res_blocks=1,
            num_channels=16,
            lstm_hidden_size=128,
            support_scale=25,
            reward_support_size=51,
            value_support_size=51,
            # whether to use discrete support to represent categorical distribution for value, value_prefix.
            categorical_distribution=True,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_manually=True,
            optim_type='SGD',
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
        other=dict(
            # NOTE: the replay_buffer_size is ineffective,
            # we specify it using ``replay_buffer_size`` in the following game config
            replay_buffer=dict(type='game_buffer_efficientzero')
        ),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        # ## common
        mcts_ctree=True,
        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='not_board_games',
        game_block_length=50,

        ## observation
        # the key difference setting between image-input and vector input.
        image_based=False,
        cvt_string=False,
        gray_scale=False,
        downsample=False,
        use_augmentation=False,

        ## reward
        clip_reward=False,

        ## learn
        num_simulations=num_simulations,
        lr_manually=True,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        support_size=25,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        # NOTE: for vector input, we don't use the ssl loss.
        ssl_loss_weight=0,
        # the size/capacity of replay_buffer
        replay_buffer_size=int(1e6),
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

pendulum_disc_efficientzero_config = EasyDict(pendulum_disc_efficientzero_config)
main_config = pendulum_disc_efficientzero_config

pendulum_disc_efficientzero_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['zoo.classic_control.pendulum.envs.pendulum_lightzero_env'],
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
pendulum_disc_efficientzero_create_config = EasyDict(pendulum_disc_efficientzero_create_config)
create_config = pendulum_disc_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import serial_pipeline_efficientzero
    serial_pipeline_efficientzero([main_config, create_config], seed=0, max_env_step=max_env_step)
