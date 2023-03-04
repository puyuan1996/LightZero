import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# only used for adjusting temperature/lr manually
average_episode_length_when_converge = 5
threshold_env_steps_for_final_lr = int(5e4)
threshold_env_steps_for_final_temperature = int(1e5)

collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 25
# update_per_collect determines the number of training steps after each collection of a batch of data.
# For different env, we have different episode_length,
# we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)

# debug config
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 2
# update_per_collect = 5
# batch_size = 4
# max_env_step = int(2e3)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

tictactoe_muzero_config = dict(
    exp_name=f'data_mz_ctree/tictactoe_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_seed1',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        battle_mode='play_with_bot_mode',
        channel_last=True,
        manager=dict(shared_memory=False, ),
        stop_value=int(2),
    ),
    policy=dict(
        # the pretrained model path.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        model_path=None,
        env_name='tictactoe',
        # whether to use cuda for network.
        cuda=True,
        model=dict(
            # ==============================================================
            # We use the small size model for tictactoe
            # ==============================================================
            # NOTE: the key difference setting between image-input and vector input.
            image_channel=1,
            frame_stack_num=1,
            downsample=False,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 3, 3, 3] -> [12, 3, 3]
            # observation_shape=(12, 3, 3),  # if frame_stack_num=4
            observation_shape=(3, 3, 3),  # if frame_stack_num=1
            action_space_size=9,
            # whether to use discrete support to represent categorical distribution for value, reward.
            categorical_distribution=True,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            ## small size model
            num_res_blocks=1,
            num_channels=16,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_reward_layers=[8],
            fc_value_layers=[8],
            fc_policy_layers=[8],
            support_scale=10,
            reward_support_size=21,
            value_support_size=21,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_manually=False,
            optim_type='Adam',
            learning_rate=0.003,  # lr for Adam optimizer
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
            replay_buffer=dict(
                type='game_buffer_muzero',
                # the size/capacity of replay_buffer, in the terms of transitions.
                replay_buffer_size=int(1e6),
            )
        ),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        # debug
        # mcts_ctree=False,

        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='board_games',
        game_block_length=5,

        ## observation
        # NOTE: the key difference setting between image-input and vector input
        image_based=False,
        cvt_string=False,
        gray_scale=False,
        use_augmentation=False,

        ## reward
        clip_reward=False,

        ## learn
        num_simulations=num_simulations,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=9,
        num_unroll_steps=3,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        # ``threshold_training_steps_for_final_lr`` is only used for adjusting lr manually.
        threshold_training_steps_for_final_lr=int(
            threshold_env_steps_for_final_lr / collector_env_num / average_episode_length_when_converge * update_per_collect),
        # ``threshold_training_steps_for_final_temperature`` is only used for adjusting temperature manually.
        threshold_training_steps_for_final_temperature=int(
            threshold_env_steps_for_final_temperature / collector_env_num / average_episode_length_when_converge * update_per_collect),

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
tictactoe_muzero_config = EasyDict(tictactoe_muzero_config)
main_config = tictactoe_muzero_config

tictactoe_muzero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
tictactoe_muzero_create_config = EasyDict(tictactoe_muzero_create_config)
create_config = tictactoe_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import serial_pipeline_muzero
    serial_pipeline_muzero([main_config, create_config], seed=1, max_env_step=max_env_step)
