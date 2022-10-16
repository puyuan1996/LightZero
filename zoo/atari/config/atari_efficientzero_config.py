from easydict import EasyDict

from atari_efficientzero_base_config import game_config
from core.model import RepresentationNetwork

representation_model = RepresentationNetwork(
    observation_shape=(12, 96, 96),
    num_blocks=1,
    num_channels=64,
    downsample=True,
    momentum=0.1,
)

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3

atari_efficientzero_config = dict(
    exp_name='data_ez_ctree/pong_efficientzero_seed0_lr0.2_ns50_ftv025_sub883_upc2000',
    # exp_name='data_ez_ptree/pong_efficientzero_seed0_lr0.2_ns50_ftv025_upc1000',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=20,
        env_name='PongNoFrameskip-v4',
        max_episode_steps=int(1.08e5),
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        # for debug
        # max_episode_steps=int(100),
        # collect_max_episode_steps=int(100),
        # eval_max_episode_steps=int(100),
        frame_skip=4,
        obs_shape=(12, 96, 96),
        episode_life=True,
        gray_scale=False,
        # cvt_string=True,
        # trade memory for speed
        cvt_string=False,
        game_wrapper=True,
        dqn_expert_data=False,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        env_name='PongNoFrameskip-v4',
        # TODO(pu): how to pass into game_config, which is class, not a dict
        # game_config=game_config,
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            projection_input_dim_type='atari',
            representation_model_type='conv_res_blocks',
            # representation_model=representation_model,
            observation_shape=(12, 96, 96),  # 3,96,96 stack=4
            action_space_size=6,
            downsample=True,
            num_blocks=1,
            num_channels=64,  # Number of channels in the ResNet, default config in EZ original repo
            lstm_hidden_size=512,  # default config in EZ original repo
            # The env step is twice as large as the original size model when converging
            # num_channels=32,  # Number of channels in the ResNet, for time efficiency
            # lstm_hidden_size=256,  # for time efficiency
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            bn_mt=0.1,
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            # for debug
            # update_per_collect=2,
            # batch_size=4,

            update_per_collect=2000,
            batch_size=256,

            learning_rate=0.2,
            # Frequency of target network update.
            target_update_freq=400,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=n_episode,
        ),
        # the eval cost is expensive, so we set eval_freq larger
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        # for debug
        # eval=dict(evaluator=dict(eval_freq=int(2), )),
        # command_mode config
        other=dict(
            # NOTE: the replay_buffer_size is ineffective, we specify it in game config
            replay_buffer=dict(type='game')
        ),
    ),
)
atari_efficientzero_config = EasyDict(atari_efficientzero_config)
main_config = atari_efficientzero_config

atari_efficientzero_create_config = dict(
    env=dict(
        type='atari-muzero',
        import_names=['zoo.atari.envs.atari_muzero_env'],
    ),
    # env_manager=dict(type='base'),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['core.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['core.worker.collector.muzero_collector'],
    )
)
atari_efficientzero_create_config = EasyDict(atari_efficientzero_create_config)
create_config = atari_efficientzero_create_config

if __name__ == "__main__":
    from core.entry import serial_pipeline_efficientzero
    serial_pipeline_efficientzero([main_config, create_config], seed=0, max_env_step=int(1e6), game_config=game_config)
