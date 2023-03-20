from easydict import EasyDict
from torch.nn import ReLU

main_config = dict(
    exp_name='breakout_bc_seed0',
    env=dict(
        manager=dict(
            episode_num=float('inf'),
            max_retry=5,
            step_timeout=None,
            auto_reset=True,
            reset_timeout=None,
            retry_type='reset',
            retry_waiting_time=0.1,
            shared_memory=False,
            copy_on_get=True,
            context='fork',
            wait_num=float('inf'),
            step_wait_timeout=None,
            connect_timeout=60,
            reset_inplace=False,
            cfg_type='SyncSubprocessEnvManagerDict',
            type='subprocess',
        ),
        stop_value=1000000,
        # collector_env_num=8,
        # evaluator_env_num=3,
        # n_evaluator_episode=3,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        env_name='BreakoutNoFrameskip-v4',
        channel_last=True,
        obs_shape=[4, 96, 96],
        render_mode_human=False,
        collect_max_episode_steps=10800,
        eval_max_episode_steps=10800,
        max_episode_steps=108000,
        frame_skip=4,
        episode_life=True,
        clip_rewards=False,
        scale=True,
        warp_frame=True,
        save_video=False,
        gray_scale=True,
        cvt_string=False,
        game_wrapper=True,
        dqn_expert_data=False,
        cfg_type='AtariLightZeroEnvDict',
        frame_stack_num=1
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[1, 96, 96],
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            optimizer='Adam',
            lr=1e-4,
            learning_rate=0.001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=400000, ),
        ),
    ),
)
main_config = EasyDict(main_config)
main_config = main_config
create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(
        cfg_type='SyncSubprocessEnvManagerDict',
        # type='subprocess',
        type='base',

    ),
    policy=dict(type='bc'),
)
create_config = EasyDict(create_config)
create_config = create_config


if __name__ == "__main__":
    from lzero.entry import serial_pipeline_efficientzero_collect_demo
    serial_pipeline_efficientzero_collect_demo([main_config, create_config], seed=0, max_env_step=1e6)
