from easydict import EasyDict

dmc2gym_sac_config = dict(
    exp_name='eval_dmc2gym_cheetah_run_sac_state_seed0',
    env=dict(
        env_id='dmc2gym-v0',
        # domain_name="cartpole",
        # task_name="swingup",
        # frame_skip=8,
        domain_name="cheetah",
        task_name="run",
        frame_skip=2,
        frame_stack=1,
        from_pixels=False,  # state obs
        channels_first=False,  # obs shape (height, width, 3)

        # save_replay_gif=False,
        save_replay_gif=True,
        replay_path_gif='./replay_gif_sac_cheetah',
        collector_env_num=16,
        # evaluator_env_num=8,
        # n_evaluator_episode=8,
        evaluator_env_num=2,
        n_evaluator_episode=2,
        stop_value=1e6,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_type='state',
        cuda=True,
        random_collect_size=10000,
        # model_path=None,
        model_path='/mnt/afs/niuyazhe/code/LightZero/dmc2gym_sac_state_nseed_5M_240903_152757/ckpt/iteration_100000.pth.tar',
        model=dict(
            # obs_shape=5,
            # action_shape=1,
            obs_shape=17,
            action_shape=6,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            ignore_done=True,
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=True,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

dmc2gym_sac_config = EasyDict(dmc2gym_sac_config)
main_config = dmc2gym_sac_config

dmc2gym_sac_create_config = dict(
    env=dict(
        type='dmc2gym',
        import_names=['dizoo.dmc2gym.envs.dmc2gym_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
dmc2gym_sac_create_config = EasyDict(dmc2gym_sac_create_config)
create_config = dmc2gym_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_sac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), model_path=main_config.policy.model_path, seed=0)