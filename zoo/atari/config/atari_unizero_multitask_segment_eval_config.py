from easydict import EasyDict

def create_config(env_id, action_space_size, collector_env_num, evaluator_env_num, n_episode, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size):
    return EasyDict(dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=(3, 64, 64),
            # observation_shape=(3, 96, 96),
            gray_scale=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            full_action_space=True,
            collect_max_episode_steps=int(5e3), # TODO ===========
            eval_max_episode_steps=int(5e3), # TODO ===========

            # ===== only for debug =====
            # collect_max_episode_steps=int(30),
            # eval_max_episode_steps=int(30),
            # collect_max_episode_steps=int(150), # TODO: DEBUG
            # eval_max_episode_steps=int(150),
            # collect_max_episode_steps=int(500),
            # eval_max_episode_steps=int(500),
        ),
        policy=dict(
            multi_gpu=True, # ======== Very important for ddp =============
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=50000,),),),  # replay_ratio=0.5时，大约对应100k envsteps 存储一次 default is 10000
            grad_correct_params=dict(
                # for MoCo
                MoCo_beta=0.5,
                MoCo_beta_sigma=0.5,
                MoCo_gamma=0.1,
                MoCo_gamma_sigma=0.5,
                MoCo_rho=0,
                # for CAGrad
                calpha=0.5,
                rescale=1,
            ),
            task_num=len(env_id_list), # ======  在ddp中需要替换为每个rank对应的task数量  ======
            task_id=0,
            model=dict(
                observation_shape=(3, 64, 64),
                # observation_shape=(3, 96, 96),
                action_space_size=action_space_size,
                norm_type=norm_type,
                # num_res_blocks=1, # NOTE: encoder for 1 game
                # num_channels=64,
                num_res_blocks=2,  # NOTE: encoder for 4 game
                # num_channels=128,
                # num_res_blocks=4,  # NOTE: encoder for 8 game
                # num_res_blocks=2,  # NOTE: encoder for 8 game
                num_channels=256,
                world_model_cfg=dict(
                    env_id_list=env_id_list,
                    analysis_mode=True,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    # device='cpu',  # 'cuda',
                    device='cuda',  # 'cuda',
                    action_space_size=action_space_size,
                    # num_layers=2,  # NOTE
                    # ============== TODO: 改exp_name ==========
                    # NOTE: rl transformer
                    # batch_size=64 8games训练时，每张卡大约占12G cuda存储
                    # num_layers=4,  
                    # num_heads=8,   
                    # embed_dim=768,

                    # NOTE: gato-79M (small) transformer
                    # batch_size=64 8games训练时，每张卡大约占12*2=24G cuda存储
                    num_layers=8,  
                    num_heads=24,
                    embed_dim=768,

                    # NOTE: gato-medium 修改版 transformer
                    # batch_size=64 8games训练时，每张卡大约占12*3=36G cuda存储
                    # num_layers=12,  
                    # num_heads=24,
                    # embed_dim=768,

                    # NOTE: gato-medium 修改版 transformer
                    # batch_size=64 8games训练时，每张卡大约占12*2*4 cuda存储
                    # num_layers=8,  
                    # num_heads=24,
                    # embed_dim=1536,

                    # NOTE: gato-364M (medium) transformer
                    # batch_size=64 8games训练时，每张卡大约占12*3*4 cuda存储
                    # num_layers=12,  
                    # num_heads=12,
                    # embed_dim=1536,

                    # n_layer=12, 
                    # n_head=12,  # gpt2-base 124M parameters
                    # embed_dim=768,

                    obs_type='image',
                    # env_num=max(collector_env_num, evaluator_env_num),
                    env_num=8,  # TODO: the max of all tasks
                    # collector_env_num=collector_env_num,
                    # evaluator_env_num=evaluator_env_num,
                    task_num=len(env_id_list), # ====== total_task_num ======
                    use_normal_head=True,
                    # use_normal_head=False,
                    use_softmoe_head=False,
                    # use_moe_head=True,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,  # NOTE
                    # moe_in_transformer=True,
                    moe_in_transformer=False,  # NOTE
                    # multiplication_moe_in_transformer=True,
                    multiplication_moe_in_transformer=False,  # NOTE
                    num_experts_of_moe_in_transformer=4,
                    # num_experts_of_moe_in_transformer=2,
                ),
            ),
            total_batch_size=total_batch_size, #TODO=======
            # allocated_batch_sizes=True,#TODO=======
            allocated_batch_sizes=False,#TODO=======
            train_start_after_envsteps=int(0), # TODO
            use_priority=False,
            # print_task_priority_logs=False,
            # use_priority=True,  # TODO
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            game_segment_length=20,
            # update_per_collect=None,
            # update_per_collect=40, # TODO: 4/8games max-bs=64*4 8*20*0.25
            update_per_collect=80, # TODO: 26games max-bs=400, 8*20*1=160
            # update_per_collect=2, # TODO: 26games max-bs=400, 8*20*1=160
            replay_ratio=0.25,
            batch_size=batch_size,
            optim_type='AdamW',
            num_segments=num_segments,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            # replay_buffer_size=int(1e6),
            replay_buffer_size=int(5e5), # TODO
            # eval_freq=int(1e4),
            eval_freq=int(2e4),
            # eval_freq=int(1),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            # ============= The key different params for reanalyze =============
            # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 2 means reanalyze once every two epochs.
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
            reanalyze_batch_size=reanalyze_batch_size,
            # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
            reanalyze_partition=reanalyze_partition,
        ),
    ))

def generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size):
    configs = []
    # TODO
    # exp_name_prefix = f'data_unizero_mt_ddp-8gpu_1201/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_nlayer4-nhead8_seed{seed}/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-res2-channel256_gsl20_{len(env_id_list)}-pred-head_lsd768-nlayer4-nh8_mbs-512-bs64_upc80_seed{seed}/'
    exp_name_prefix = f'data_unizero_mt_ddp-8gpu_eval_1201/latent_state_tsne/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-res2-channel256_gsl20_{len(env_id_list)}-pred-head_lsd768-nlayer8-nh24_mbs-512-bs64_upc80_seed{seed}/'
    # exp_name_prefix = f'data_unizero_mt_ddp-1gpu_fintune_1201/{len(env_id_list)}games_eval200min_brf{buffer_reanalyze_freq}_nlayer12-nhead24_seed{seed}/{len(env_id_list)}games_brf{buffer_reanalyze_freq}_1-encoder-{norm_type}-res2-channel256_gsl20_{len(env_id_list)}-pred-head_lsd768-nlayer12-nh24_mbs-512-bs64_upc80_seed{seed}/'

    for task_id, env_id in enumerate(env_id_list):
        config = create_config(
            env_id,
            action_space_size,
            # collector_env_num if env_id not in ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'] else 2,  # TODO: different collector_env_num for Pong and Boxing
            # evaluator_env_num if env_id not in ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'] else 2,
            # n_episode if env_id not in ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'] else 2,
            collector_env_num,
            evaluator_env_num,
            n_episode,
            num_simulations,
            reanalyze_ratio,
            batch_size,
            num_unroll_steps,
            infer_context_length,
            norm_type,
            buffer_reanalyze_freq,
            reanalyze_batch_size,
            reanalyze_partition,
            num_segments,
            total_batch_size
        )
        config.policy.task_id = task_id
        config.exp_name = exp_name_prefix + f"{env_id.split('NoFrameskip')[0]}_unizero-mt_seed{seed}"

        configs.append([task_id, [config, create_env_manager()]])
    return configs


def create_env_manager():
    return EasyDict(dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='unizero_multitask',
            import_names=['lzero.policy.unizero_multitask'],
        ),
    ))

if __name__ == "__main__":
    from lzero.entry import train_unizero_multitask_segment_eval
    # TODO
    # env_id_list = [
    #     'PongNoFrameskip-v4',
    #     'MsPacmanNoFrameskip-v4',
    #     'SeaquestNoFrameskip-v4',
    #     'BoxingNoFrameskip-v4'
    # ]

    # 8games
    env_id_list = [
        'PongNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'BoxingNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'ChopperCommandNoFrameskip-v4',
        'HeroNoFrameskip-v4',
        'RoadRunnerNoFrameskip-v4',
    ]

    # env_id_list = [
    #     'DemonAttackNoFrameskip-v4',
    #         #     'AssaultNoFrameskip-v4',

    # #     'BankHeistNoFrameskip-v4',
    # #     'AmidarNoFrameskip-v4',
    # ]

    # 26games
    # env_id_list = [
    #     'PongNoFrameskip-v4',
    #     'MsPacmanNoFrameskip-v4',
    #     'SeaquestNoFrameskip-v4',
    #     'BoxingNoFrameskip-v4',
    #     'AlienNoFrameskip-v4',
    #     'ChopperCommandNoFrameskip-v4',
    #     'HeroNoFrameskip-v4',
    #     'RoadRunnerNoFrameskip-v4',

    #     'AmidarNoFrameskip-v4',
    #     'AssaultNoFrameskip-v4',
    #     'AsterixNoFrameskip-v4',
    #     'BankHeistNoFrameskip-v4',
    #     'BattleZoneNoFrameskip-v4',
    #     'CrazyClimberNoFrameskip-v4',
    #     'DemonAttackNoFrameskip-v4',
    #     'FreewayNoFrameskip-v4',
    #     'FrostbiteNoFrameskip-v4',
    #     'GopherNoFrameskip-v4',
    #     'JamesbondNoFrameskip-v4',
    #     'KangarooNoFrameskip-v4',
    #     'KrullNoFrameskip-v4',
    #     'KungFuMasterNoFrameskip-v4',
    #     'PrivateEyeNoFrameskip-v4',
    #     'UpNDownNoFrameskip-v4',
    #     'QbertNoFrameskip-v4',
    #     'BreakoutNoFrameskip-v4',
    # ]


    action_space_size = 18  # Full action space
    
    # TODO ==========
    import os 
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TIMEOUT"] = "3600000000"

    # for seed in [2, 3, 0, 1]: # TODO
    # for seed in [1]: # TODO
    for seed in [0]: # TODO
    
        collector_env_num = 8
        num_segments = 8
        n_episode = 8
        evaluator_env_num = 3
        num_simulations = 50
        max_env_step = int(4e5) # TODO
        reanalyze_ratio = 0.

        #应该根据一个样本sequence的占用显存量，和最大显存来设置
        total_batch_size = 512
        # total_batch_size = 3600
        batch_size = [int(min(100, total_batch_size / len(env_id_list))) for _ in range(len(env_id_list))]
        print(f'=========== batch_size: {batch_size} ===========')
        # batch_size = [int(64) for i in range(len(env_id_list))]

        num_unroll_steps = 10
        infer_context_length = 4
        norm_type = 'LN'

        # Defines the frequency of reanalysis. E.g., 1 means reanalyze once per epoch, 1/10 means reanalyze once every ten epochs.
        buffer_reanalyze_freq = 1/50 # TODO
        # Each reanalyze process will reanalyze <reanalyze_batch_size> sequences (<cfg.policy.num_unroll_steps> transitions per sequence)
        reanalyze_batch_size = 160
        # The partition of reanalyze. E.g., 1 means reanalyze_batch samples from the whole buffer, 0.5 means samples from the first half of the buffer.
        reanalyze_partition = 0.75

        # ======== TODO: only for debug ========
        collector_env_num = 2
        num_segments = 2
        n_episode = 2
        evaluator_env_num = 2
        num_simulations = 50
        # batch_size = [4,4,4,4,4,4,4,4]
        batch_size = [4 for i in range(len(env_id_list))]

        total_batch_size = 2

        configs = generate_configs(env_id_list, action_space_size, collector_env_num, n_episode, evaluator_env_num, num_simulations, reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length, norm_type, seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, num_segments, total_batch_size)

        """
        Overview:
            This script should be executed with <nproc_per_node> GPUs.
            Run the following command to launch the script:
            export NCCL_TIMEOUT=3600000  # NOTE
            python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 ./zoo/atari/config/atari_unizero_multitask_segment_finetune_config.py
            torchrun --nproc_per_node=8 ./zoo/atari/config/atari_unizero_multitask_segment_8games_ddp_config.py
        """
        from ding.utils import DDPContext
        from easydict import EasyDict

        # 8games
        pretrained_model_path = '/mnt/afs/niuyazhe/code/LightZero/data_unizero_mt_ddp-8gpu_1127/8games_brf0.02_nlayer8-nhead24_seed1/8games_brf0.02_1-encoder-LN-res2-channel256_gsl20_8-pred-head_lsd768-nlayer8-nh24_mbs-512-bs64_upc80_seed1/Pong_unizero-mt_seed1/ckpt/iteration_200000.pth.tar'
        
        # 26games
        # pretrained_model_path = '/mnt/afs/niuyazhe/code/LightZero/data_unizero_mt_ddp-8gpu-26game_1127/26games_brf0.02_nlayer8-nhead24_seed0/26games_brf0.02_1-encoder-LN-res2-channel256_gsl20_26-pred-head_lsd768-nlayer8-nh24_mbs-512-bs64_upc80_seed0/Pong_unizero-mt_seed0/ckpt/iteration_150000.pth.tar'
        with DDPContext():
            train_unizero_multitask_segment_eval(configs, seed=seed, model_path=pretrained_model_path, max_env_step=max_env_step)