from easydict import EasyDict
import torch
torch.cuda.set_device(0)
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================


collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 25
update_per_collect = None
model_update_ratio = 0.5
max_env_step = int(2e5)
reanalyze_ratio = 0
batch_size = 64
num_unroll_steps = 5
# num_unroll_steps = 2


# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cartpole_unizero_config = dict(
    # TODO: world_model.py decode_obs_tokens
    # TODO: tokenizer: lpips loss
    exp_name=f'data_debug/cartpole_unizero_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_bs{batch_size}_seed0',
    env=dict(
        env_name='CartPole-v0',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        tokenizer_start_after_envsteps=int(0),
        transformer_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            observation_shape=4,
            action_space_size=2,
            model_type='mlp',
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            # reward_support_size=601,
            # value_support_size=601,
            # support_scale=300,
            reward_support_size=21,
            value_support_size=21,
            support_scale=10,
        ),
        cuda=True,
        use_augmentation=False,
        env_type='not_board_games',
        game_segment_length=50,
        model_update_ratio=model_update_ratio,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value = 0.5, # TODO: 10
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        # eval_freq=int(1e4),
        eval_freq=int(500),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

cartpole_unizero_config = EasyDict(cartpole_unizero_config)
main_config = cartpole_unizero_config

cartpole_unizero_create_config = dict(
    env=dict(
        type='cartpole_lightzero',
        import_names=['zoo.classic_control.cartpole.envs.cartpole_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
cartpole_unizero_create_config = EasyDict(cartpole_unizero_create_config)
create_config = cartpole_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)

    # 下面为cprofile的代码
    # from lzero.entry import train_unizero
    # def run(max_env_step: int):
    #     train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)
    # import cProfile
    # cProfile.run(f"run({10000})", filename="cartpole_unizero_ctree_cprofile_10k_envstep", sort="cumulative")