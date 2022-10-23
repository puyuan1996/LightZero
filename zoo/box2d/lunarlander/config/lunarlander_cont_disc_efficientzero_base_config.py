import torch
from easydict import EasyDict

from core.rl_utils import GameBaseConfig, DiscreteSupport

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

game_config = EasyDict(
    dict(
        env_name='lunarlander_cont_disc',
        env_type='no_board_games',
        device=device,
        # mcts_ctree=False,
        mcts_ctree=True,

        # TODO: for board_games, mcts_ctree now only support env_num=1, because in cpp MCTS root node,
        #  we must specify the one same action mask,
        #  when env_num>1, the action mask for different env may be different.
        battle_mode='one_player_mode',
        game_history_length=200,

        image_based=False,
        cvt_string=False,

        # clip_reward=True,
        # TODO(pu)
        clip_reward=False,

        game_wrapper=True,
        action_space_size=16,  # 4**2
        amp_type='none',


        # [S, W, H, C] -> [S x C, W, H]
        # [4,8,1,1] -> [4*1, 8, 1]
        image_channel=1,
        obs_shape=(4, 8, 1),  # if frame_stack_nums=4
        frame_stack_num=4,

        # obs_shape=(1, 8, 1),  # if frame_stack_num=1
        # frame_stack_num=1,

        gray_scale=False,
        downsample=False,
        vis_result=True,
        # TODO(pu): test the effect of augmentation,
        # use_augmentation=True,  # only for atari image obs
        use_augmentation=False,
        # Style of augmentation
        # choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity']
        augmentation=['shift', 'intensity'],

        # debug
        # collector_env_num=1,
        # evaluator_env_num=1,
        # num_simulations=9,
        # batch_size=4,
        # total_transitions=int(1e5),
        # lstm_hidden_size=256,
        # # # to make sure the value target is the final outcome
        # td_steps=5,
        # num_unroll_steps=3,
        # lstm_horizon_len=3,

        collector_env_num=8,
        evaluator_env_num=5,
        # num_simulations=50,
        # TODO
        num_simulations=100,
        batch_size=256,
        total_transitions=int(1e5),
        lstm_hidden_size=512,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,

        # TODO(pu): why 0.99?
        revisit_policy_search_rate=0.99,

        # TODO(pu): why not use adam?
        # lr_manually=True,
        lr_manually=False,

        # TODO(pu): if true, no priority to sample
        use_max_priority=True,  # if true, sample without priority
        # use_max_priority=False,
        use_priority=True,

        # TODO(pu): only used for adjust temperature manually
        max_training_steps=int(1e5),
        auto_temperature=False,
        # only effective when auto_temperature=False
        fixed_temperature_value=0.25,
        # TODO(pu): whether to use root value in reanalyzing?
        use_root_value=False,

        # TODO(pu): test the effect
        last_linear_layer_init_zero=True,
        state_norm=False,
        mini_infer_size=2,
        # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
        priority_prob_alpha=0.6,
        # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
        # TODO(pu): test effect of 0.4->1
        priority_prob_beta=0.4,
        prioritized_replay_eps=1e-6,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        auto_td_steps=int(0.3 * 2e5),
        auto_td_steps_ratio=0.3,

        # UCB formula
        pb_c_base=19652,
        pb_c_init=1.25,
        support_size=300,
        value_support=DiscreteSupport(-300, 300, delta=1),
        reward_support=DiscreteSupport(-300, 300, delta=1),
        max_grad_norm=10,
        test_interval=10000,
        log_interval=1000,
        vis_interval=1000,
        checkpoint_interval=100,
        target_model_interval=200,
        save_ckpt_interval=10000,
        discount=0.997,
        dirichlet_alpha=0.3,
        value_delta_max=0.01,
        num_actors=1,
        # network initialization/ & normalization
        episode_life=True,
        # replay window
        start_transitions=8,
        transition_num=1,
        # frame skip & stack observation
        frame_skip=4,
        # TODO(pu): EfficientZero -> MuZero
        # coefficient
        reward_loss_coeff=1,
        value_loss_coeff=0.25,
        policy_loss_coeff=1,
        consistency_coeff=2,

        # siamese
        proj_hid=1024,
        proj_out=1024,
        pred_hid=512,
        pred_out=1024,
        bn_mt=0.1,
        blocks=1,  # Number of blocks in the ResNet
        reduced_channels_reward=16,  # x36 Number of channels in reward head
        reduced_channels_value=16,  # x36 Number of channels in value head
        reduced_channels_policy=16,  # x36 Number of channels in policy head
        resnet_fc_reward_layers=[32],  # Define the hidden layers in the reward head of the dynamic network
        resnet_fc_value_layers=[32],  # Define the hidden layers in the value head of the prediction network
        resnet_fc_policy_layers=[32],  # Define the hidden layers in the policy head of the prediction network
    )
)

game_config = GameBaseConfig(game_config)