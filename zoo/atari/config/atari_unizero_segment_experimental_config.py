"""UniZero Atari segment experiment launcher.

Defaults reproduce the current best arm of the 2026-09 MsPacman representation
stability matrix: observation-loss weight 1, separate encoder clipping, no
reanalysis, no prioritized replay, no augmentation, and the experimental
mechanisms (raw-token KV rebuild, contextual value bootstrap, open-loop
consistency loss) off. Pass ``--rebuild-kv-window-from-tokens
--bootstrap-value-context --open-loop-consistency-weight 1.0`` for the full v3
recipe. This module intentionally contains the performance, diagnostics,
recovery, and ablation controls used by iterative experiments. Keep the stable
baseline in ``atari_unizero_segment_config.py`` free of these overrides.
"""

import json
import logging
import os
import socket
import sys
import traceback
from datetime import datetime

from easydict import EasyDict
from zoo.atari.config._atari_unizero_segment_utils import _resolve_grad_clip_mode, _Tee, _safe_run_name
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map


_CORE_TB_METRIC_FILTER = {
    metric_name: True for metric_name in (
        # Loss/target quality: all are populated by every standard UniZero update.
        'loss/total', 'loss/obs', 'loss/policy', 'loss/policy_orig',
        'loss/policy_entropy', 'loss/value', 'loss/reward',
        'target/reward', 'target/value', 'target/policy_entropy',
        'target/transformed_reward', 'target/transformed_value',

        # Replay, segment validity, optimizer, and resource health.
        'priority/value', 'priority/mean', 'priority/max', 'priority/valid_ratio',
        'segment/length_actual', 'segment/valid_ratio',
        'replay/value_priority_mean', 'replay/value_priority_std',
        'replay/value_priority_min', 'replay/value_priority_max',
        'replay/buffer_fill_fraction', 'replay/sample_age_fraction_mean',
        'replay/sample_age_fraction_std', 'replay/sample_oldest_quarter_fraction',
        'replay/sample_newest_quarter_fraction',
        'replay/target_reward_mean', 'replay/target_reward_std',
        'replay/target_reward_min', 'replay/target_reward_max',
        'replay/target_reward_nonzero_fraction',
        'replay/target_reward_positive_fraction',
        'replay/target_value_mean', 'replay/target_value_std',
        'replay/target_value_min', 'replay/target_value_max',
        'lr', 'grad_norm', 'param_norm', 'weight_decay',
        'grad/clip_threshold', 'grad/clip_applied', 'grad/clip_scale',
        'grad/world_model_post_clip_norm',
        'memory/current_gpu_gb', 'memory/max_gpu_gb',
        'collect/epsilon', 'collect/mcts_temperature', 'schedule/policy_label_eps',

        # Learning dynamics and value calibration used for plateau diagnosis.
        'analysis/dormant_ratio_encoder', 'analysis/dormant_ratio_transformer',
        'analysis/dormant_ratio_head', 'analysis/avg_weight_mag_encoder',
        'analysis/avg_weight_mag_transformer', 'analysis/avg_weight_mag_head',
        'analysis/e_rank_last_linear', 'analysis/e_rank_sim_norm',
        'analysis/latent_state_l2_norms', 'analysis/latent_action_l2_norms',
        'analysis/first_step_loss_value', 'analysis/first_step_loss_policy',
        'analysis/first_step_loss_rewards', 'analysis/first_step_loss_obs',
        'analysis/middle_step_loss_value', 'analysis/middle_step_loss_policy',
        'analysis/middle_step_loss_rewards', 'analysis/middle_step_loss_obs',
        'analysis/last_step_loss_value', 'analysis/last_step_loss_policy',
        'analysis/last_step_loss_rewards', 'analysis/last_step_loss_obs',
        'temperature_value', 'temperature_reward', 'temperature_policy',
        'value_calibration/pred_mean', 'value_calibration/pred_std',
        'value_calibration/target_mean', 'value_calibration/target_std',
        'value_calibration/bias', 'value_calibration/mae',
        'value_calibration/rmse', 'value_calibration/correlation',
        'target/param_lag_l2', 'target/param_lag_relative',
        'target/online_param_l2', 'target/ema_param_l2',
        'target/latent_lag_l2_mean', 'target/latent_lag_relative_mean',
        'target/latent_cosine_mean', 'target/latent_l2_mean',

        # Periodic model/gradient/activation checks. They are absent between the start
        # of training and their first scheduled calculation, rather than emitted as zero.
        'norm/encoder/_total_norm', 'norm/transformer/_total_norm',
        'norm/head_value/_total_norm', 'norm/head_reward/_total_norm',
        'norm/head_policy/_total_norm', 'grad/encoder/_total_norm',
        'grad/transformer/_total_norm', 'grad/action_embedding/_total_norm',
        'grad/head_observation/_total_norm', 'grad/head_reward/_total_norm',
        'grad/head_value/_total_norm', 'grad/head_policy/_total_norm',
        'grad/encoder/global_norm_fraction', 'grad/transformer/global_norm_fraction',
        'grad/action_embedding/global_norm_fraction',
        'grad/head_observation/global_norm_fraction',
        'grad/head_reward/global_norm_fraction', 'grad/head_value/global_norm_fraction',
        'grad/head_policy/global_norm_fraction',
        'norm/x_token/mean', 'norm/x_token/std', 'norm/x_token/max', 'norm/x_token/min',
        'activation/x_token/feature_std_mean', 'activation/x_token/feature_std_min',
        'activation/x_token/near_constant_fraction',
        'logits/value/mean', 'logits/value/std', 'logits/value/max',
        'logits/value/min', 'logits/value/abs_max',
        'logits/policy/mean', 'logits/policy/std', 'logits/policy/max',
        'logits/policy/min', 'logits/policy/abs_max',
        'logits/reward/mean', 'logits/reward/std', 'logits/reward/max',
        'logits/reward/min', 'logits/reward/abs_max',
        'embeddings/obs/norm_mean', 'embeddings/obs/norm_std',
        'embeddings/obs/norm_max', 'embeddings/obs/norm_min',
        'stability/warning_count', 'stability/last_check_iter',
    )
}


def _build_learn_tb_metric_filter(
        *, use_priority, reanalysis_enabled, bootstrap_value_context,
        grad_clip_mode, adaptive_entropy_enabled, encoder_clip_enabled,
        open_loop_diagnostic_freq, gradient_diagnostic_freq,
        open_loop_consistency_weight, open_loop_recurrent_weight,
):
    """Return metrics that are genuinely produced by the resolved experiment.

    Optional features often expose placeholder zeros in ``_forward_learn`` for API
    compatibility.  Filter those feature families at configuration time; never filter
    by the observed scalar value because zero can be a valid learning signal.
    """
    metric_filter = _CORE_TB_METRIC_FILTER.copy()

    if use_priority:
        metric_filter.update({name: True for name in (
            'replay/is_weight_mean', 'replay/is_weight_std', 'replay/is_weight_min',
            'replay/is_weight_max', 'replay/is_weight_ess_fraction',
        )})
    if reanalysis_enabled:
        metric_filter.update({name: True for name in (
            'reanalyze/count', 'reanalyze/freq_actual', 'reanalyze/target_age_mean',
            'reanalyze/target_age_p50', 'reanalyze/target_age_p90',
            'reanalyze/target_age_max', 'reanalyze/roots_refreshed',
        )})
    if bootstrap_value_context:
        metric_filter['segment/bootstrap_context_len'] = True
    if grad_clip_mode == 'separate_encoder':
        metric_filter.update({name: True for name in (
            'grad/clip_encoder_scale', 'grad/clip_non_encoder_scale',
            'grad/encoder_pre_clip_norm', 'grad/non_encoder_pre_clip_norm',
        )})
    if adaptive_entropy_enabled:
        metric_filter.update({name: True for name in (
            'entropy/adaptive_alpha', 'entropy/target_ratio', 'entropy/alpha_loss',
        )})
    if encoder_clip_enabled:
        metric_filter.update({name: True for name in (
            'current_encoder_clip_value', 'stability/current_encoder_clip_value',
            'encoder_clip/enabled', 'encoder_clip/applied', 'encoder_clip/apply_count',
            'encoder_clip/scale_factor', 'encoder_clip/max_latent_norm',
            'encoder_clip/threshold',
        )})
    if open_loop_consistency_weight > 0:
        metric_filter['open_loop_consistency_loss'] = True
    if open_loop_recurrent_weight > 0:
        metric_filter.update({name: True for name in (
            'open_loop_recurrent_loss', 'open_loop_recurrent_latent_loss',
            'open_loop_recurrent_reward_loss', 'open_loop_recurrent_value_loss',
            'open_loop_recurrent_policy_loss',
        )})
    if open_loop_diagnostic_freq > 0:
        metric_filter.update({f'analysis/{name}': True for name in (
            'open_loop_latent_mse_mean', 'open_loop_latent_mse_first',
            'open_loop_latent_mse_middle', 'open_loop_latent_mse_last',
            'rolling_teacher_latent_mse_mean', 'rolling_teacher_latent_mse_first',
            'rolling_teacher_latent_mse_middle', 'rolling_teacher_latent_mse_last',
            'teacher_forced_latent_mse_mean', 'teacher_forced_latent_mse_first',
            'rolling_context_ratio', 'open_loop_exposure_ratio', 'open_loop_total_ratio',
        )})
    if gradient_diagnostic_freq > 0:
        groups = (
            'encoder', 'transformer', 'action_embedding', 'head_observation',
            'head_reward', 'head_value', 'head_policy', '_tracked_total_norm',
        )
        metric_filter['grad_component/last_check_iter'] = True
        metric_filter.update({
            f'grad_component/{component}/{group}': True
            for component in ('obs', 'reward', 'value', 'policy')
            for group in groups
        })
    return metric_filter


def _atari_game_name(env_id):
    return env_id.split('/')[-1].split('-')[0]


def _validate_open_loop_requirements(
        *, rebuild_kv_window_from_tokens, open_loop_diagnostic_freq,
        open_loop_consistency_weight, open_loop_recurrent_weight,
):
    """Reject open-loop features that cannot reconstruct the required token history.

    Failing during CLI/config construction avoids allocating environments and GPU
    workers only to fail on the first scheduled diagnostic or auxiliary-loss batch.
    """
    requested = []
    if open_loop_diagnostic_freq > 0:
        requested.append('open-loop diagnostics')
    if open_loop_consistency_weight > 0:
        requested.append('open-loop consistency loss')
    if open_loop_recurrent_weight > 0:
        requested.append('open-loop recurrent loss')
    if requested and not rebuild_kv_window_from_tokens:
        raise ValueError(
            f'{", ".join(requested)} require --rebuild-kv-window-from-tokens; '
            'the legacy KV path cannot reconstruct aligned open-loop prefixes.'
        )


def _validate_reanalysis_context(*, buffer_reanalyze_freq_override, contextual_reanalysis):
    """Keep reanalysis frequency and target-context topology explicit.

    ``None`` means "use the config default".  Any numeric override is an
    explicit request for replay refresh and must opt into the information-state
    aligned implementation.  Rejecting the ambiguous combination is safer than
    silently mutating a baseline (or silently using the legacy, misaligned
    target path).
    """
    if buffer_reanalyze_freq_override is not None and not contextual_reanalysis:
        raise ValueError(
            'Explicit buffer_reanalyze_freq requires --contextual-reanalysis. '
            'Omit --buffer-reanalyze-freq for the no-reanalysis baseline; '
            'do not pass the historical 2e-10 sentinel explicitly.'
        )


def _default_run_name(
        game_name, seed, timestamp, *,
        num_unroll_steps, infer_context_length, game_segment_length, batch_size,
        replay_ratio, collect_temperature, obs_loss_weight, value_loss_weight,
        open_loop_consistency_weight, use_priority, use_augmentation,
        bootstrap_value_context, rebuild_kv_window_from_tokens, contextual_reanalysis,
        buffer_reanalyze_freq, stab_fix, grad_clip_mode, max_env_step,
        collector_episode_life=True, collector_clip_rewards=True,
        target_update_theta=0.005,
        cosine_lr_scheduler=False,
        frame_stack_num=1,
        eval_temperature=0.0,
        no_decay_bias_norm=True,
):
    """Build a self-describing run name from the resolved key config settings."""
    parts = [
        'uz',
        f'h{num_unroll_steps}',
        f'ctx{infer_context_length}',
        f'gsl{game_segment_length}',
        f'bs{batch_size}',
        f'rr{replay_ratio:g}',
        f'temp{collect_temperature:g}',
        f'obs{obs_loss_weight:g}',
        f'value{value_loss_weight:g}',
        f'ema{target_update_theta:g}',
        'coslr' if cosine_lr_scheduler else 'constlr',
        # The LN/bias no-decay grouping is the default; only the legacy
        # decay-everything optimizer is encoded in the run name.
        None if no_decay_bias_norm else 'wdall',
        # Only non-default stacking is encoded so legacy stack-1 run names stay
        # directly comparable with historical batches.
        f'stack{frame_stack_num}' if frame_stack_num != 1 else None,
        f'evaltemp{eval_temperature:g}' if eval_temperature > 0 else None,
        'stabfix' if stab_fix else 'nostabfix',
        f'clip-{grad_clip_mode}',
        'rebuildkv' if rebuild_kv_window_from_tokens else 'norebuildkv',
        'ctxreanalyze' if contextual_reanalysis else 'noctxreanalyze',
        f'reanalyze{buffer_reanalyze_freq:g}',
        'bootctx' if bootstrap_value_context else 'nobootctx',
        f'olc{open_loop_consistency_weight:g}',
        'per' if use_priority else 'noper',
        'aug' if use_augmentation else 'noaug',
        'lifeboundary' if collector_episode_life else 'fullgame',
        'cliprew' if collector_clip_rewards else 'rawrew',
    ]
    parts = [part for part in parts if part is not None]
    parts.append(f'seed{seed}')
    parts.append(f'{max_env_step / 1e6:g}m')
    parts.append(timestamp)
    return f'{game_name.lower()}/{"_".join(parts)}'


def _prepare_run_directory(run_dir, resume_from=None, resume_in_place=False):
    """Create a run directory, or explicitly reopen it for checkpoint recovery."""
    if resume_in_place and resume_from is None:
        raise ValueError('resume_in_place requires resume_from')
    if os.path.exists(run_dir):
        if not resume_in_place:
            raise FileExistsError(
                f'Run directory already exists: {os.path.abspath(run_dir)}. '
                'Use a new --run-name for a fresh run, or provide both --resume-from and '
                '--resume-in-place to continue an existing run.'
            )
        if not os.path.isdir(run_dir):
            raise NotADirectoryError(f'Run path is not a directory: {os.path.abspath(run_dir)}')
        return
    os.makedirs(run_dir)


def _resolve_collect_temperature(value):
    collect_temperature = 0.25 if value is None else float(value)
    if collect_temperature <= 0:
        raise ValueError(
            f'collect_temperature must be positive, got {collect_temperature}'
        )
    return collect_temperature


def _resolve_target_update_theta(value):
    """Validate the target EMA coefficient before constructing GPU workers."""
    theta = 0.005 if value is None else float(value)
    if not 0.0 < theta <= 1.0:
        raise ValueError(f'target_update_theta must be in (0, 1], got {theta}')
    return theta


def _resolve_frame_stack_num(value):
    """Validate the frame-stack override before deriving env/model shapes.

    Stack 1 keeps the historical RGB (3, 64, 64) recipe.  Stack 4 switches to
    the MuZero-style grayscale stacking path: the env warps single-channel
    frames and the collector stacks ``frame_stack_num`` of them into a
    (4, 64, 64) model input, so velocity is observable even with zero token
    context.
    """
    frame_stack_num = 1 if value is None else int(value)
    if frame_stack_num not in (1, 4):
        raise ValueError(f'frame_stack_num must be 1 or 4, got {frame_stack_num}')
    return frame_stack_num


def _resolve_eval_temperature(value):
    """0 keeps the legacy deterministic argmax evaluation; >0 samples actions."""
    eval_temperature = 0.0 if value is None else float(value)
    if eval_temperature < 0:
        raise ValueError(f'eval_temperature must be non-negative, got {eval_temperature}')
    return eval_temperature


def _stacked_observation_spec(frame_stack_num):
    """Return (observation_shape, gray_scale, image_channel) for the stack mode."""
    if frame_stack_num == 1:
        return (3, 64, 64), False, 3
    return (frame_stack_num, 64, 64), True, 1


def _resolve_inference_env_num(collector_env_num, evaluator_env_num, isolate_eval_cache):
    """Size root-cache namespaces without changing legacy shared-cache experiments."""
    return collector_env_num + evaluator_env_num if isolate_eval_cache else collector_env_num


def _experimental_config_overrides(
        infer_context_length=None,
        exact_kv_window_reset=False,
        rebuild_kv_window_from_tokens=False,
        contextual_reanalysis=False,
        bootstrap_value_context=False,
        open_loop_consistency_weight=None,
        open_loop_recurrent_weight=None,
        open_loop_consistency_batch_size=None,
        open_loop_consistency_horizon=None,
        open_loop_prefix_transitions=None,
        encoder_clip_enabled=None,
):
    """Build only explicitly requested UniZero experimental config overrides.

    The stable disabled defaults live in ``UniZeroPolicy.config``.  Keeping this helper sparse
    prevents Atari launch configs from copying policy defaults and makes every behavioral change
    visible at the command line.
    """
    policy_overrides = {}
    world_model_overrides = {}

    if infer_context_length is not None:
        infer_context_length = int(infer_context_length)
        if infer_context_length < 1:
            raise ValueError(
                f'infer_context_length must be positive, got {infer_context_length}'
            )
        world_model_overrides['context_length'] = 2 * infer_context_length

    if exact_kv_window_reset:
        world_model_overrides['exact_kv_window_reset'] = True
    if rebuild_kv_window_from_tokens:
        world_model_overrides['rebuild_kv_window_from_tokens'] = True
    if contextual_reanalysis:
        policy_overrides['contextual_reanalysis'] = True
    if bootstrap_value_context:
        policy_overrides['bootstrap_value_context'] = True

    consistency_weight = (
        0. if open_loop_consistency_weight is None
        else float(open_loop_consistency_weight)
    )
    recurrent_weight = (
        0. if open_loop_recurrent_weight is None
        else float(open_loop_recurrent_weight)
    )
    if consistency_weight < 0:
        raise ValueError(
            f'open_loop_consistency_weight must be non-negative, got {consistency_weight}'
        )
    if recurrent_weight < 0:
        raise ValueError(
            f'open_loop_recurrent_weight must be non-negative, got {recurrent_weight}'
        )
    if consistency_weight > 0 and recurrent_weight > 0:
        raise ValueError(
            'open_loop_consistency_weight and open_loop_recurrent_weight are mutually exclusive'
        )
    if open_loop_consistency_weight is not None:
        world_model_overrides['open_loop_consistency_loss_weight'] = consistency_weight
    if open_loop_recurrent_weight is not None:
        world_model_overrides['open_loop_recurrent_loss_weight'] = recurrent_weight

    if open_loop_consistency_batch_size is not None:
        batch_size = int(open_loop_consistency_batch_size)
        if batch_size <= 0:
            raise ValueError('open_loop_consistency_batch_size must be positive')
        world_model_overrides['open_loop_consistency_batch_size'] = batch_size
    if open_loop_consistency_horizon is not None:
        horizon = int(open_loop_consistency_horizon)
        if horizon <= 0:
            raise ValueError('open_loop_consistency_horizon must be positive')
        world_model_overrides['open_loop_consistency_horizon'] = horizon
    if open_loop_prefix_transitions is not None:
        prefix_transitions = int(open_loop_prefix_transitions)
        if prefix_transitions < 0:
            raise ValueError('open_loop_prefix_transitions must be non-negative')
        world_model_overrides['open_loop_prefix_transitions'] = prefix_transitions

    if encoder_clip_enabled is not None:
        encoder_clip_enabled = bool(encoder_clip_enabled)
        policy_overrides.update(
            use_encoder_clip_annealing=encoder_clip_enabled,
            latent_norm_clip_threshold=10.0 if encoder_clip_enabled else 0.0,
        )

    return policy_overrides, world_model_overrides


def main(
        env_id,
        seed,
        output_root='data_unizero_segment',
        run_name=None,
        use_new_cache_manager=False,
        use_augmentation_override=None,
        replay_ratio_override=None,
        batch_size_override=None,
        num_unroll_steps_override=None,
        obs_loss_weight_override=None,
        value_loss_weight_override=None,
        target_update_theta_override=None,
        cosine_lr_scheduler_override=None,
        no_decay_bias_norm_override=None,
        latent_norm_reg_weight_override=None,
        frame_stack_num_override=None,
        eval_temperature_override=None,
        root_cache_key_round_decimals_override=None,
        kv_cache_clear_interval_override=None,
        empty_cuda_cache_on_cache_reset_override=None,
        isolate_eval_cache=False,
        evaluator_env_num_override=None,
        collect_num_simulations_override=None,
        collect_temperature_override=None,
        grad_clip_value_override=None,
        grad_clip_mode_override=None,
        replay_buffer_size_override=None,
        gradient_diagnostic_freq_override=None,
        disable_adaptive_alpha=True,
        fixed_alpha=5e-3,
        disable_policy_label_smoothing=True,
        encoder_clip_enabled=None,
        resume_from=None,
        resume_in_place=False,
        max_env_step_override=None,
        use_priority=None,
        stab_fix=True,
        game_segment_length_override=None,
        infer_context_length_override=None,
        exact_kv_window_reset=False,
        rebuild_kv_window_from_tokens=False,
        contextual_reanalysis=False,
        bootstrap_value_context=False,
        resume_buffer_min_transitions_override=None,
        buffer_reanalyze_freq_override=None,
        reanalyze_batch_size_override=None,
        save_ckpt_after_iter_override=None,
        periodic_ckpt_keep_last_override=None,
        ignore_checkpoint_save_errors=False,
        save_ckpt_in_eval=True,
        open_loop_diagnostic_freq_override=None,
        open_loop_consistency_weight_override=None,
        open_loop_recurrent_weight_override=None,
        open_loop_consistency_batch_size_override=None,
        open_loop_consistency_horizon_override=None,
        open_loop_prefix_transitions_override=None,
        legacy_resume_alpha=None,
        log_metric=True,
        tb_log_all=False,
        collector_episode_life=True,
        collector_clip_rewards=True,
):
    action_space_size = atari_env_action_space_map[env_id]
    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    num_segments = 8
    # Eight deterministic episodes reduce the duplicate-trajectory noise seen
    # with the historical three-episode evaluator while remaining inexpensive.
    evaluator_env_num = (
        8 if evaluator_env_num_override is None else int(evaluator_env_num_override)
    )
    eval_freq = int(1e4)
    if evaluator_env_num <= 0:
        raise ValueError(f'evaluator_env_num must be positive, got {evaluator_env_num}')
    collect_num_simulations = (
        25
        if collect_num_simulations_override is None
        else int(collect_num_simulations_override)
    )
    if collect_num_simulations <= 0:
        raise ValueError(
            f'collect_num_simulations must be positive, got {collect_num_simulations}'
        )
    collect_temperature = _resolve_collect_temperature(collect_temperature_override)
    use_augmentation = (
        False if use_augmentation_override is None else bool(use_augmentation_override)
    )
    resolved_use_priority = False if use_priority is None else bool(use_priority)
    replay_ratio = 0.1 if replay_ratio_override is None else float(replay_ratio_override)
    if replay_ratio <= 0:
        raise ValueError(f'replay_ratio must be positive, got {replay_ratio}')
    batch_size = 256 if batch_size_override is None else int(batch_size_override)
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    num_unroll_steps = 10 if num_unroll_steps_override is None else int(num_unroll_steps_override)
    if num_unroll_steps <= 0:
        raise ValueError(f'num_unroll_steps must be positive, got {num_unroll_steps}')
    if infer_context_length_override is None:
        # The best MsPacman recipe retains 5 observation/action blocks for online KV inference.
        infer_context_length_override = min(5, num_unroll_steps)
    if int(infer_context_length_override) > num_unroll_steps:
        raise ValueError(
            'infer_context_length cannot exceed num_unroll_steps because the '
            f'transformer cache has only that many blocks: {infer_context_length_override} > {num_unroll_steps}'
        )
    # The latest representation-stability matrix favors obs=1; keep it explicit
    # and independently overridable for controlled ablations.
    obs_loss_weight = 1.0 if obs_loss_weight_override is None else float(obs_loss_weight_override)
    # The historical config declared 0.25, but the historical code hard-coded an effective 0.5.
    # The v3 arm of the 2026-08 MsPacman matrix isolated this drift and won, so 0.5 is default.
    value_loss_weight = 0.5 if value_loss_weight_override is None else float(value_loss_weight_override)
    if obs_loss_weight < 0 or value_loss_weight < 0:
        raise ValueError('loss weights must be non-negative')
    target_update_theta = _resolve_target_update_theta(target_update_theta_override)
    cosine_lr_scheduler = (
        False if cosine_lr_scheduler_override is None
        else bool(cosine_lr_scheduler_override)
    )
    no_decay_bias_norm = (
        True if no_decay_bias_norm_override is None
        else bool(no_decay_bias_norm_override)
    )
    latent_norm_reg_weight = (
        0.0 if latent_norm_reg_weight_override is None
        else float(latent_norm_reg_weight_override)
    )
    if latent_norm_reg_weight < 0:
        raise ValueError('latent_norm_reg_weight must be non-negative')
    frame_stack_num = _resolve_frame_stack_num(frame_stack_num_override)
    observation_shape, gray_scale, image_channel = _stacked_observation_spec(frame_stack_num)
    eval_temperature = _resolve_eval_temperature(eval_temperature_override)
    root_cache_key_round_decimals = (
        0 if root_cache_key_round_decimals_override is None
        else int(root_cache_key_round_decimals_override)
    )
    if root_cache_key_round_decimals < 0 or root_cache_key_round_decimals > 7:
        raise ValueError('root cache key decimals must be between 0 and 7')
    kv_cache_clear_interval = (
        2000 if kv_cache_clear_interval_override is None
        else int(kv_cache_clear_interval_override)
    )
    if kv_cache_clear_interval < 0:
        raise ValueError('kv_cache_clear_interval must be non-negative')
    empty_cuda_cache_on_cache_reset = (
        True if empty_cuda_cache_on_cache_reset_override is None
        else bool(empty_cuda_cache_on_cache_reset_override)
    )
    grad_clip_value = 5.0 if grad_clip_value_override is None else float(grad_clip_value_override)
    if grad_clip_value <= 0:
        raise ValueError(f'grad_clip_value must be positive, got {grad_clip_value}')
    # Isolate encoder clipping by default even without augmentation.  The global
    # clipping baseline remains available through --grad-clip-mode global.
    resolved_grad_clip_mode_override = (
        'separate_encoder' if grad_clip_mode_override is None else grad_clip_mode_override
    )
    grad_clip_mode = _resolve_grad_clip_mode(
        use_augmentation, resolved_grad_clip_mode_override
    )
    replay_buffer_size = (
        int(5e5) if replay_buffer_size_override is None
        else int(replay_buffer_size_override)
    )
    if replay_buffer_size <= 0:
        raise ValueError(f'replay_buffer_size must be positive, got {replay_buffer_size}')
    gradient_diagnostic_freq = (
        0 if gradient_diagnostic_freq_override is None
        else int(gradient_diagnostic_freq_override)
    )
    if gradient_diagnostic_freq < 0:
        raise ValueError(
            f'gradient_diagnostic_freq must be non-negative, got {gradient_diagnostic_freq}'
        )

    td_steps = 5
    # With Htrain=10 and td_steps=5, game_segment_length=20 leaves only five complete
    # non-terminal roots. GSL200 avoids that structural replay waste for both H5 and H10 variants.
    game_segment_length = 200 if game_segment_length_override is None else int(game_segment_length_override)
    if game_segment_length <= num_unroll_steps + td_steps:
        raise ValueError(
            'game_segment_length must exceed num_unroll_steps + td_steps, got '
            f'{game_segment_length} <= {num_unroll_steps + td_steps}'
        )
    save_ckpt_after_iter = (
        50000 if save_ckpt_after_iter_override is None else int(save_ckpt_after_iter_override)
    )
    if save_ckpt_after_iter <= 0:
        raise ValueError(f'save_ckpt_after_iter must be positive, got {save_ckpt_after_iter}')
    periodic_ckpt_keep_last = (
        0 if periodic_ckpt_keep_last_override is None else int(periodic_ckpt_keep_last_override)
    )
    if periodic_ckpt_keep_last < 0:
        raise ValueError(
            f'periodic_ckpt_keep_last must be non-negative, got {periodic_ckpt_keep_last}'
        )
    resume_buffer_min_transitions = (
        100000 if resume_buffer_min_transitions_override is None
        else int(resume_buffer_min_transitions_override)
    )
    if resume_buffer_min_transitions < 0:
        raise ValueError(
            'resume_buffer_min_transitions must be non-negative, got '
            f'{resume_buffer_min_transitions}'
        )
    open_loop_diagnostic_freq = (
        0 if open_loop_diagnostic_freq_override is None
        else int(open_loop_diagnostic_freq_override)
    )
    if open_loop_diagnostic_freq < 0:
        raise ValueError(
            f'open_loop_diagnostic_freq must be non-negative, got {open_loop_diagnostic_freq}'
        )
    # The open-loop consistency loss is off by default. Enabling it via
    # --open-loop-consistency-weight uses the v3 recipe geometry below.
    if open_loop_consistency_batch_size_override is None:
        open_loop_consistency_batch_size_override = 8
    if open_loop_consistency_horizon_override is None:
        open_loop_consistency_horizon_override = 4
    if open_loop_prefix_transitions_override is None:
        open_loop_prefix_transitions_override = 3
    if legacy_resume_alpha is not None and legacy_resume_alpha <= 0:
        raise ValueError(f'legacy_resume_alpha must be positive, got {legacy_resume_alpha}')
    _validate_reanalysis_context(
        buffer_reanalyze_freq_override=buffer_reanalyze_freq_override,
        contextual_reanalysis=contextual_reanalysis,
    )

    policy_experiment_overrides, world_model_experiment_overrides = (
        _experimental_config_overrides(
            infer_context_length=infer_context_length_override,
            exact_kv_window_reset=exact_kv_window_reset,
            rebuild_kv_window_from_tokens=rebuild_kv_window_from_tokens,
            contextual_reanalysis=contextual_reanalysis,
            bootstrap_value_context=bootstrap_value_context,
            open_loop_consistency_weight=open_loop_consistency_weight_override,
            open_loop_recurrent_weight=open_loop_recurrent_weight_override,
            open_loop_consistency_batch_size=open_loop_consistency_batch_size_override,
            open_loop_consistency_horizon=open_loop_consistency_horizon_override,
            open_loop_prefix_transitions=open_loop_prefix_transitions_override,
            encoder_clip_enabled=encoder_clip_enabled,
        )
    )
    resolved_open_loop_consistency_weight = float(
        world_model_experiment_overrides.get('open_loop_consistency_loss_weight', 0.0)
    )
    resolved_open_loop_recurrent_weight = float(
        world_model_experiment_overrides.get('open_loop_recurrent_loss_weight', 0.0)
    )
    _validate_open_loop_requirements(
        rebuild_kv_window_from_tokens=rebuild_kv_window_from_tokens,
        open_loop_diagnostic_freq=open_loop_diagnostic_freq,
        open_loop_consistency_weight=resolved_open_loop_consistency_weight,
        open_loop_recurrent_weight=resolved_open_loop_recurrent_weight,
    )

    num_simulations = 50

    num_layers = 2
    norm_type = "LN"

    if env_id == 'ALE/Pong-v5':
        max_env_step = int(1e6)
    elif env_id == 'ALE/MsPacman-v5':
        max_env_step = int(3e6)
    else:
        max_env_step = int(10e6)
    if max_env_step_override is not None:
        max_env_step = int(max_env_step_override)

    # Reanalyze settings
    buffer_reanalyze_freq = (
        1 / 5000000000
        if buffer_reanalyze_freq_override is None
        else float(buffer_reanalyze_freq_override)
    )
    if buffer_reanalyze_freq <= 0:
        raise ValueError(
            f'buffer_reanalyze_freq must be positive, got {buffer_reanalyze_freq}'
        )
    reanalyze_batch_size = (
        160 if reanalyze_batch_size_override is None
        else int(reanalyze_batch_size_override)
    )
    if reanalyze_batch_size <= 0:
        raise ValueError(
            f'reanalyze_batch_size must be positive, got {reanalyze_batch_size}'
        )
    reanalyze_partition = 0.75
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================

    atari_unizero_config = dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=observation_shape,
            gray_scale=gray_scale,
            frame_stack_num=frame_stack_num,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            collector_episode_life=bool(collector_episode_life),
            collector_clip_rewards=bool(collector_clip_rewards),
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            # Grayscale is a policy-level flag consumed by the replay game
            # segments (jpeg decode shape); it must track the env stack mode.
            gray_scale=gray_scale,
            model=dict(
                observation_shape=observation_shape,
                image_channel=image_channel,
                frame_stack_num=frame_stack_num,
                action_space_size=action_space_size,
                reward_support_range=(-300., 301., 1.),
                value_support_range=(-300., 301., 1.),
                norm_type=norm_type,
                # num_res_blocks=2,
                # num_channels=128,
                num_res_blocks=1,
                num_channels=64,
                world_model_cfg=dict(
                    latent_recon_loss_weight=0.0,
                    perceptual_loss_weight=0.0,
                    latent_norm_reg_weight=latent_norm_reg_weight,
                    norm_type=norm_type,
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse',
                    support_size=601,
                    # Used as the fixed entropy coefficient when adaptive alpha is disabled.
                    policy_entropy_weight=fixed_alpha,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=8,
                    embed_dim=768,
                    # Reserve disjoint root-cache namespaces for collector
                    # envs [0, collector_env_num) and evaluator envs after it.
                    env_num=_resolve_inference_env_num(
                        collector_env_num, evaluator_env_num, isolate_eval_cache
                    ),
                    num_simulations=num_simulations,
                    game_segment_length=game_segment_length,
                    # Mirror learner loss weights so auxiliary recurrent
                    # supervision cannot silently override the experiment.
                    obs_loss_weight=obs_loss_weight,
                    reward_loss_weight=1.0,
                    value_loss_weight=value_loss_weight,
                    policy_loss_weight=1.0,
                    device='cuda',
                    use_priority=True,
                    encoder_type='resnet',
                    use_normal_head=True,
                    optim_type='AdamW_mix_lr_wdecay',
                    use_new_cache_manager=use_new_cache_manager,
                    root_cache_key_round_decimals=root_cache_key_round_decimals,
                    open_loop_diagnostic_freq=open_loop_diagnostic_freq,
                    open_loop_diagnostic_batch_size=collector_env_num,
                    # Policy-stability protections (500K crash chain: extreme target_policy ->
                    # logits explosion -> x_token collapse). Off by default; --stab-fix enables.
                    use_policy_logits_clip=stab_fix,
                    policy_logits_clip_method='soft_tanh',
                    policy_logits_clip_min=-10.0,
                    policy_logits_clip_max=10.0,
                ),
            ),
            # Learning settings
            optim_type='AdamW_mix_lr_wdecay',
            learning_rate=0.0001,
            weight_decay=1e-2,
            no_decay_bias_norm=no_decay_bias_norm,
            target_update_theta=target_update_theta,
            cos_lr_scheduler=cosine_lr_scheduler,
            batch_size=batch_size,
            replay_ratio=replay_ratio,
            num_unroll_steps=num_unroll_steps,
            num_segments=num_segments,
            game_segment_length=game_segment_length,
            td_steps=td_steps,
            # Full learner checkpoints are ~530MB each; save every 50k train iters instead of
            # the default 10k to bound disk usage. ckpt_best (on new best eval) is unaffected.
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=save_ckpt_after_iter))),
            periodic_ckpt_keep_last=periodic_ckpt_keep_last,
            # Curve-first cluster runs may continue through transient/full-filesystem checkpoint
            # failures. Other training, evaluation, and TensorBoard errors remain fatal.
            ignore_checkpoint_save_errors=bool(ignore_checkpoint_save_errors),
            save_ckpt_in_eval=bool(save_ckpt_in_eval),
            # KV caches are cleared once per env per this many env steps. Was hardcoded to
            # game_segment_length, which wiped all MCTS kv caches after every single segment.
            kv_cache_clear_interval=kv_cache_clear_interval,
            empty_cuda_cache_on_cache_reset=empty_cuda_cache_on_cache_reset,
            num_simulations=num_simulations,
            fixed_temperature_value=collect_temperature,
            eval_temperature=eval_temperature,
            obs_loss_weight=obs_loss_weight,
            value_loss_weight=value_loss_weight,
            grad_clip_value=grad_clip_value,
            grad_clip_mode=grad_clip_mode,
            use_augmentation=use_augmentation,

            # Adaptive target entropy settings from the 2025 Pong run.
            use_adaptive_entropy_weight=not disable_adaptive_alpha,
            adaptive_entropy_alpha_lr=1e-4,
            legacy_resume_adaptive_alpha=legacy_resume_alpha,
            target_entropy_start_ratio=0.98,
            target_entropy_end_ratio=0.7,
            target_entropy_decay_steps=100000,

            # Policy smoothing decays 0.05->0.01; value/reward use 0.1.
            policy_ls_eps_start=0.0 if disable_policy_label_smoothing else 0.05,
            policy_ls_eps_end=0.0 if disable_policy_label_smoothing else 0.01,
            policy_ls_eps_decay_steps=50000,
            label_smoothing_eps=0.1,
            # Numerical stability and target regularization are separate interventions. Permanent
            # 0.05 smoothing biases a sharp improved policy even after the instability is gone.
            use_continuous_label_smoothing=False,
            continuous_ls_eps=0.05,
            monitor_norm_freq=10000,
            gradient_diagnostic_freq=gradient_diagnostic_freq,
            # Detailed logits statistics are debug-only; core losses, priorities, and health
            # signals remain available through the compact whitelist below.
            use_enhanced_policy_monitoring=bool(tb_log_all),
            log_metric=bool(log_metric),
            tb_log_all=bool(tb_log_all),
            tb_metric_filter=_build_learn_tb_metric_filter(
                use_priority=resolved_use_priority,
                reanalysis_enabled=buffer_reanalyze_freq_override is not None,
                bootstrap_value_context=bool(bootstrap_value_context),
                grad_clip_mode=grad_clip_mode,
                adaptive_entropy_enabled=not disable_adaptive_alpha,
                encoder_clip_enabled=bool(encoder_clip_enabled),
                open_loop_diagnostic_freq=open_loop_diagnostic_freq,
                gradient_diagnostic_freq=gradient_diagnostic_freq,
                open_loop_consistency_weight=resolved_open_loop_consistency_weight,
                open_loop_recurrent_weight=resolved_open_loop_recurrent_weight,
            ),

            # Priority settings.
            # Default OFF: uniform replay won the 2026-08 MsPacman 3M matrix (v3 arm).
            # NOTE: model.world_model_cfg.use_priority intentionally remains True so the model
            # always returns a shape-[B] value_priority diagnostic. Setting policy.use_priority
            # False here still gives uniform replay/IS weights; the buffer discards priority
            # write-backs. This is deliberate model-vs-buffer asymmetry, not a synchronized flag.
            use_priority=resolved_use_priority,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,

            # Reanalyze settings
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            # A refresh expands each sequence into H+1 roots. Keep every MCTS group within the
            # same recurrent-KV capacity used by the online collector (8 envs x 50 simulations).
            reanalyze_search_chunk_size=collector_env_num,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,

            # Environment settings
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            isolate_eval_cache=bool(isolate_eval_cache),
            eval_freq=eval_freq,
            replay_buffer_size=replay_buffer_size,
            # Policy checkpoints omit the ~25GB full Atari replay buffer. Refill a small diverse
            # on-policy window before updating a mature model after preemption.
            resume_buffer_min_transitions=resume_buffer_min_transitions,
        ),
    )
    atari_unizero_config['policy'].update(policy_experiment_overrides)
    atari_unizero_config['policy']['model']['world_model_cfg'].update(
        world_model_experiment_overrides
    )
    atari_unizero_config['policy']['collect_num_simulations'] = collect_num_simulations
    atari_unizero_config = EasyDict(atari_unizero_config)
    main_config = atari_unizero_config

    atari_unizero_create_config = dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='unizero',
            import_names=['lzero.policy.unizero'],
        ),
    )
    atari_unizero_create_config = EasyDict(atari_unizero_create_config)
    create_config = atari_unizero_create_config

    # ============ use muzero_segment_collector instead of muzero_collector =============
    from lzero.entry import train_unizero_segment
    game_name = _atari_game_name(env_id)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if run_name is None:
        run_name = _default_run_name(
            game_name, seed, timestamp,
            num_unroll_steps=num_unroll_steps,
            infer_context_length=infer_context_length_override,
            game_segment_length=game_segment_length,
            batch_size=batch_size,
            replay_ratio=replay_ratio,
            collect_temperature=collect_temperature,
            obs_loss_weight=obs_loss_weight,
            value_loss_weight=value_loss_weight,
            target_update_theta=target_update_theta,
            cosine_lr_scheduler=cosine_lr_scheduler,
            no_decay_bias_norm=no_decay_bias_norm,
            frame_stack_num=frame_stack_num,
            eval_temperature=eval_temperature,
            open_loop_consistency_weight=open_loop_consistency_weight_override or 0,
            use_priority=resolved_use_priority,
            use_augmentation=use_augmentation,
            bootstrap_value_context=bootstrap_value_context,
            rebuild_kv_window_from_tokens=rebuild_kv_window_from_tokens,
            contextual_reanalysis=contextual_reanalysis,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            stab_fix=stab_fix,
            grad_clip_mode=grad_clip_mode,
            max_env_step=max_env_step,
            collector_episode_life=collector_episode_life,
            collector_clip_rewards=collector_clip_rewards,
        )
    run_name = _safe_run_name(run_name)

    # LightZero internally prefixes exp_name with "./", so keep it relative to
    # the current working directory even when an absolute output root is given.
    run_dir = os.path.relpath(os.path.abspath(os.path.join(output_root, run_name)), os.getcwd())
    _prepare_run_directory(run_dir, resume_from=resume_from, resume_in_place=resume_in_place)
    main_config.exp_name = run_dir
    with open(os.path.join(run_dir, 'pid'), 'w', encoding='utf-8') as file:
        file.write(f'{os.getpid()}\n')

    original_stdout, original_stderr = sys.stdout, sys.stderr
    console_path = os.path.join(run_dir, 'console.log')
    with open(console_path, 'a', encoding='utf-8', buffering=1) as console:
        sys.stdout = _Tee(original_stdout, console)
        sys.stderr = _Tee(original_stderr, console)
        try:
            print(f'Run directory: {os.path.abspath(run_dir)}')
            train_unizero_segment(
                [main_config, create_config], seed=seed, model_path=resume_from, max_env_step=max_env_step
            )
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Run configurable UniZero Atari segment experiments.'
    )
    parser.add_argument('--env', type=str, help='The environment to use', default='ALE/MsPacman-v5')
    parser.add_argument('--seed', type=int, help='The seed to use', default=0)
    parser.add_argument(
        '--output-root', type=str, default='data_unizero_segment',
        help='Root directory containing one self-contained folder per run.'
    )
    parser.add_argument(
        '--run-name', type=str, default=None,
        help='Optional unique run folder name; defaults to a self-describing name built from '
             'the key config settings plus seed/budget/timestamp.'
    )
    parser.add_argument(
        '--use-new-cache-manager', action='store_true',
        help='Enable the new UniZero KV cache manager; disabled by default for baseline compatibility.'
    )
    augmentation_group = parser.add_mutually_exclusive_group()
    augmentation_group.add_argument(
        '--use-augmentation', dest='use_augmentation', action='store_true',
        help='Enable Atari shift/intensity augmentation.'
    )
    augmentation_group.add_argument(
        '--no-augmentation', dest='use_augmentation', action='store_false',
        help='Disable Atari augmentation (default).'
    )
    adaptive_alpha_group = parser.add_mutually_exclusive_group()
    adaptive_alpha_group.add_argument(
        '--use-adaptive-alpha', dest='disable_adaptive_alpha', action='store_false',
        help='Enable adaptive entropy alpha instead of the fixed coefficient.'
    )
    adaptive_alpha_group.add_argument(
        '--disable-adaptive-alpha', dest='disable_adaptive_alpha', action='store_true',
        help='Disable adaptive entropy alpha and use --fixed-alpha instead (default).'
    )
    parser.add_argument(
        '--fixed-alpha', type=float, default=5e-3,
        help='Fixed policy entropy coefficient used when adaptive alpha is disabled.'
    )
    encoder_clip_group = parser.add_mutually_exclusive_group()
    encoder_clip_group.add_argument(
        '--enable-encoder-clip', dest='encoder_clip_enabled', action='store_true',
        help='Opt in to the encoder latent-norm projection with the policy default 30->10 schedule.'
    )
    encoder_clip_group.add_argument(
        '--disable-encoder-clip', dest='encoder_clip_enabled', action='store_false',
        help='Explicitly disable encoder latent-norm projection (also the policy default).'
    )
    policy_smoothing_group = parser.add_mutually_exclusive_group()
    policy_smoothing_group.add_argument(
        '--enable-policy-label-smoothing', dest='disable_policy_label_smoothing', action='store_false',
        help='Enable the 0.05 to 0.01 policy label-smoothing schedule.'
    )
    policy_smoothing_group.add_argument(
        '--disable-policy-label-smoothing', dest='disable_policy_label_smoothing', action='store_true',
        help='Set policy label-smoothing epsilon to zero (default); value/reward smoothing is unchanged.'
    )
    parser.set_defaults(
        disable_adaptive_alpha=True,
        disable_policy_label_smoothing=True,
        encoder_clip_enabled=None,
        use_augmentation=None,
    )
    parser.add_argument(
        '--resume-from', dest='resume_from', type=str, default=None,
        help='Optional learner checkpoint path to resume weights/optimizer/train_iter/envstep from.'
    )
    parser.add_argument(
        '--resume-in-place', action='store_true',
        help='Reopen an existing run directory when resuming from a checkpoint. This is intended '
             'for infrastructure-level automatic restarts.'
    )
    parser.add_argument(
        '--max-env-step', dest='max_env_step', type=int, default=None,
        help='Override the default max env-step budget (e.g. for continuing a run past its cap).'
    )
    parser.add_argument(
        '--evaluator-env-num', dest='evaluator_env_num', type=int, default=None,
        help='Number of deterministic evaluation environments/episodes (default 8).'
    )
    parser.add_argument(
        '--collect-num-simulations', dest='collect_num_simulations', type=int, default=None,
        help='Override MCTS simulations per action during collection (default 25).'
    )
    parser.add_argument(
        '--collect-temperature', dest='collect_temperature', type=float, default=None,
        help='Override the fixed MCTS visit-count temperature used during collection (default 0.25).'
    )
    collector_semantics = parser.add_argument_group('collector Atari semantics')
    collector_semantics.add_argument(
        '--collector-episode-life', dest='collector_episode_life', action='store_true',
        help='Treat each lost life as a training episode boundary (default).'
    )
    collector_semantics.add_argument(
        '--no-collector-episode-life', dest='collector_episode_life', action='store_false',
        help='Keep full games intact in collector trajectories; useful for value-target ablations.'
    )
    collector_semantics.add_argument(
        '--collector-clip-rewards', dest='collector_clip_rewards', action='store_true',
        help='Clip collector rewards to sign(reward) (default).'
    )
    collector_semantics.add_argument(
        '--no-collector-clip-rewards', dest='collector_clip_rewards', action='store_false',
        help='Keep raw frame-skip-aggregated rewards in collector trajectories.'
    )
    parser.set_defaults(collector_episode_life=True, collector_clip_rewards=True)
    parser.add_argument(
        '--grad-clip-value', dest='grad_clip_value', type=float, default=None,
        help='Override the world-model gradient norm threshold (Atari default 5).'
    )
    parser.add_argument(
        '--grad-clip-mode', choices=('global', 'separate_encoder'), default=None,
        help='Clipping topology; defaults to separate_encoder (current best recipe).'
    )
    parser.add_argument(
        '--replay-buffer-size', dest='replay_buffer_size', type=int, default=None,
        help='Override replay capacity in transitions (Atari default 500000).'
    )
    parser.add_argument('--replay-ratio', type=float, default=None,
                        help='Override replay ratio (default 0.1).')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override learner batch size (default 256).')
    parser.add_argument(
        '--num-unroll-steps', type=int, default=None,
        help='Override learner unroll horizon (default 10; H5 is the faster MuZero-reference setting).'
    )
    parser.add_argument('--obs-loss-weight', type=float, default=None,
                        help='Override observation reconstruction loss weight (default 1).')
    parser.add_argument('--value-loss-weight', type=float, default=None,
                        help='Override value loss weight (default 0.5; the historical declared value was 0.25).')
    parser.add_argument('--target-update-theta', dest='target_update_theta', type=float, default=None,
                        help='Momentum target EMA coefficient (default 0.005; larger values chase online weights faster).')
    lr_scheduler_group = parser.add_mutually_exclusive_group()
    lr_scheduler_group.add_argument(
        '--cosine-lr-scheduler', dest='cosine_lr_scheduler', action='store_true',
        help='Cosine-decay learner LR from 1e-4 to final_learning_rate (4e-5) over total_iterations.'
    )
    lr_scheduler_group.add_argument(
        '--constant-lr-scheduler', dest='cosine_lr_scheduler', action='store_false',
        help='Keep the constant learner LR (default).'
    )
    parser.set_defaults(cosine_lr_scheduler=None)
    wd_group = parser.add_mutually_exclusive_group()
    wd_group.add_argument(
        '--no-decay-bias-norm', dest='no_decay_bias_norm', action='store_true',
        help='Exclude LayerNorm gains/biases and other 1-D tensors from weight decay (default; '
             'protects the LN-anchored latent scale from the weight-decay ratchet).'
    )
    wd_group.add_argument(
        '--decay-bias-norm', dest='no_decay_bias_norm', action='store_false',
        help='Apply weight decay to every parameter including LayerNorm gains and biases '
             '(legacy AdamW_mix_lr_wdecay behavior).'
    )
    parser.set_defaults(no_decay_bias_norm=None)
    parser.add_argument('--latent-norm-reg-weight', dest='latent_norm_reg_weight', type=float, default=None,
                        help='Weight of the soft anchor pulling per-token encoder latent L2 norms to '
                             'sqrt(embed_dim) (default 0 = disabled; R3 fallback for latent scale drift).')
    parser.add_argument('--frame-stack-num', dest='frame_stack_num', type=int, default=None,
                        choices=(1, 4),
                        help='Observation frame stack. 1 keeps the legacy RGB (3,64,64) recipe; '
                             '4 switches to grayscale MuZero-style stacking with a (4,64,64) model '
                             'input so velocity is observable without token context.')
    parser.add_argument('--eval-temperature', dest='eval_temperature', type=float, default=None,
                        help='Evaluation MCTS action-selection temperature. 0 (default) keeps the '
                             'legacy deterministic argmax; a positive value samples from the visit '
                             'distribution to probe deterministic-cycle artifacts.')
    parser.add_argument('--root-cache-key-round-decimals', type=int, default=None,
                        help='Quantize root latent cache keys to this decimal precision (0 disables).')
    parser.add_argument('--kv-cache-clear-interval', type=int, default=None,
                        help='Per-environment root-cache clear interval in episode steps; 0 disables.')
    cuda_cache_group = parser.add_mutually_exclusive_group()
    cuda_cache_group.add_argument(
        '--empty-cuda-cache-on-cache-reset', dest='empty_cuda_cache_on_cache_reset', action='store_true',
        help='Return cached CUDA allocations to the driver after inference-cache resets.'
    )
    cuda_cache_group.add_argument(
        '--no-empty-cuda-cache-on-cache-reset', dest='empty_cuda_cache_on_cache_reset', action='store_false',
        help='Keep the PyTorch CUDA allocator warm after cache resets for higher online-training throughput.'
    )
    parser.set_defaults(empty_cuda_cache_on_cache_reset=None)
    eval_cache_group = parser.add_mutually_exclusive_group()
    eval_cache_group.add_argument(
        '--isolate-eval-cache', dest='isolate_eval_cache', action='store_true',
        help='Use disjoint collector/evaluator root-cache namespaces.'
    )
    eval_cache_group.add_argument(
        '--shared-eval-cache', dest='isolate_eval_cache', action='store_false',
        help='Use the historical shared collector/evaluator cache namespace (default; '
             'the best MsPacman recipe shares it).'
    )
    parser.set_defaults(isolate_eval_cache=False)
    parser.add_argument(
        '--gradient-diagnostic-freq', dest='gradient_diagnostic_freq', type=int, default=None,
        help='Attribute gradient norms to each core loss and module every N learner iterations; '
             '0/default disables the detached diagnostic.'
    )
    parser.add_argument(
        '--resume-buffer-min-transitions', dest='resume_buffer_min_transitions',
        type=int, default=None,
        help='Collect at least this many replay transitions before updating a resumed mature '
             'checkpoint; default 100000. Fresh runs still use the normal one-batch warmup.'
    )
    priority_group = parser.add_mutually_exclusive_group()
    priority_group.add_argument(
        '--use-priority', dest='use_priority', action='store_true',
        help='Force prioritized replay on.'
    )
    priority_group.add_argument(
        '--no-priority', dest='use_priority', action='store_false',
        help='Disable prioritized replay (uniform sampling); the default since the 2026-08 '
             'MsPacman matrix showed PER-off wins.'
    )
    parser.set_defaults(use_priority=None)
    stab_fix_group = parser.add_mutually_exclusive_group()
    stab_fix_group.add_argument(
        '--stab-fix', dest='stab_fix', action='store_true',
        help='Enable the differentiable soft_tanh policy-logits stability guard (default).'
    )
    stab_fix_group.add_argument(
        '--no-stab-fix', dest='stab_fix', action='store_false',
        help='Disable the policy-stability protections for ablations.'
    )
    parser.set_defaults(stab_fix=True)
    parser.add_argument(
        '--game-segment-length', dest='game_segment_length', type=int, default=None,
        help='Override game_segment_length (default 200; the 2025-10 known-good Pong run used 20).'
    )
    parser.add_argument(
        '--infer-context-length', dest='infer_context_length', type=int, default=None,
        help='Override the number of observation/action blocks retained by online KV inference (default 5).'
    )
    kv_window_group = parser.add_mutually_exclusive_group()
    kv_window_group.add_argument(
        '--exact-kv-window-reset', action='store_true',
        help='Hard-reset online KV context to its latest latent observation instead of applying an '
             'inexact algebraic positional shift (diagnostic mode).'
    )
    bootstrap_group = parser.add_mutually_exclusive_group()
    bootstrap_group.add_argument(
        '--bootstrap-value-context', dest='bootstrap_value_context', action='store_true',
        help='Compute TD bootstrap values from the same exact rolling replay history available '
             'to online UniZero planning instead of the longer training-only sequence context.'
    )
    bootstrap_group.add_argument(
        '--no-bootstrap-value-context', dest='bootstrap_value_context', action='store_false',
        help='Use the longer training-only sequence context for TD bootstrap values (default).'
    )
    parser.set_defaults(bootstrap_value_context=False)
    contextual_reanalysis_group = parser.add_mutually_exclusive_group()
    contextual_reanalysis_group.add_argument(
        '--contextual-reanalysis', dest='contextual_reanalysis', action='store_true',
        help='Opt in to rebuilding replay MCTS root priors and KV caches from the same short '
             'observation/action history used by online planning. Required when an explicit '
             '--buffer-reanalyze-freq is supplied.'
    )
    contextual_reanalysis_group.add_argument(
        '--no-contextual-reanalysis', dest='contextual_reanalysis', action='store_false',
        help='Keep replay roots history-free (default); explicit reanalysis frequency is rejected '
             'in this mode to prevent a silent legacy-target mismatch.'
    )
    parser.set_defaults(contextual_reanalysis=False)
    kv_window_group.add_argument(
        '--rebuild-kv-window-from-tokens', dest='rebuild_kv_window_from_tokens', action='store_true',
        help=(
            'Exactly rebuild a full learned-absolute-position KV window from retained raw '
            'observation/action embeddings whenever the window advances.'
        ),
    )
    kv_window_group.add_argument(
        '--no-rebuild-kv-window-from-tokens', dest='rebuild_kv_window_from_tokens', action='store_false',
        help='Keep the legacy KV window update path (default).'
    )
    parser.set_defaults(rebuild_kv_window_from_tokens=False)
    parser.add_argument(
        '--buffer-reanalyze-freq', type=float, default=None,
        help='Override periodic replay-buffer policy-target reanalysis frequency '
             '(e.g. 0.02 means once every 50 collect/train epochs).'
    )
    parser.add_argument(
        '--tb-log-all', action='store_true',
        help='Restore the full historical TensorBoard scalar surface for deep debugging.'
    )
    parser.add_argument(
        '--reanalyze-batch-size', type=int, default=None,
        help='Override replay sequences refreshed per reanalysis event (default 160).'
    )
    parser.add_argument(
        '--save-ckpt-after-iter', dest='save_ckpt_after_iter', type=int, default=None,
        help='Override periodic learner-checkpoint interval; useful on preemptible clusters.'
    )
    parser.add_argument(
        '--periodic-ckpt-keep-last', dest='periodic_ckpt_keep_last', type=int, default=None,
        help='Keep iteration_0 plus this many newest periodic learner checkpoints; '
             '0/default disables pruning and never affects ckpt_best.'
    )
    parser.add_argument(
        '--ignore-checkpoint-save-errors', action='store_true',
        help='Log OSError/PyTorch RuntimeError from learner checkpoint hooks and continue training. '
             'This affects only checkpoint writes; other failures remain fatal.'
    )
    eval_checkpoint_group = parser.add_mutually_exclusive_group()
    eval_checkpoint_group.add_argument(
        '--save-ckpt-in-eval', dest='save_ckpt_in_eval', action='store_true',
        help='Save ckpt_best whenever evaluation reaches a new best (default).'
    )
    eval_checkpoint_group.add_argument(
        '--no-save-ckpt-in-eval', dest='save_ckpt_in_eval', action='store_false',
        help='Keep evaluation curve logging but skip ckpt_best writes; periodic recovery '
             'checkpoints are unaffected.'
    )
    parser.set_defaults(save_ckpt_in_eval=True)
    parser.add_argument(
        '--open-loop-diagnostic-freq', dest='open_loop_diagnostic_freq', type=int, default=None,
        help='Measure detached MCTS-style autoregressive latent rollout error every N learner '
             'iterations; 0/default disables the diagnostic.'
    )
    parser.add_argument(
        '--open-loop-consistency-weight', dest='open_loop_consistency_weight', type=float, default=None,
        help='Weight for the short differentiable MCTS-style latent rollout consistency loss; '
             '0/default disables it, pass 1.0 for the v3 recipe (geometry defaults to batch 8, '
             'horizon 4, prefix 3).'
    )
    parser.add_argument(
        '--open-loop-recurrent-weight', dest='open_loop_recurrent_weight', type=float, default=None,
        help='Weight for MuZero-style latent/reward/value/policy supervision on recursively '
             'predicted states; 0/default disables it and it is mutually exclusive with '
             '--open-loop-consistency-weight.'
    )
    parser.add_argument(
        '--open-loop-consistency-batch-size', dest='open_loop_consistency_batch_size', type=int, default=None,
        help='Number of replay samples used by the optional open-loop consistency loss (default 8).'
    )
    parser.add_argument(
        '--open-loop-consistency-horizon', dest='open_loop_consistency_horizon', type=int, default=None,
        help='Number of predicted-latent transitions in the optional consistency rollout (default 4).'
    )
    parser.add_argument(
        '--open-loop-prefix-transitions', dest='open_loop_prefix_transitions', type=int, default=None,
        help='Number of real replay transitions used as a teacher prefix before the optional '
             'open-loop rollout (default 3).'
    )
    parser.add_argument(
        '--legacy-resume-alpha', dest='legacy_resume_alpha', type=float, default=None,
        help='Adaptive alpha recorded externally for a legacy checkpoint without log_alpha.'
    )
    args = parser.parse_args()

    main(
        args.env,
        args.seed,
        output_root=args.output_root,
        run_name=args.run_name,
        use_new_cache_manager=args.use_new_cache_manager,
        use_augmentation_override=args.use_augmentation,
        replay_ratio_override=args.replay_ratio,
        batch_size_override=args.batch_size,
        num_unroll_steps_override=args.num_unroll_steps,
        obs_loss_weight_override=args.obs_loss_weight,
        value_loss_weight_override=args.value_loss_weight,
        target_update_theta_override=args.target_update_theta,
        cosine_lr_scheduler_override=args.cosine_lr_scheduler,
        no_decay_bias_norm_override=args.no_decay_bias_norm,
        latent_norm_reg_weight_override=args.latent_norm_reg_weight,
        frame_stack_num_override=args.frame_stack_num,
        eval_temperature_override=args.eval_temperature,
        root_cache_key_round_decimals_override=args.root_cache_key_round_decimals,
        kv_cache_clear_interval_override=args.kv_cache_clear_interval,
        empty_cuda_cache_on_cache_reset_override=args.empty_cuda_cache_on_cache_reset,
        isolate_eval_cache=args.isolate_eval_cache,
        evaluator_env_num_override=args.evaluator_env_num,
        collect_num_simulations_override=args.collect_num_simulations,
        collect_temperature_override=args.collect_temperature,
        grad_clip_value_override=args.grad_clip_value,
        grad_clip_mode_override=args.grad_clip_mode,
        replay_buffer_size_override=args.replay_buffer_size,
        gradient_diagnostic_freq_override=args.gradient_diagnostic_freq,
        disable_adaptive_alpha=args.disable_adaptive_alpha,
        fixed_alpha=args.fixed_alpha,
        disable_policy_label_smoothing=args.disable_policy_label_smoothing,
        encoder_clip_enabled=args.encoder_clip_enabled,
        resume_from=args.resume_from,
        resume_in_place=args.resume_in_place,
        max_env_step_override=args.max_env_step,
        use_priority=args.use_priority,
        stab_fix=args.stab_fix,
        game_segment_length_override=args.game_segment_length,
        infer_context_length_override=args.infer_context_length,
        exact_kv_window_reset=args.exact_kv_window_reset,
        rebuild_kv_window_from_tokens=args.rebuild_kv_window_from_tokens,
        contextual_reanalysis=args.contextual_reanalysis,
        bootstrap_value_context=args.bootstrap_value_context,
        resume_buffer_min_transitions_override=args.resume_buffer_min_transitions,
        buffer_reanalyze_freq_override=args.buffer_reanalyze_freq,
        reanalyze_batch_size_override=args.reanalyze_batch_size,
        save_ckpt_after_iter_override=args.save_ckpt_after_iter,
        periodic_ckpt_keep_last_override=args.periodic_ckpt_keep_last,
        tb_log_all=args.tb_log_all,
        ignore_checkpoint_save_errors=args.ignore_checkpoint_save_errors,
        save_ckpt_in_eval=args.save_ckpt_in_eval,
        open_loop_diagnostic_freq_override=args.open_loop_diagnostic_freq,
        open_loop_consistency_weight_override=args.open_loop_consistency_weight,
        open_loop_recurrent_weight_override=args.open_loop_recurrent_weight,
        open_loop_consistency_batch_size_override=args.open_loop_consistency_batch_size,
        open_loop_consistency_horizon_override=args.open_loop_consistency_horizon,
        open_loop_prefix_transitions_override=args.open_loop_prefix_transitions,
        legacy_resume_alpha=args.legacy_resume_alpha,
        collector_episode_life=args.collector_episode_life,
        collector_clip_rewards=args.collector_clip_rewards,
    )
