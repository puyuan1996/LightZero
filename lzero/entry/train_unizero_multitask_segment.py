import logging
import os
from functools import partial
from typing import Tuple, Optional, List

import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.mcts import UniZeroGameBuffer as GameBuffer
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroSegmentCollector as Collector
from ding.utils import EasyTimer
timer = EasyTimer()
import torch.distributed as dist

import concurrent.futures


def eval_async(evaluator, learner_save_checkpoint, learner_train_iter, collector_envstep):
    # 确保 evaluator 的模型在正确的设备上
    # print(f"======in eval_async Rank {get_rank()}======")
    # device = torch.cuda.current_device()
    # print(f"当前默认的 GPU 设备编号: {device}")
    # torch.cuda.set_device(device)
    # print(f"set device后的 GPU 设备编号: {device}")

    # 使用 ThreadPool 来异步执行评估任务
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(evaluator.eval, learner_save_checkpoint, learner_train_iter, collector_envstep)
        return future


def allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=1):
    """
    根据不同任务的 num_of_collected_episodes 反比分配 batch_size，
    并动态调整 batch_size 限制范围以提高训练的稳定性和效率。
    
    参数:
    - cfgs: 每个任务的配置列表
    - game_buffers: 每个任务的 replay_buffer 实例列表
    - alpha: 控制反比程度的超参数 (默认为1.0)
    
    返回:
    - 分配后的 batch_size 列表
    """
    
    # 提取每个任务的 num_of_collected_episodes
    buffer_num_of_collected_episodes = [buffer.num_of_collected_episodes for buffer in game_buffers]
    
    # 获取当前的 world_size 和 rank
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # 收集所有 rank 的 num_of_collected_episodes 列表
    all_task_num_of_collected_episodes = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_task_num_of_collected_episodes, buffer_num_of_collected_episodes)

    # 将所有 rank 的 num_of_collected_episodes 拼接成一个大列表
    all_task_num_of_collected_episodes = [item for sublist in all_task_num_of_collected_episodes for item in sublist]
    if rank == 0:
        print(f'all_task_num_of_collected_episodes:{all_task_num_of_collected_episodes}')

    # 计算每个任务的反比权重
    inv_episodes = np.array([1.0 / (episodes + 1) for episodes in all_task_num_of_collected_episodes])
    inv_sum = np.sum(inv_episodes)

    # 计算总的 batch_size (所有任务 cfg.policy.batch_size 的和)
    # total_batch_size = sum([cfg.policy.batch_size for cfg in cfgs])
    total_batch_size = cfgs[0].policy.total_batch_size


    # 动态调整的部分：最小和最大的 batch_size 范围
    avg_batch_size = total_batch_size / world_size
    min_batch_size = avg_batch_size / clip_scale
    max_batch_size = avg_batch_size * clip_scale

    # 动态调整 alpha，让 batch_size 的变化更加平滑
    task_weights = (inv_episodes / inv_sum) ** alpha
    batch_sizes = total_batch_size * task_weights
    
    # 控制 batch_size 在 [min_batch_size, max_batch_size] 之间
    batch_sizes = np.clip(batch_sizes, min_batch_size, max_batch_size)
    
    # 确保 batch_size 是整数
    batch_sizes = [int(size) for size in batch_sizes]
    
    # 返回最终分配的 batch_size 列表
    return batch_sizes



"""
对所有game的任务继续均匀划分：
每个game 对应 1个gpu进程
或者多个game对应 1个gpu进程

collector和learner是串行的
evaluator是异步的进程，以避免一个环境评估时的一局步数过长会导致超时

修复了当games>gpu数量时的bug
"""
def train_unizero_multitask_segment(
        input_cfg_list: List[Tuple[int, Tuple[dict, dict]]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        The train entry for UniZero, proposed in our paper UniZero: Generalized and Efficient Planning with Scalable Latent World Models.
        UniZero aims to enhance the planning capabilities of reinforcement learning agents by addressing the limitations found in MuZero-style algorithms,
        particularly in environments requiring the capture of long-term dependencies. More details can be found in https://arxiv.org/abs/2406.10667.
    Arguments:
        - input_cfg_list (List[Tuple[int, Tuple[dict, dict]]]): List of configurations for different tasks.
        - seed (int): Random seed.
        - model (Optional[torch.nn.Module]): Instance of torch.nn.Module.
        - model_path (Optional[str]): The pretrained model path, which should point to the ckpt file of the pretrained model.
        - max_train_iter (Optional[int]): Maximum policy update iterations in training.
        - max_env_step (Optional[int]): Maximum collected environment interaction steps.
    Returns:
        - policy (Policy): Converged policy.
    """
    # 获取当前进程的 rank 和总的进程数
    rank = get_rank()
    world_size = get_world_size()

    # 任务划分
    total_tasks = len(input_cfg_list)
    tasks_per_rank = total_tasks // world_size
    remainder = total_tasks % world_size

    if rank < remainder:
        start_idx = rank * (tasks_per_rank + 1)
        end_idx = start_idx + tasks_per_rank + 1
    else:
        start_idx = rank * tasks_per_rank + remainder
        end_idx = start_idx + tasks_per_rank

    tasks_for_this_rank = input_cfg_list[start_idx:end_idx]

    # 确保至少有一个任务
    if len(tasks_for_this_rank) == 0:
        print(f"Rank {rank}: No tasks assigned, exiting.")
        return

    print(f"Rank {rank}/{world_size}, handling tasks {start_idx} to {end_idx - 1}")

    cfgs = []
    game_buffers = []
    collector_envs = []
    evaluator_envs = []
    collectors = []
    evaluators = []

    # 使用第一个任务的配置来创建共享的 policy
    # task_id, [cfg, create_cfg] = input_cfg_list[0]
    # TODO: 使用该rank的第一个任务的配置来创建共享的 policy
    task_id, [cfg, create_cfg] = tasks_for_this_rank[0]

    # TODO: task_num for base_index for learner_log
    for config in tasks_for_this_rank:
        config[1][0].policy.task_num = tasks_per_rank

    # 确保指定的 policy 类型是支持的
    assert create_cfg.policy.type in ['unizero_multitask'], "train_unizero entry now only supports 'unizero_multitask'"

    # 根据 CUDA 可用性设置设备
    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f'cfg.policy.device: {cfg.policy.device}')

    # 编译配置
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # 创建共享的 policy
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # 如果指定了预训练模型，则加载
    if model_path is not None:
        logging.info(f'Loading model from {model_path} begin...')
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info(f'Loading model from {model_path} end!')

    # 创建 TensorBoard 的日志记录器
    # tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None

    # =========== TODO: for unizero_multitask ddp_v2 ========
    # tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

    log_dir = os.path.join('./{}/log'.format(cfg.exp_name), f'serial_rank_{rank}')
    tb_logger = SummaryWriter(log_dir)

    # 创建共享的 learner
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # tb_logge_train = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial_train')) if get_rank() == 0 else None
    # learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger_train, exp_name=cfg.exp_name)

    policy_config = cfg.policy
    batch_size = policy_config.batch_size[0]

    # 只处理当前进程分配到的任务
    for local_task_id, (task_id, [cfg, create_cfg]) in enumerate(tasks_for_this_rank):
        # 设置每个任务自己的随机种子
        cfg.policy.device = 'cuda' if cfg.policy.cuda and torch.cuda.is_available() else 'cpu'
        cfg = compile_config(cfg, seed=seed + task_id, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
        policy_config = cfg.policy
        policy.collect_mode.get_attribute('cfg').n_episode = policy_config.n_episode
        policy.eval_mode.get_attribute('cfg').n_episode = policy_config.n_episode

        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
        collector_env.seed(cfg.seed + task_id)
        evaluator_env.seed(cfg.seed + task_id, dynamic_seed=False)
        set_pkg_seed(cfg.seed + task_id, use_cuda=cfg.policy.cuda)

        # 为每个任务创建不同的 game buffer、collector、evaluator
        replay_buffer = GameBuffer(policy_config)
        collector = Collector(
            env=collector_env,
            policy=policy.collect_mode,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            policy_config=policy_config,
            task_id=task_id
        )
        evaluator = Evaluator(
            eval_freq=cfg.policy.eval_freq,
            n_evaluator_episode=cfg.env.n_evaluator_episode,
            stop_value=cfg.env.stop_value,
            env=evaluator_env,
            policy=policy.eval_mode,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            policy_config=policy_config,
            task_id=task_id
        )

        cfgs.append(cfg)
        replay_buffer.batch_size = cfg.policy.batch_size[task_id]

        # print(f"Rank {rank}/{world_size}, cfg.policy.batch_size:{cfg.policy.batch_size}, task_id: {task_id}")

        game_buffers.append(replay_buffer)
        collector_envs.append(collector_env)
        evaluator_envs.append(evaluator_env)
        collectors.append(collector)
        evaluators.append(evaluator)

    learner.call_hook('before_run')
    value_priority_tasks = {}

    buffer_reanalyze_count = 0
    train_epoch = 0
    reanalyze_batch_size = cfg.policy.reanalyze_batch_size
    update_per_collect = cfg.policy.update_per_collect

    while True:
        # 预先计算位置嵌入矩阵（如果需要）
        # policy._collect_model.world_model.precompute_pos_emb_diff_kv()
        # policy._target_model.world_model.precompute_pos_emb_diff_kv()

        if  cfg.policy.allocated_batch_sizes:
            # TODO==========
            # 线性变化的 随着 train_epoch 从 0 增加到 1000, clip_scale 从 1 线性增加到 4
            clip_scale = np.clip(1 + (3 * train_epoch / 1000), 1, 4)
            allocated_batch_sizes = allocate_batch_size(cfgs, game_buffers, alpha=1.0, clip_scale=clip_scale)
            if rank == 0:
                print("分配后的 batch_sizes: ", allocated_batch_sizes)
            for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                    zip(cfgs, collectors, evaluators, game_buffers)):
                cfg.policy.batch_size = allocated_batch_sizes
                policy._cfg.batch_size = allocated_batch_sizes
            # replay_buffer.batch_size 

        # 对于当前进程的每个任务，进行数据收集和评估
        for idx, (cfg, collector, evaluator, replay_buffer) in enumerate(
                zip(cfgs, collectors, evaluators, game_buffers)):
            
            # TODO: DEBUG =========
            # stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            
            # 调用异步评估
            # print(f"=========before eval_async Rank {rank}/{world_size}===========")
            # device = torch.cuda.current_device()
            # print(f"当前默认的 GPU 设备编号: {device}")
            # torch.cuda.set_device(device)
            # print(f"set device后的 GPU 设备编号: {device}")

            # eval_future = eval_async(evaluator, learner.save_checkpoint, learner.train_iter, collector.envstep, f'cuda:{rank}')
            # # 训练继续进行，不等待评估完成
            # # 你可以在某个时刻检查评估是否完成
            # if eval_future.done():
            #     stop, reward = eval_future.result()
            # else:
            #     logging.info(f"Rank {rank} Evaluation is still running...")
            
            # print(f"======after eval_async Rank {rank}/{world_size}======")
            # device = torch.cuda.current_device()
            # print(f"当前默认的 GPU 设备编号: {device}")
            # torch.cuda.set_device(device)
            # print(f"set device后的 GPU 设备编号: {device}")

            log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger, cfg.policy.task_id)

            collect_kwargs = {
                'temperature': visit_count_temperature(
                    policy_config.manual_temperature_decay,
                    policy_config.fixed_temperature_value,
                    policy_config.threshold_training_steps_for_final_temperature,
                    trained_steps=learner.train_iter
                ),
                'epsilon': 0.0  # 默认的 epsilon 值
            }

            if policy_config.eps.eps_greedy_exploration_in_collect:
                epsilon_greedy_fn = get_epsilon_greedy_fn(
                    start=policy_config.eps.start,
                    end=policy_config.eps.end,
                    decay=policy_config.eps.decay,
                    type_=policy_config.eps.type
                )
                collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

            if evaluator.should_eval(learner.train_iter):
                print('=' * 20)
                print(f'Rank {rank} evaluates task_id: {cfg.policy.task_id}...')
                # stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                
                eval_future = eval_async(evaluator, learner.save_checkpoint, learner.train_iter, collector.envstep)
                # 训练继续进行，不等待评估完成
                # 你可以在某个时刻检查评估是否完成
                if eval_future.done():
                    stop, reward = eval_future.result()
                else:
                    logging.info(f"Rank {rank} Evaluation is still running...")
                
                if stop:
                    break

            print('=' * 20)
            print(f'entry: Rank {rank} collects task_id: {cfg.policy.task_id}...')

            # NOTE: 在每次收集之前重置初始数据，这对于多任务设置非常重要
            collector._policy.reset(reset_init_data=True)
            # 收集数据
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

            # 更新 replay buffer
            replay_buffer.push_game_segments(new_data)
            replay_buffer.remove_oldest_data_to_fit()

            # 周期性地重新分析缓冲区
            if cfg.policy.buffer_reanalyze_freq >= 1:
                # 在一个训练 epoch 中重新分析缓冲区 <buffer_reanalyze_freq> 次
                reanalyze_interval = update_per_collect // cfg.policy.buffer_reanalyze_freq
            else:
                # 每 <1/buffer_reanalyze_freq> 个训练 epoch 重新分析一次缓冲区
                if train_epoch % int(1/cfg.policy.buffer_reanalyze_freq) == 0 and replay_buffer.get_num_of_transitions()//cfg.policy.num_unroll_steps > int(reanalyze_batch_size/cfg.policy.reanalyze_partition):
                    with timer:
                        # 每个重新分析过程将重新分析 <reanalyze_batch_size> 个序列
                        replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                    buffer_reanalyze_count += 1
                    logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                    logging.info(f'Buffer reanalyze time: {timer.value}')

            # 数据收集结束后添加日志
            logging.info(f'Rank {rank}: Completed data collection for task {cfg.policy.task_id}')

        # batch_size = policy_config.batch_size[0]
        # 检查是否有足够的数据进行训练
        not_enough_data = any(replay_buffer.get_num_of_transitions() < cfgs[0].policy.total_batch_size/world_size for replay_buffer in game_buffers)

        # 同步训练前所有 rank 的准备状态
        try:
            # logging.info(f'Rank {rank}: Reached barrier before training')
            dist.barrier()
            logging.info(f'Rank {rank}: Passed barrier before training')
        except Exception as e:
            logging.error(f'Rank {rank}: Barrier failed with error {e}')
            break  # 或者进行其他错误处理

        # 学习策略
        if not not_enough_data:
            # Learner 将在一次迭代中训练 update_per_collect 次
            for i in range(update_per_collect):
                train_data_multi_task = []
                envstep_multi_task = 0
                for idx, (cfg, collector, replay_buffer) in enumerate(zip(cfgs, collectors, game_buffers)):
                    envstep_multi_task += collector.envstep
                    batch_size = cfg.policy.batch_size[cfg.policy.task_id]
                    if replay_buffer.get_num_of_transitions() > batch_size:
                        # batch_size = cfg.policy.batch_size[cfg.policy.task_id]

                        if cfg.policy.buffer_reanalyze_freq >= 1:
                            # 在一个训练 epoch 中重新分析缓冲区 <buffer_reanalyze_freq> 次
                            if i % reanalyze_interval == 0 and replay_buffer.get_num_of_transitions() // cfg.policy.num_unroll_steps > int(
                                    reanalyze_batch_size / cfg.policy.reanalyze_partition):
                                with timer:
                                    # 每个重新分析过程将重新分析 <reanalyze_batch_size> 个序列
                                    replay_buffer.reanalyze_buffer(reanalyze_batch_size, policy)
                                buffer_reanalyze_count += 1
                                logging.info(f'Buffer reanalyze count: {buffer_reanalyze_count}')
                                logging.info(f'Buffer reanalyze time: {timer.value}')

                        train_data = replay_buffer.sample(batch_size, policy)
                        # 追加 task_id，以便在训练时区分任务
                        train_data.append(cfg.policy.task_id)
                        logging.info(f'Rank {rank}: cfg.policy.task_id : {cfg.policy.task_id}')

                        train_data_multi_task.append(train_data)
                    else:
                        logging.warning(
                            f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                            f'batch_size: {batch_size}, replay_buffer: {replay_buffer}'
                        )
                        break

                if train_data_multi_task:
                    # 在训练时，DDP 会自动同步梯度和参数
                    # log_vars = learner.train(train_data_multi_task, envstep_multi_task)
                    # logging.info(f'Rank {rank}: cfg.policy.batch_size : {cfg.policy.batch_size}, batch_size: {batch_size}')
                    try:
                        log_vars = learner.train(train_data_multi_task, envstep_multi_task)
                    except Exception as e:
                        logging.error(f'Rank {rank}: Training failed with error {e}')
                        break  # 或者进行其他错误处理

                if cfg.policy.use_priority:
                    for idx, (cfg, replay_buffer) in enumerate(zip(cfgs, game_buffers)):
                        # 更新任务特定的 replay buffer 的优先级
                        task_id = cfg.policy.task_id
                        replay_buffer.update_priority(train_data_multi_task[idx], log_vars[0][f'value_priority_task{task_id}'])

                        current_priorities = log_vars[0][f'value_priority_task{task_id}']

                        mean_priority = np.mean(current_priorities)
                        std_priority = np.std(current_priorities)

                        alpha = 0.1  # 运行均值的平滑因子
                        if f'running_mean_priority_task{task_id}' not in value_priority_tasks:
                            # 如果不存在，则初始化运行均值
                            value_priority_tasks[f'running_mean_priority_task{task_id}'] = mean_priority
                        else:
                            # 更新运行均值
                            value_priority_tasks[f'running_mean_priority_task{task_id}'] = (
                                alpha * mean_priority + (1 - alpha) * value_priority_tasks[f'running_mean_priority_task{task_id}']
                            )

                        # 使用运行均值计算归一化的优先级
                        running_mean_priority = value_priority_tasks[f'running_mean_priority_task{task_id}']
                        normalized_priorities = (current_priorities - running_mean_priority) / (std_priority + 1e-6)

                        # 如果需要，可以将归一化的优先级存储回 replay buffer
                        # replay_buffer.update_priority(train_data_multi_task[idx], normalized_priorities)

                        # 如果设置了 print_task_priority_logs 标志，则记录统计信息
                        if cfg.policy.print_task_priority_logs:
                            print(f"Task {task_id} - Mean Priority: {mean_priority:.8f}, "
                                  f"Running Mean Priority: {running_mean_priority:.8f}, "
                                  f"Standard Deviation: {std_priority:.8f}")

        train_epoch += 1
        policy.recompute_pos_emb_diff_and_clear_cache()

        # 同步所有 Rank，确保所有 Rank 都完成了训练
        try:
            # logging.info(f'Rank {rank}: Reached barrier after training')
            dist.barrier()
            logging.info(f'Rank {rank}: Passed barrier after training')
        except Exception as e:
            logging.error(f'Rank {rank}: Barrier failed with error {e}')
            break  # 或者进行其他错误处理


        # 检查是否需要终止训练
        try:
            # local_envsteps 不再需要填充
            local_envsteps = [collector.envstep for collector in collectors]

            total_envsteps = [None for _ in range(world_size)]
            # logging.info(f'Rank {rank}: Gathering envsteps...')
            dist.all_gather_object(total_envsteps, local_envsteps)
            # logging.info(f'Rank {rank}: Gathering envsteps completed.')

            # 将所有 envsteps 拼接在一起
            all_envsteps = torch.cat([torch.tensor(envsteps, device=cfg.policy.device) for envsteps in total_envsteps])
            max_envstep_reached = torch.all(all_envsteps >= max_env_step)

            # 收集所有进程的 train_iter
            global_train_iter = torch.tensor([learner.train_iter], device=cfg.policy.device)
            all_train_iters = [torch.zeros_like(global_train_iter) for _ in range(world_size)]
            # logging.info(f'Rank {rank}: Gathering train_iters...')
            dist.all_gather(all_train_iters, global_train_iter)
            # logging.info(f'Rank {rank}: Gathering train_iters completed.')

            max_train_iter_reached = torch.any(torch.stack(all_train_iters) >= max_train_iter)

            if max_envstep_reached.item() or max_train_iter_reached.item():
                logging.info(f'Rank {rank}: Termination condition met')
                dist.barrier()  # 确保所有进程同步
                break
            else:
                # logging.info(f'Rank {rank}: Termination condition not met')
                pass

        except Exception as e:
            logging.error(f'Rank {rank}: Termination check failed with error {e}')
            break  # 或者进行其他错误处理
        


    learner.call_hook('after_run')
    return policy