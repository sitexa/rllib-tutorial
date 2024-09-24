# @OldAPIStack

"""
Example of a fully deterministic, repeatable RLlib train run using
the "seed" config key.
"""
import argparse

import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.examples.envs.classes.env_using_remote_actor import (
    CartPoleWithRemoteParamServer,
    ParameterStorage,
)
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
from ray.rllib.utils.test_utils import check
from ray.tune.registry import get_trainable_cls

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--framework", choices=["tf2", "tf", "torch"], default="torch")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=2)
parser.add_argument("--num-gpus", type=float, default=0)
parser.add_argument("--num-gpus-per-env-runner", type=float, default=0)

if __name__ == "__main__":
    args = parser.parse_args()

    param_storage = ParameterStorage.options(name="param-server").remote()

    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(
            CartPoleWithRemoteParamServer,
            env_config={"param_server": "param-server"},
        )
        .framework(args.framework)
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=2,
            rollout_fragment_length=50,
            num_gpus_per_env_runner=args.num_gpus_per_env_runner,
        )
        # The new Learner API.
        .learners(
            num_learners=int(args.num_gpus),
            num_gpus_per_learner=int(args.num_gpus > 0),
        )
        # Old gpu-training API.
        .resources(
            num_gpus=args.num_gpus,
        )
        # Make sure every environment gets a fixed seed.
        .debugging(seed=args.seed)
        .training(
            train_batch_size=100,
        )
    )

    if args.run == "PPO":
        # Simplify to run this example script faster.
        config.training(            
            train_batch_size=100,
            sgd_minibatch_size=10,
            num_sgd_iter=5)

    stop = {TRAINING_ITERATION: args.stop_iters}

    results1 = tune.Tuner(
        args.run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop, verbose=1, failure_config=air.FailureConfig(fail_fast="raise")
        ),
    ).fit()
    results2 = tune.Tuner(
        args.run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop, verbose=1, failure_config=air.FailureConfig(fail_fast="raise")
        ),
    ).fit()

    if args.as_test:
        results1 = results1.get_best_result().metrics
        results2 = results2.get_best_result().metrics
        # Test rollout behavior.
        check(
            results1[ENV_RUNNER_RESULTS]["hist_stats"],
            results2[ENV_RUNNER_RESULTS]["hist_stats"],
        )
        # As well as training behavior (minibatch sequence during SGD
        # iterations).
        if config.enable_rl_module_and_learner:
            check(
                results1["info"][LEARNER_INFO][DEFAULT_MODULE_ID],
                results2["info"][LEARNER_INFO][DEFAULT_MODULE_ID],
            )
        else:
            check(
                results1["info"][LEARNER_INFO][DEFAULT_MODULE_ID]["learner_stats"],
                results2["info"][LEARNER_INFO][DEFAULT_MODULE_ID]["learner_stats"],
            )
    ray.shutdown()
    
    
"""
Trial PPO_CartPoleWithRemoteParamServer_804f4_00000 completed after 2 iterations at 2024-09-24 14:59:16. Total running time: 5s
╭────────────────────────────────────────────────────────────────────────╮
│ Trial PPO_CartPoleWithRemoteParamServer_804f4_00000 result             │
├────────────────────────────────────────────────────────────────────────┤
│ env_runners/episode_len_mean                                   20.3333 │
│ env_runners/episode_return_mean                                20.3333 │
│ num_env_steps_sampled_lifetime                                     200 │
╰────────────────────────────────────────────────────────────────────────╯

解释这个运行结果。

1. 试验概况:
   - 试验名称: `PPO_CartPoleWithRemoteParamServer_804f4_00000`
   - 完成时间: 2024年9月24日 14:59:16
   - 总运行时间: 5秒
   - 完成迭代次数: 2次

2. 性能指标:
   
   a. `env_runners/episode_len_mean: 20.3333`
      - 这表示每个episode的平均长度。
      - 在CartPole环境中,这个值表示杆子保持平衡的平均时间步数。

   b. `env_runners/episode_return_mean: 20.3333`
      - 这是每个episode的平均回报(reward)。
      - 在这个例子中,它与episode长度相同,说明每个时间步可能都获得了1分的奖励。

   c. `num_env_steps_sampled_lifetime: 200`
      - 这是整个训练过程中采样的环境步骤总数。
      - 200步意味着算法在环境中总共执行了200次动作。

分析:
1. 训练时间很短(仅5秒),这符合我们之前为了快速运行而简化配置的目标。
2. 平均episode长度和回报都是20.3333,这对于CartPole任务来说相对较低。在一个完全训练好的模型中,这些值应该更高。
3. 总采样步数为200,这是一个相对较小的数字,表明这只是一个非常初步的训练。

总结: This output shows the results of a very brief training run. The agent has started learning, but given the short duration and limited number of steps, it hasn't had time to significantly improve its performance on the CartPole task. 在实际应用中,你可能需要增加训练时间和迭代次数来获得更好的性能。


Trial status: 1 TERMINATED
Current time: 2024-09-24 14:59:16. Total running time: 5s
Logical resource usage: 2.0/8 CPUs, 0/0 GPUs
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                                      status         iter     total time (s)     ts     num_healthy_workers     ...async_sample_reqs     ...e_worker_restarts     ...ent_steps_sampled │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PPO_CartPoleWithRemoteParamServer_804f4_00000   TERMINATED        2           0.245502    200                       1                        0                        0                      200 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
(PPO pid=13207) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/Users/xnpeng/ray_results/PPO_2024-09-24_14-59-11/PPO_CartPoleWithRemoteParamServer_804f4_00000_0_2024-09-24_14-59-11/checkpoint_000000)

解释这个运行结果。

1. 总体状态:
   - 试验状态: 1 TERMINATED (已终止)
   - 当前时间: 2024年9月24日 14:59:16
   - 总运行时间: 5秒
   - 逻辑资源使用: 2.0/8 CPUs, 0/0 GPUs

2. 试验详情:
   - 试验名称: PPO_CartPoleWithRemoteParamServer_804f4_00000
   - 状态: TERMINATED (已终止)
   - 迭代次数 (iter): 2
   - 总时间: 0.245502秒
   - 时间步 (ts): 200
   - 健康工作进程数 (num_healthy_workers): 1
   - 异步采样请求 (async_sample_reqs): 0
   - 工作进程重启次数 (...e_worker_restarts): 0
   - 环境步骤采样数 (...ent_steps_sampled): 200

3. 检查点信息:
   成功创建了检查点,路径为:
   `/Users/xnpeng/ray_results/PPO_2024-09-24_14-59-11/PPO_CartPoleWithRemoteParamServer_804f4_00000_0_2024-09-24_14-59-11/checkpoint_000000`

分析:
1. 训练过程很短,只进行了2次迭代,总共用时约0.25秒。
2. 使用了2个CPU核心,没有使用GPU。
3. 总共采样了200个环境步骤,这与之前的结果一致。
4. 没有发生工作进程重启或异步采样请求,表明训练过程稳定。

总结:虽然训练时间短,但系统成功创建了检查点,这对于后续的训练或评估可能有用。

Trial PPO_CartPoleWithRemoteParamServer_83a74_00000 completed after 2 iterations at 2024-09-24 14:59:21. Total running time: 4s
╭────────────────────────────────────────────────────────────────────────╮
│ Trial PPO_CartPoleWithRemoteParamServer_83a74_00000 result             │
├────────────────────────────────────────────────────────────────────────┤
│ env_runners/episode_len_mean                                   20.3333 │
│ env_runners/episode_return_mean                                20.3333 │
│ num_env_steps_sampled_lifetime                                     200 │
╰────────────────────────────────────────────────────────────────────────╯
2024-09-24 14:59:21,628	INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/Users/xnpeng/ray_results/PPO_2024-09-24_14-59-16' in 0.0063s.

Trial status: 1 TERMINATED
Current time: 2024-09-24 14:59:21. Total running time: 4s
Logical resource usage: 2.0/8 CPUs, 0/0 GPUs
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                                      status         iter     total time (s)     ts     num_healthy_workers     ...async_sample_reqs     ...e_worker_restarts     ...ent_steps_sampled │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PPO_CartPoleWithRemoteParamServer_83a74_00000   TERMINATED        2           0.244572    200                       1                        0                        0                      200 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
(PPO pid=13224) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/Users/xnpeng/ray_results/PPO_2024-09-24_14-59-16/PPO_CartPoleWithRemoteParamServer_83a74_00000_0_2024-09-24_14-59-16/checkpoint_000000)

解释这个运行结果。

1. 试验概况:
   - 试验名称: `PPO_CartPoleWithRemoteParamServer_83a74_00000`
   - 完成时间: 2024年9月24日 14:59:21
   - 总运行时间: 4秒
   - 完成迭代次数: 2次

2. 性能指标:
   - `env_runners/episode_len_mean`: 20.3333
   - `env_runners/episode_return_mean`: 20.3333
   - `num_env_steps_sampled_lifetime`: 200

3. 资源使用:
   - CPU使用: 2.0/8 CPUs
   - GPU使用: 0/0 GPUs

4. 详细试验信息:
   - 状态: TERMINATED
   - 总时间: 0.244572秒
   - 时间步: 200
   - 健康工作进程数: 1
   - 异步采样请求: 0
   - 工作进程重启次数: 0

5. 检查点信息:
   成功创建了检查点,路径为:
   `/Users/xnpeng/ray_results/PPO_2024-09-24_14-59-16/PPO_CartPoleWithRemoteParamServer_83a74_00000_0_2024-09-24_14-59-16/checkpoint_000000`

分析:
1. 这次运行的结果与之前的运行几乎完全相同,包括平均episode长度和回报。
2. 训练时间和资源使用也非常相似。
3. 同样采样了200个环境步骤。

总结:  这表明训练过程是确定性的,符合脚本的设计目标。
两次运行的结果几乎完全相同,这证明了脚本成功实现了确定性训练的目标。、
"""
