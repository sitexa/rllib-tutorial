# @OldAPIStack

"""Example of using a custom training workflow.

This example creates a number of CartPole agents, some of which are trained with
DQN, and some of which are trained with PPO. Both are executed concurrently
with a custom training workflow.
"""

import argparse
import os

import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
    MultiAgentReplayBuffer,
)
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_TARGET_UPDATES,
    LAST_TARGET_UPDATE_TS,
)
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import register_env

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--mixed-torch-tf", action="store_true")
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
         "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=600, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=200000, help="Number of timesteps to train."
)
# 600.0 = 4 (num_agents) x 150.0
parser.add_argument(
    "--stop-reward", type=float, default=600.0, help="Reward at which we stop training."
)


# Define new Algorithm with custom `training_step()` method (training workflow).
class MyAlgo(Algorithm):
    @override(Algorithm)
    def setup(self, config):
        # Call super's `setup` to create rollout workers.
        super().setup(config)
        # Create local replay buffer.
        self.local_replay_buffer = MultiAgentReplayBuffer(num_shards=1, capacity=50000)

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        # Generate common experiences, collect batch for PPO, store every (DQN) batch
        # into replay buffer.
        ppo_batches = []
        num_env_steps = 0

        # PPO batch size fixed at 200.
        # TODO: Use `max_env_steps=200` option of synchronous_parallel_sample instead.
        while num_env_steps < 200:
            ma_batches = synchronous_parallel_sample(
                worker_set=self.env_runner_group, concat=False
            )
            # Loop through ma-batches (which were collected in parallel).
            for ma_batch in ma_batches:
                # Update sampled counters.
                self._counters[NUM_ENV_STEPS_SAMPLED] += ma_batch.count
                self._counters[NUM_AGENT_STEPS_SAMPLED] += ma_batch.agent_steps()
                # Add collected batches (only for DQN policy) to replay buffer.
                ppo_batch = ma_batch.policy_batches.pop("ppo_policy")
                self.local_replay_buffer.add(ma_batch)

                ppo_batches.append(ppo_batch)
                num_env_steps += ppo_batch.count

        # DQN sub-flow.
        dqn_train_results = {}
        # Start updating DQN policy once we have some samples in the buffer.
        if self._counters[NUM_ENV_STEPS_SAMPLED] > 1000:
            # Update DQN policy n times while updating PPO policy once.
            for _ in range(10):
                dqn_train_batch = self.local_replay_buffer.sample(num_items=64)
                dqn_train_results = train_one_step(
                    self, dqn_train_batch, ["dqn_policy"]
                )
                self._counters[
                    "agent_steps_trained_DQN"
                ] += dqn_train_batch.agent_steps()
                print(
                    "DQN policy learning on samples from",
                    "agent steps trained",
                    dqn_train_batch.agent_steps(),
                )
        # Update DQN's target net every n train steps (determined by the DQN config).
        if (
                self._counters["agent_steps_trained_DQN"]
                - self._counters[LAST_TARGET_UPDATE_TS]
                >= self.get_policy("dqn_policy").config["target_network_update_freq"]
        ):
            self.env_runner.get_policy("dqn_policy").update_target()
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = self._counters[
                "agent_steps_trained_DQN"
            ]

        # PPO sub-flow.
        ppo_train_batch = concat_samples(ppo_batches)
        self._counters["agent_steps_trained_PPO"] += ppo_train_batch.agent_steps()
        # Standardize advantages.
        ppo_train_batch[Postprocessing.ADVANTAGES] = standardized(
            ppo_train_batch[Postprocessing.ADVANTAGES]
        )
        print(
            "PPO policy learning on samples from",
            "agent steps trained",
            ppo_train_batch.agent_steps(),
        )
        ppo_train_batch = MultiAgentBatch(
            {"ppo_policy": ppo_train_batch}, ppo_train_batch.count
        )
        ppo_train_results = train_one_step(self, ppo_train_batch, ["ppo_policy"])

        # Combine results for PPO and DQN into one results dict.
        results = dict(ppo_train_results, **dqn_train_results)
        return results


if __name__ == "__main__":
    args = parser.parse_args()
    assert not (
            args.torch and args.mixed_torch_tf
    ), "Use either --torch or --mixed-torch-tf, not both!"

    ray.init(local_mode=args.local_mode)

    # Simple environment with 4 independent cartpole entities
    register_env(
        "multi_agent_cartpole", lambda _: MultiAgentCartPole({"num_agents": 4})
    )

    # Note that since the algorithm below does not include a default policy or
    # policy configs, we have to explicitly set it in the multiagent config:
    policies = {
        "ppo_policy": (
            PPOTorchPolicy if args.torch or args.mixed_torch_tf else PPOTF1Policy,
            None,
            None,
            # Provide entire AlgorithmConfig object, not just an override.
            PPOConfig()
            .training(num_sgd_iter=10, sgd_minibatch_size=128)
            .framework("torch" if args.torch or args.mixed_torch_tf else "tf"),
        ),
        "dqn_policy": (
            DQNTorchPolicy if args.torch else DQNTFPolicy,
            None,
            None,
            # Provide entire AlgorithmConfig object, not just an override.
            DQNConfig().training(target_network_update_freq=500).framework("tf"),
        ),
    }


    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id % 2 == 0:
            return "ppo_policy"
        else:
            return "dqn_policy"


    config = (
        AlgorithmConfig()
        .api_stack(enable_rl_module_and_learner=False)
        .environment("multi_agent_cartpole")
        .framework("torch" if args.torch else "tf")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .env_runners(num_env_runners=0, rollout_fragment_length=50)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .reporting(metrics_num_episodes_for_smoothing=30)
    )

    stop = {
        TRAINING_ITERATION: args.stop_iters,
        NUM_ENV_STEPS_SAMPLED_LIFETIME: args.stop_timesteps,
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
    }

    results = tune.Tuner(
        MyAlgo, param_space=config.to_dict(), run_config=air.RunConfig(stop=stop)
    ).fit()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()

"""
这个脚本实现了一个自定义的强化学习训练工作流，结合了在线策略（PPO）和离线策略（DQN）算法。
以下是对脚本的主要部分的分析：

1. 导入和参数设置：
   - 导入了必要的Ray和RLlib模块。
   - 设置了命令行参数解析器，允许用户自定义运行参数。

2. 自定义算法类 MyAlgo：
   - 继承自Ray的Algorithm类。
   - 重写了setup方法，创建了一个本地回放缓冲区。
   - 重写了training_step方法，实现了自定义的训练流程。

3. 训练流程（training_step方法）：
   - 同时为PPO和DQN收集经验。
   - PPO使用同步并行采样，DQN使用回放缓冲区。
   - DQN每训练10次，PPO训练1次。
   - 实现了DQN的目标网络更新机制。
   - 对PPO的优势进行标准化处理。

4. 主函数：
   - 注册了一个多智能体CartPole环境。
   - 定义了PPO和DQN的策略配置。
   - 设置了策略映射函数，将奇数智能体分配给DQN，偶数智能体分配给PPO。
   - 配置了算法参数，包括环境、框架选择、多智能体设置等。
   - 使用Ray Tune进行调优和训练。

5. 特点：
   - 结合了在线（PPO）和离线（DQN）学习方法。
   - 支持多智能体环境。
   - 允许使用PyTorch或TensorFlow作为后端。
   - 实现了自定义的训练工作流，允许更灵活的算法组合。

6. 实验设置：
   - 使用MultiAgentCartPole环境，包含4个独立的CartPole实体。
   - 可以设置停止条件，如迭代次数、时间步数和奖励阈值。

这个脚本展示了如何在RLlib中创建复杂的自定义训练流程，特别是如何在同一个环境中结合不同类型的强化学习算法。它为研究人员和开发者提供了一个灵活的框架，用于实验和开发新的强化学习方法。
"""

"""
Trial MyAlgo_multi_agent_cartpole_8e201_00000 completed after 117 iterations at 2024-09-06 16:11:30. Total running time: 28s
╭──────────────────────────────────────────────────────────────────╮
│ Trial MyAlgo_multi_agent_cartpole_8e201_00000 result             │
├──────────────────────────────────────────────────────────────────┤
│ env_runners/episode_len_mean                             279.133 │
│ env_runners/episode_return_mean                            609.5 │
│ num_env_steps_sampled_lifetime                             22400 │
╰──────────────────────────────────────────────────────────────────╯
2024-09-06 16:11:30,488	INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/Users/xnpeng/ray_results/MyAlgo_2024-09-06_16-11-01' in 0.0168s.

Trial status: 1 TERMINATED
Current time: 2024-09-06 16:11:30. Total running time: 28s
Logical resource usage: 1.0/8 CPUs, 0/0 GPUs
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                                status         iter     total time (s)      ts     num_healthy_workers     ...async_sample_reqs     ...e_worker_restarts     ...ent_steps_sampled │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MyAlgo_multi_agent_cartpole_8e201_00000   TERMINATED      117            22.9355   22400                       0                        0                        0                    67799 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

这个运行结果提供了关于强化学习实验的重要信息。让我们逐项分析：

1. 总体情况：
   - 实验名称：MyAlgo_multi_agent_cartpole_8e201_00000
   - 完成时间：2024-09-06 16:11:30
   - 总运行时间：28秒
   - 迭代次数：117次

2. 性能指标：
   - env_runners/episode_len_mean：279.133
     这表示每个回合的平均长度约为279步。
   
   - env_runners/episode_return_mean：609.5
     这是每个回合的平均回报。达到了预设的停止奖励（600.0），说明学习效果良好。

   - num_env_steps_sampled_lifetime：22400
     这是整个实验过程中采样的环境步数总和。

3. 资源使用：
   - CPU使用：1.0/8 CPUs
   - GPU使用：0/0 GPUs（未使用GPU）

4. 详细统计：
   - total time (s)：22.9355
     实际训练时间约23秒。
   
   - ts（time steps）：22400
     与num_env_steps_sampled_lifetime一致，表示总步数。

   - num_healthy_workers：0
     这可能表示没有额外的工作进程，所有计算都在主进程中完成。

   - async_sample_reqs和remote_worker_restarts：0
     没有异步采样请求和远程工作器重启。

   - agent_steps_sampled：67799
     智能体采样的总步数，大约是环境步数的3倍（因为有4个智能体）。

分析结论：
1. 学习效果：实验成功达到了预设的奖励目标（609.5 > 600.0），表明算法学习效果良好。
2. 效率：在短时间内（28秒）就完成了学习，说明算法收敛速度快。
3. 稳定性：平均回合长度（279步）相对较高，表明智能体能够较好地平衡CartPole。
4. 资源利用：实验只使用了CPU，没有使用GPU，适合在普通计算机上运行。
5. 采样效率：智能体步数（67799）远大于环境步数（22400），说明多智能体设置有效地增加了样本利用率。

总的来说，这个实验展示了结合PPO和DQN的自定义算法在多智能体CartPole环境中的有效性，能够快速、高效地学习到良好的策略。        
"""
