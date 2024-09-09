"""Example showing how to restore an Algorithm from a checkpoint and resume training.

Use the setup shown in this script if your experiments tend to crash after some time,
and you would therefore like to make your setup more robust and fault-tolerant.

This example:
    - runs a single- or multi-agent CartPole experiment (for multi-agent, we use
    different learning rates) thereby checkpointing the state of the Algorithm every n
    iterations.
    - stops the experiment due to an expected crash in the algorithm's main process
    after a certain number of iterations.
    - just for testing purposes, restores the entire algorithm from the latest
    checkpoint and checks, whether the state of the restored algo exactly match the
    state of the crashed one.
    - then continues training with the restored algorithm until the desired final
    episode return is reached.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --num-agents=[0 or 2]
--stop-reward-crash=[the episode return after which the algo should crash]
--stop-reward=[the final episode return to achieve after(!) restoration from the
checkpoint]
`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
First, you should see the initial tune.Tuner do it's thing:

Trial status: 1 RUNNING
Current time: 2024-06-03 12:03:39. Total running time: 30s
Logical resource usage: 3.0/12 CPUs, 0/0 GPUs
╭────────────────────────────────────────────────────────────────────────
│ Trial name                    status       iter     total time (s)
├────────────────────────────────────────────────────────────────────────
│ PPO_CartPole-v1_7b1eb_00000   RUNNING         6             15.362
╰────────────────────────────────────────────────────────────────────────
───────────────────────────────────────────────────────────────────────╮
..._sampled_lifetime     ..._trained_lifetime     ...episodes_lifetime │
───────────────────────────────────────────────────────────────────────┤
               24000                    24000                      340 │
───────────────────────────────────────────────────────────────────────╯
...

then, you should see the experiment crashing as soon as the `--stop-reward-crash`
has been reached:

```RuntimeError: Intended crash after reaching trigger return.```

At some point, the experiment should resume exactly where it left off (using
the checkpoint and restored Tuner):

Trial status: 1 RUNNING
Current time: 2024-06-03 12:05:00. Total running time: 1min 0s
Logical resource usage: 3.0/12 CPUs, 0/0 GPUs
╭────────────────────────────────────────────────────────────────────────
│ Trial name                    status       iter     total time (s)
├────────────────────────────────────────────────────────────────────────
│ PPO_CartPole-v1_7b1eb_00000   RUNNING        27            66.1451
╰────────────────────────────────────────────────────────────────────────
───────────────────────────────────────────────────────────────────────╮
..._sampled_lifetime     ..._trained_lifetime     ...episodes_lifetime │
───────────────────────────────────────────────────────────────────────┤
              108000                   108000                      531 │
───────────────────────────────────────────────────────────────────────╯

And if you are using the `--as-test` option, you should see a finel message:

```
`env_runners/episode_return_mean` of 500.0 reached! ok
```
"""
import re
import time

from ray import train, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    check_learning_achieved,
)
from ray.tune.registry import get_trainable_cls, register_env
from ray.air.integrations.wandb import WandbLoggerCallback

parser = add_rllib_example_script_args(
    default_reward=500.0, default_timesteps=10000000, default_iters=2000
)
parser.add_argument(
    "--stop-reward-crash",
    type=float,
    default=200.0,
    help="Mean episode return after which the Algorithm should crash.",
)
# By default, set `args.checkpoint_freq` to 1 and `args.checkpoint_at_end` to True.
parser.set_defaults(checkpoint_freq=1, checkpoint_at_end=True)


class CrashAfterNIters(DefaultCallbacks):
    """Callback that makes the algo crash after a certain avg. return is reached."""

    def __init__(self):
        super().__init__()
        # We have to delay crashing by one iteration just so the checkpoint still
        # gets created by Tune after(!) we have reached the trigger avg. return.
        self._should_crash = False

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        # We had already reached the mean-return to crash, the last checkpoint written
        # (the one from the previous iteration) should yield that exact avg. return.
        if self._should_crash:
            raise RuntimeError("Intended crash after reaching trigger return.")
        # Reached crashing criterion, crash on next iteration.
        elif result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN] >= args.stop_reward_crash:
            print(
                "Reached trigger return of "
                f"{result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}"
            )
            self._should_crash = True


if __name__ == "__main__":
    args = parser.parse_args()

    register_env(
        "ma_cart", lambda cfg: MultiAgentCartPole({"num_agents": args.num_agents})
    )

    # Simple generic config.
    config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .api_stack(
            enable_rl_module_and_learner=args.enable_new_api_stack,
            enable_env_runner_and_connector_v2=args.enable_new_api_stack,
        )
        .environment("CartPole-v1" if args.num_agents == 0 else "ma_cart")
        .env_runners(create_env_on_local_worker=True)
        .training(lr=0.0001)
        .callbacks(CrashAfterNIters)
    )

    # Tune config.
    # Need a WandB callback?
    tune_callbacks = []
    if args.wandb_key:
        project = args.wandb_project or (
                args.algo.lower() + "-" + re.sub("\\W+", "-", str(config.env).lower())
        )
        tune_callbacks.append(
            WandbLoggerCallback(
                api_key=args.wandb_key,
                project=args.wandb_project,
                upload_checkpoints=False,
                **({"name": args.wandb_run_name} if args.wandb_run_name else {}),
            )
        )

    # Setup multi-agent, if required.
    if args.num_agents > 0:
        config.multi_agent(
            policies={
                f"p{aid}": PolicySpec(
                    config=AlgorithmConfig.overrides(
                        lr=5e-5
                           * (aid + 1),  # agent 1 has double the learning rate as 0.
                    )
                )
                for aid in range(args.num_agents)
            },
            policy_mapping_fn=lambda aid, *a, **kw: f"p{aid}",
        )

    # Define some stopping criterion. Note that this criterion is an avg episode return
    # to be reached. The stop criterion does not consider the built-in crash we are
    # triggering through our callback.
    stop = {
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
    }

    # Run tune for some iterations and generate checkpoints.
    tuner = tune.Tuner(
        trainable=config.algo_class,
        param_space=config,
        run_config=train.RunConfig(
            callbacks=tune_callbacks,
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=args.checkpoint_at_end,
            ),
            stop=stop,
        ),
    )
    tuner_results = tuner.fit()

    # Perform a very quick test to make sure our algo (upon restoration) did not lose
    # its ability to perform well in the env.
    # - Extract the best checkpoint.
    metric = f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"
    best_result = tuner_results.get_best_result(metric=metric, mode="max")
    assert (
            best_result.metrics[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
            >= args.stop_reward_crash
    )
    # - Change our config, such that the restored algo will have an env on the local
    # EnvRunner (to perform evaluation) and won't crash anymore (remove the crashing
    # callback).
    config.callbacks(None)
    # Rebuild the algorithm (just for testing purposes).
    test_algo = config.build()
    # Load algo's state from best checkpoint.
    test_algo.restore(best_result.checkpoint)
    # Perform some checks on the restored state.
    assert test_algo.training_iteration > 0
    # Evaluate on the restored algorithm.
    test_eval_results = test_algo.evaluate()
    assert (
            test_eval_results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
            >= args.stop_reward_crash
    ), test_eval_results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
    # Train one iteration to make sure, the performance does not collapse (e.g. due
    # to the optimizer weights not having been restored properly).
    test_results = test_algo.train()
    assert (
            test_results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN] >= args.stop_reward_crash
    ), test_results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
    # Stop the test algorithm again.
    test_algo.stop()

    # Create a new Tuner from the existing experiment path (which contains the tuner's
    # own checkpoint file). Note that even the WandB logging will be continued without
    # creating a new WandB run name.
    restored_tuner = tune.Tuner.restore(
        path=tuner_results.experiment_path,
        trainable=config.algo_class,
        param_space=config,
        # Important to set this to True b/c the previous trial had failed (due to our
        # `CrashAfterNIters` callback).
        resume_errored=True,
    )
    # Continue the experiment exactly where we left off.
    tuner_results = restored_tuner.fit()

    # Not sure, whether this is really necessary, but we have observed the WandB
    # logger sometimes not logging some of the last iterations. This sleep here might
    # give it enough time to do so.
    time.sleep(20)

    if args.as_test:
        check_learning_achieved(tuner_results, args.stop_reward, metric=metric)

"""
`continue_training_from_checkpoint.py` 是一个展示如何从检查点恢复算法并继续训练的示例脚本。让我们详细分析这个程序：

1. 主要目的：
   - 展示如何在实验崩溃后从检查点恢复训练。
   - 提供一个更健壮和容错的实验设置。

2. 实验流程：
   - 运行单智能体或多智能体CartPole实验。
   - 每隔n次迭代保存算法状态的检查点。
   - 模拟在达到特定回报值后的算法崩溃。
   - 从最新检查点恢复整个算法。
   - 验证恢复的算法状态是否与崩溃前相匹配。
   - 继续训练直到达到最终目标回报。

3. 关键组件：
   - 自定义回调类 `CrashAfterNIters`：
     - 在达到特定平均回报后触发崩溃。
   - 使用 `tune.Tuner` 进行实验管理。
   - 支持WandB（Weights and Biases）集成进行日志记录。

4. 配置选项：
   - 支持单智能体和多智能体设置。
   - 可以选择PyTorch或TensorFlow作为后端。
   - 可自定义崩溃触发回报和最终目标回报。

5. 实验设置：
   - 使用PPO（Proximal Policy Optimization）算法。
   - 对于多智能体设置，使用不同的学习率。

6. 恢复和验证过程：
   - 从最佳检查点恢复算法状态。
   - 进行快速测试以确保恢复的算法性能未受影响。
   - 评估恢复的算法并进行一次训练迭代以验证稳定性。

7. 特点：
   - 提供了一个健壮的实验框架，适用于长时间运行的实验。
   - 展示了如何处理实验中的意外中断。
   - 集成了性能监控和日志记录工具。

8. 潜在应用：
   - 长时间运行的强化学习实验。
   - 需要频繁检查点和恢复能力的大规模训练。
   - 对实验稳定性和可靠性有高要求的研究项目。

9. 注意事项：
   - 脚本包含详细的注释，解释了预期的运行结果和如何解释输出。
   - 提供了调试选项，允许在RLlib代码中设置断点。

总的来说，这个脚本提供了一个全面的示例，展示了如何在实际应用中处理强化学习实验的中断和恢复。它对于需要长时间训练或在不稳定环境中运行实验的研究者和开发者特别有用。

Trial PPO_ma_cart_208c9_00000 completed after 8 iterations at 2024-09-06 17:20:41. Total running time: 21s
╭─────────────────────────────────────────────────╮
│ Trial PPO_ma_cart_208c9_00000 result            │
├─────────────────────────────────────────────────┤
│ env_runners/episode_len_mean             332.18 │
│ env_runners/episode_return_mean          551.33 │
│ num_env_steps_sampled_lifetime            32000 │
│ num_env_steps_trained_lifetime            32000 │
│ num_episodes_lifetime                       301 │
╰─────────────────────────────────────────────────╯
2024-09-06 17:20:41,806	INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/Users/xnpeng/ray_results/PPO_2024-09-06_17-19-31' in 0.0072s.

Trial status: 1 TERMINATED
Current time: 2024-09-06 17:20:41. Total running time: 21s
Logical resource usage: 3.0/8 CPUs, 0/0 GPUs
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                status         iter     total time (s)     ..._sampled_lifetime     ...episodes_lifetime     ..._trained_lifetime │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PPO_ma_cart_208c9_00000   TERMINATED        8            34.7972                    32000                      301                    32000 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
这个结果展示了一个使用PPO（Proximal Policy Optimization）算法在多智能体CartPole环境（ma_cart）中的训练过程。让我们详细分析这个结果：

1. 总体情况：
   - 试验名称：PPO_ma_cart_208c9_00000
   - 完成时间：2024-09-06 17:20:41
   - 总运行时间：21秒
   - 迭代次数：8次

2. 性能指标：
   - env_runners/episode_len_mean：332.18
     这表示每个回合的平均长度约为332步，相对较高，说明智能体表现不错。

   - env_runners/episode_return_mean：551.33
     这是每个回合的平均回报，接近但未达到预设的停止奖励（可能是600）。

   - num_env_steps_sampled_lifetime：32000
     整个实验过程中采样的环境步数总和。

   - num_env_steps_trained_lifetime：32000
     训练中使用的环境步数，与采样步数相同，说明所有采样数据都用于了训练。

   - num_episodes_lifetime：301
     整个实验过程中完成的回合总数。

3. 资源使用：
   - CPU使用：3.0/8 CPUs
   - GPU使用：0/0 GPUs（未使用GPU）

4. 效率分析：
   - 平均每次迭代耗时约4.35秒（34.7972秒 / 8次迭代）
   - 平均每个回合包含约106步（32000步 / 301回合）

5. 学习效果分析：
   - 平均回合长度（332.18步）远高于每个回合的平均步数（106步），说明智能体的表现在不断提升。
   - 平均回报（551.33）接近但未达到可能的目标值（600），表明学习效果良好但还有提升空间。

6. 稳定性：
   - 在短时间内（21秒）完成了8次迭代，显示了算法的稳定性和效率。

7. 潜在改进：
   - 可以考虑增加训练时间或迭代次数，看是否能达到更高的回报。
   - 可以尝试调整学习率或其他超参数，以提高学习效率。

总结：
这个实验展示了PPO算法在多智能体CartPole环境中的有效性。
在短时间内，算法就学习到了相对不错的策略。
虽然还没有达到最优表现，但学习曲线看起来很有希望。
考虑到多智能体环境的复杂性，这个结果是令人鼓舞的。
如果继续训练，很可能会达到更好的性能。
"""