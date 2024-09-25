"""Example extracting a checkpoint from n trials using one or more custom criteria.

This example:
    - runs a CartPole experiment with three different learning rates (three tune
    "trials"). During the experiment, for each trial, we create a checkpoint at each
    iteration.
    - at the end of the experiment, we compare the trials and pick the one that
    performed best, based on the criterion: Lowest episode count per single iteration
    (for CartPole, a low episode count means the episodes are very long and thus the
    reward is also very high).
    - from that best trial (with the lowest episode count), we then pick those
    checkpoints that a) have the lowest policy loss (good) and b) have the highest value
    function loss (bad).


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
In the console output, you can see the performance of the three different learning
rates used here:

+-----------------------------+------------+-----------------+--------+--------+
| Trial name                  | status     | loc             |     lr |   iter |
|-----------------------------+------------+-----------------+--------+--------+
| PPO_CartPole-v1_d7dbe_00000 | TERMINATED | 127.0.0.1:98487 | 0.01   |     17 |
| PPO_CartPole-v1_d7dbe_00001 | TERMINATED | 127.0.0.1:98488 | 0.001  |      8 |
| PPO_CartPole-v1_d7dbe_00002 | TERMINATED | 127.0.0.1:98489 | 0.0001 |      9 |
+-----------------------------+------------+-----------------+--------+--------+

+------------------+-------+----------+----------------------+----------------------+
|   total time (s) |    ts |   reward |   episode_reward_max |   episode_reward_min |
|------------------+-------+----------+----------------------+----------------------+
|          28.1068 | 39797 |   151.11 |                  500 |                   12 |
|          13.304  | 18728 |   158.91 |                  500 |                   15 |
|          14.8848 | 21069 |   167.36 |                  500 |                   13 |
+------------------+-------+----------+----------------------+----------------------+

+--------------------+
|   episode_len_mean |
|--------------------|
|             151.11 |
|             158.91 |
|             167.36 |
+--------------------+
"""

from ray import tune
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    LEARNER_RESULTS,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls

parser = add_rllib_example_script_args(
    default_reward=450.0, default_timesteps=100000, default_iters=200
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Force-set `args.checkpoint_freq` to 1.
    args.checkpoint_freq = 1

    # Simple generic config.
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("CartPole-v1")
        # Run 3 trials, each w/ a different learning rate.
        .training(lr=tune.grid_search([0.01, 0.001, 0.0001]), train_batch_size=2341)
    )
    # Run tune for some iterations and generate checkpoints.
    results = run_rllib_example_script_experiment(base_config, args)

    # Get the best of the 3 trials by using some metric.
    # NOTE: Choosing the min `episodes_this_iter` automatically picks the trial
    # with the best performance (over the entire run (scope="all")):
    # The fewer episodes, the longer each episode lasted, the more reward we
    # got each episode.
    # Setting scope to "last", "last-5-avg", or "last-10-avg" will only compare
    # (using `mode=min|max`) the average values of the last 1, 5, or 10
    # iterations with each other, respectively.
    # Setting scope to "avg" will compare (using `mode`=min|max) the average
    # values over the entire run.
    metric = "env_runners/num_episodes"
    # notice here `scope` is `all`, meaning for each trial,
    # all results (not just the last one) will be examined.
    best_result = results.get_best_result(metric=metric, mode="min", scope="all")
    value_best_metric = best_result.metrics_dataframe[metric].min()
    best_return_best = best_result.metrics_dataframe[
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"
    ].max()
    print(
        f"Best trial was the one with lr={best_result.metrics['config']['lr']}. "
        f"Reached lowest episode count ({value_best_metric}) in a single iteration and "
        f"an average return of {best_return_best}."
    )

    # Confirm, we picked the right trial.

    assert (
            value_best_metric
            == results.get_dataframe(filter_metric=metric, filter_mode="min")[metric].min()
    )

    # Get the best checkpoints from the trial, based on different metrics.
    # Checkpoint with the lowest policy loss value:
    if args.enable_new_api_stack:
        policy_loss_key = f"{LEARNER_RESULTS}/{DEFAULT_MODULE_ID}/policy_loss"
    else:
        policy_loss_key = "info/learner/default_policy/learner_stats/policy_loss"
    best_result = results.get_best_result(metric=policy_loss_key, mode="min")
    ckpt = best_result.checkpoint
    lowest_policy_loss = best_result.metrics_dataframe[policy_loss_key].min()
    print(f"Checkpoint w/ lowest policy loss ({lowest_policy_loss}): {ckpt}")

    # Checkpoint with the highest value-function loss:
    if args.enable_new_api_stack:
        vf_loss_key = f"{LEARNER_RESULTS}/{DEFAULT_MODULE_ID}/vf_loss"
    else:
        vf_loss_key = "info/learner/default_policy/learner_stats/vf_loss"
    best_result = results.get_best_result(metric=vf_loss_key, mode="max")
    ckpt = best_result.checkpoint
    highest_value_fn_loss = best_result.metrics_dataframe[vf_loss_key].max()
    print(f"Checkpoint w/ highest value function loss: {ckpt}")
    print(f"Highest value function loss: {highest_value_fn_loss}")

"""
这个程序是一个使用Ray RLlib框架的强化学习示例，主要目的是展示如何根据自定义标准从多个试验中提取最佳检查点。让我们详细分析这个程序：

1. 实验设置：
   - 使用PPO（Proximal Policy Optimization）算法。
   - 环境是CartPole-v1。
   - 运行三个不同学习率的试验（0.01, 0.001, 0.0001）。

2. 主要功能：
   - 运行实验并在每次迭代中创建检查点。
   - 根据自定义标准选择最佳试验。
   - 从最佳试验中选择特定的检查点。

3. 自定义选择标准：
   - 最佳试验选择：选择单次迭代中回合数最少的试验（意味着每个回合持续时间长，奖励高）。
   - 最佳检查点选择：
     a. 选择策略损失最低的检查点。
     b. 选择值函数损失最高的检查点。

4. 程序流程：
   - 设置命令行参数解析器。
   - 配置实验参数，包括环境、算法、学习率等。
   - 运行实验并生成检查点。
   - 分析结果，选择最佳试验。
   - 从最佳试验中选择特定检查点。

5. 结果分析：
   - 打印三个不同学习率的试验结果。
   - 显示最佳试验的学习率、最低回合数和平均回报。
   - 输出最低策略损失和最高值函数损失对应的检查点。

6. 特点：
   - 支持新旧API栈。
   - 使用Ray Tune进行实验管理和结果分析。
   - 展示了如何使用自定义标准评估和选择模型。

7. 应用场景：
   - 这种方法适用于需要根据特定标准选择最佳模型的场景。
   - 对于需要平衡多个指标的复杂强化学习任务特别有用。

8. 潜在改进：
   - 可以添加更多自定义标准来评估模型性能。
   - 可以扩展到更复杂的环境和算法。

总的来说，这个程序展示了如何在强化学习中进行高级模型选择和评估，为研究人员和开发者提供了一个灵活的框架来实现自定义的模型选择策略。
"""

"""
+-----------------------------+------------+-----------------+--------+--------+------------------+--------+-----------------------+------------------------+------------------------+
| Trial name                  | status     | loc             |     lr |   iter |   total time (s) |     ts |   num_healthy_workers |   num_in_flight_async_ |   num_remote_worker_re |
|                             |            |                 |        |        |                  |        |                       |            sample_reqs |                 starts |
|-----------------------------+------------+-----------------+--------+--------+------------------+--------+-----------------------+------------------------+------------------------|
| PPO_CartPole-v1_89624_00000 | TERMINATED | 127.0.0.1:63693 | 0.01   |     43 |          80.9289 | 100663 |                     2 |                      0 |                      0 |
| PPO_CartPole-v1_89624_00001 | TERMINATED | 127.0.0.1:63694 | 0.001  |     30 |          56.428  |  70230 |                     2 |                      0 |                      0 |
| PPO_CartPole-v1_89624_00002 | TERMINATED | 127.0.0.1:63734 | 0.0001 |     24 |          41.6519 |  56184 |                     2 |                      0 |                      0 |
+-----------------------------+------------+-----------------+--------+--------+------------------+--------+-----------------------+------------------------+------------------------+

这个结果表格展示了三个不同学习率的PPO（Proximal Policy Optimization）算法在CartPole-v1环境中的训练结果。让我们逐项分析：

1. 试验概览：
   - 共进行了3个试验，每个使用不同的学习率。
   - 所有试验都成功完成（状态为TERMINATED）。

2. 学习率比较：
   - 试验00000：lr = 0.01
   - 试验00001：lr = 0.001
   - 试验00002：lr = 0.0001

3. 迭代次数和时间：
   - lr = 0.01：43次迭代，总时间80.9289秒
   - lr = 0.001：30次迭代，总时间56.428秒
   - lr = 0.0001：24次迭代，总时间41.6519秒
   观察：较高的学习率需要更多的迭代和时间。

4. 时间步数（ts）：
   - lr = 0.01：100663步
   - lr = 0.001：70230步
   - lr = 0.0001：56184步
   观察：较高的学习率在相同时间内完成了更多的环境交互。

5. 工作进程：
   - 每个试验都有2个健康的工作进程（num_healthy_workers）。
   - 没有进行中的异步采样请求或远程工作进程重启。

6. 性能分析：
   - 学习率0.01的试验完成了最多的时间步数，可能表示学习效率最高。
   - 学习率0.0001的试验完成的迭代次数最少，但每次迭代可能更稳定。
   - 学习率0.001似乎在迭代次数和总时间步数之间取得了平衡。

7. 效率考虑：
   - 较高的学习率（0.01）可能导致更快的学习，但也可能带来不稳定性。
   - 较低的学习率（0.0001）可能学习较慢，但可能更稳定。

8. 建议：
   - 根据具体需求选择合适的学习率。如果追求快速学习，可以选择较高的学习率；如果追求稳定性，可以选择较低的学习率。
   - 可以考虑使用学习率衰减策略，开始时使用较高的学习率，然后逐渐降低。

总的来说，这个实验很好地展示了学习率对PPO算法在CartPole环境中性能的影响。选择合适的学习率对于平衡学习速度和稳定性至关重要。
"""