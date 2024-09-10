"""Example of defining a custom gymnasium Env to be learned by an RLlib Algorithm.

This example:
    - demonstrates how to write your own (single-agent) gymnasium Env class, define its
    physics and mechanics, the reward function used, the allowed actions (action space),
    and the type of observations (observation space), etc..
    - shows how to configure and setup this environment class within an RLlib
    Algorithm config.
    - runs the experiment with the configured algo, trying to solve the environment.

To see more details on which env we are building for this example, take a look at the
`SimpleCorridor` class defined below.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack`

Use the `--corridor-length` option to set a custom length for the corridor. Note that
for extremely long corridors, the algorithm should take longer to learn.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
You should see results similar to the following in your console output:

+--------------------------------+------------+-----------------+--------+
| Trial name                     | status     | loc             |   iter |
|--------------------------------+------------+-----------------+--------+
| PPO_SimpleCorridor_78714_00000 | TERMINATED | 127.0.0.1:85794 |      7 |
+--------------------------------+------------+-----------------+--------+

+------------------+-------+----------+--------------------+
|   total time (s) |    ts |   reward |   episode_len_mean |
|------------------+-------+----------+--------------------|
|          18.3034 | 28000 | 0.908918 |            12.9676 |
+------------------+-------+----------+--------------------+
"""
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random

from typing import Optional

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env  # noqa


parser = add_rllib_example_script_args(
    default_reward=0.9, default_iters=50, default_timesteps=100000
)
parser.add_argument(
    "--corridor-length",
    type=int,
    default=10,
    help="The length of the corridor in fields. Note that this number includes the "
    "starting- and goal states.",
)


class SimpleCorridor(gym.Env):
    """Example of a custom env in which the agent has to walk down a corridor.

    ------------
    |S........G|
    ------------
    , where S is the starting position, G is the goal position, and fields with '.'
    mark free spaces, over which the agent may step. The length of the above example
    corridor is 10.
    Allowed actions are left (0) and right (1).
    The reward function is -0.01 per step taken and a uniform random value between
    0.5 and 1.5 when reaching the goal state.

    You can configure the length of the corridor via the env's config. Thus, in your
    AlgorithmConfig, you can do:
    `config.environment(env_config={"corridor_length": ..})`.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.end_pos = config.get("corridor_length", 7)
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.cur_pos = 0
        # Return obs and (empty) info dict.
        return np.array([self.cur_pos], np.float32), {"env_state": "reset"}

    def step(self, action):
        assert action in [0, 1], action
        # Move left.
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        # Move right.
        elif action == 1:
            self.cur_pos += 1

        # The environment only ever terminates when we reach the goal state.
        terminated = self.cur_pos >= self.end_pos
        truncated = False
        # Produce a random reward from [0.5, 1.5] when we reach the goal.
        reward = random.uniform(0.5, 1.5) if terminated else -0.01
        infos = {}
        return (
            np.array([self.cur_pos], np.float32),
            reward,
            terminated,
            truncated,
            infos,
        )


if __name__ == "__main__":
    args = parser.parse_args()

    # Can also register the env creator function explicitly with:
    # register_env("corridor-env", lambda config: SimpleCorridor())

    # Or you can hard code certain settings into the Env's constructor (`config`).
    # register_env(
    #    "corridor-env-w-len-100",
    #    lambda config: SimpleCorridor({**config, **{"corridor_length": 100}}),
    # )

    # Or allow the RLlib user to set more c'tor options via their algo config:
    # config.environment(env_config={[c'tor arg name]: [value]})
    # register_env("corridor-env", lambda config: SimpleCorridor(config))

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            SimpleCorridor,  # or provide the registered string: "corridor-env"
            env_config={"corridor_length": args.corridor_length},
        )
    )

    run_rllib_example_script_experiment(base_config, args)

"""
好的，我来为您分析解读custom_gym_env.py文件的主要内容：

1. 文件概述：
   这个文件展示了如何创建一个自定义的Gymnasium环境，并使用RLlib算法来学习这个环境。

2. 主要组成部分：

   a. SimpleCorridor类：
   ```python:custom_gym_env.py
   class SimpleCorridor(gym.Env):
       def __init__(self, config: Optional[dict] = None):
           # 初始化环境
       
       def reset(self, *, seed=None, options=None):
           # 重置环境状态
       
       def step(self, action):
           # 执行动作并返回结果
   ```
   这个类定义了一个简单的走廊环境，智能体需要从起点走到终点。

   b. 命令行参数解析：
   使用argparse模块来处理命令行参数，如走廊长度等。

   c. 主函数：
   设置环境配置和算法配置，然后运行实验。

3. 环境细节：
   - 动作空间：离散空间，只有两个动作（左移0，右移1）
   - 观察空间：一维Box空间，表示当前位置
   - 奖励函数：每步-0.01，到达目标时随机给予0.5到1.5之间的奖励

4. RLlib集成：
   展示了如何将自定义环境注册到RLlib中，以及如何配置算法来使用这个环境。

5. 可配置性：
   通过env_config参数，允许用户自定义走廊长度等参数。

6. 实验运行：
   使用RLlib的实用函数来运行实验，简化了实验设置和执行过程。

这个例子非常适合学习如何创建自定义强化学习环境，以及如何将其与RLlib框架集成。它展示了从环境定义到算法配置的完整流程，对于理解强化学习实践很有帮助。
"""

"""
Number of trials: 1/1 (1 TERMINATED)
+--------------------------------+------------+-----------------+--------+------------------+------------------------+------------------------+------------------------+
| Trial name                     | status     | loc             |   iter |   total time (s) |   num_env_steps_sample |   num_episodes_lifetim |   num_env_steps_traine |
|                                |            |                 |        |                  |             d_lifetime |                      e |             d_lifetime |
|--------------------------------+------------+-----------------+--------+------------------+------------------------+------------------------+------------------------|
| PPO_SimpleCorridor_a53c8_00000 | TERMINATED | 127.0.0.1:17608 |      6 |          14.1293 |                  24000 |                   1107 |                  24000 |
+--------------------------------+------------+-----------------+--------+------------------+------------------------+------------------------+------------------------+

让我们分析一下这个运行结果：

1. 试验概况：
   - 总共进行了1次试验（Number of trials: 1/1）
   - 试验状态为TERMINATED，表示已经正常完成

2. 试验详情：
   - 试验名称：PPO_SimpleCorridor_a53c8_00000
     这表明使用的是PPO（Proximal Policy Optimization）算法，环境是SimpleCorridor

3. 性能指标：
   - 迭代次数（iter）：6次
   - 总运行时间（total time）：约14.13秒
   - 采样的环境步数（num_env_steps_sampled_lifetime）：24000步
   - 完成的回合数（num_episodes_lifetime）：1107回合
   - 训练的环境步数（num_env_steps_trained_lifetime）：24000步

4. 分析：
   - 算法效率：在相对较短的时间内（14秒）完成了6次迭代，说明PPO算法在这个简单环境中运行得很快
   - 样本效率：采样和训练的步数相同（24000），表明所有采样的数据都用于了训练
   - 回合数：完成了1107个回合，平均每个回合约21.7步（24000/1107）
   - 学习进度：由于没有提供奖励信息，我们无法直接判断学习效果，但从回合数和步数来看，智能体似乎能够相对快速地完成任务

5. 结论：
   - PPO算法在这个简单走廊环境中表现良好，能够快速完成多个回合的学习
   - 环境设置合理，允许智能体在短时间内进行大量尝试
   - 为了更全面地评估性能，建议查看奖励曲线或成功率等额外指标

这个结果表明，您的自定义环境和PPO算法配置工作正常，并且能够有效地进行强化学习训练。如果您想进一步优化性能，可以考虑调整学习率、批量大小等超参数，或者增加训练时间来观察长期学习效果。
"""