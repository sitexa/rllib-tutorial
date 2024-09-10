"""Example of implementing a custom `render()` method for your gymnasium RL environment.

This example:
    - shows how to write a simple gym.Env class yourself, in this case a corridor env,
    in which the agent starts at the left side of the corridor and has to reach the
    goal state all the way at the right.
    - in particular, the new class overrides the Env's `render()` method to show, how
    you can write your own rendering logic.
    - furthermore, we use the RLlib callbacks class introduced in this example here:
    https://github.com/ray-project/ray/blob/master/rllib/examples/envs/env_rendering_and_recording.py  # noqa
    in order to compile videos of the worst and best performing episodes in each
    iteration and log these videos to your WandB account, so you can view them.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack
--wandb-key=[your WandB API key] --wandb-project=[some WandB project name]
--wandb-run-name=[optional: WandB run name within --wandb-project]`

In order to see the actual videos, you need to have a WandB account and provide your
API key and a project name on the command line (see above).

Use the `--num-agents` argument to set up the env as a multi-agent env. If
`--num-agents` > 0, RLlib will simply run as many of the defined single-agent
environments in parallel and with different policies to be trained for each agent.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.


Results to expect
-----------------
After the first training iteration, you should see the videos in your WandB account
under the provided `--wandb-project` name. Filter for "videos_best" or "videos_worst".

Note that the default Tune TensorboardX (TBX) logger might complain about the videos
being logged. This is ok, the TBX logger will simply ignore these. The WandB logger,
however, will recognize the video tensors shaped
(1 [batch], T [video len], 3 [rgb], [height], [width]) and properly create a WandB video
object to be sent to their server.

Your terminal output should look similar to this (the following is for a
`--num-agents=2` run; expect similar results for the other `--num-agents`
settings):
+---------------------+------------+----------------+--------+------------------+
| Trial name          | status     | loc            |   iter |   total time (s) |
|---------------------+------------+----------------+--------+------------------+
| PPO_env_fb1c0_00000 | TERMINATED | 127.0.0.1:8592 |      3 |          21.1876 |
+---------------------+------------+----------------+--------+------------------+
+-------+-------------------+-------------+-------------+
|    ts |   combined return |   return p1 |   return p0 |
|-------+-------------------+-------------+-------------|
| 12000 |           12.7655 |      7.3605 |      5.4095 |
+-------+-------------------+-------------+-------------+
"""
from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from PIL import Image, ImageDraw

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.examples.envs.env_rendering_and_recording import EnvRenderCallback
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray import tune

parser = add_rllib_example_script_args(
    default_iters=10,
    default_reward=9.0,
    default_timesteps=10000,
)


class CustomRenderedCorridorEnv(gym.Env):
    """Example of a custom env, for which we specify rendering behavior."""

    def __init__(self, config):
        self.end_pos = config.get("corridor_length", 10)
        self.max_steps = config.get("max_steps", 100)
        self.cur_pos = 0
        self.steps = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, 999.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.cur_pos = 0.0
        self.steps = 0
        return np.array([self.cur_pos], np.float32), {}

    def step(self, action):
        self.steps += 1
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1.0
        elif action == 1:
            self.cur_pos += 1.0
        truncated = self.steps >= self.max_steps
        terminated = self.cur_pos >= self.end_pos
        return (
            np.array([self.cur_pos], np.float32),
            10.0 if terminated else -0.1,
            terminated,
            truncated,
            {},
        )

    def render(self, mode='rgb_array'):
        """Implements rendering logic for this env (given the current observation).

        You should return a numpy RGB image like so:
        np.array([height, width, 3], dtype=np.uint8).

        Returns:
            np.ndarray: A numpy uint8 3D array (image) to render.
        """
        if mode == 'rgb_array':
            # Image dimensions.
            # Each position in the corridor is 50 pixels wide.
            width = (self.end_pos + 2) * 50
            # Fixed height of the image.
            height = 100

            # Create a new image with white background
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)

            # Draw the corridor walls
            # Grey rectangle for the corridor.
            draw.rectangle([50, 30, width - 50, 70], fill="grey")

            # Draw the agent.
            # Calculate the x coordinate of the agent.
            agent_x = (self.cur_pos + 1) * 50
            # Blue rectangle for the agent.
            draw.rectangle([agent_x + 10, 40, agent_x + 40, 60], fill="blue")

            # Draw the goal state.
            # Calculate the x coordinate of the goal.
            goal_x = self.end_pos * 50
            # Green rectangle for the goal state.
            draw.rectangle([goal_x + 10, 40, goal_x + 40, 60], fill="green")

            # Convert the image to a uint8 numpy array.
            return np.array(image, dtype=np.uint8)
        elif mode == 'human':
            # 使用 PIL 显示图像
            img = self.render(mode='rgb_array')
            img = Image.fromarray(img)
            img.show()
        else:
            super().render()  # 默认行为

# Create a simple multi-agent version of the above Env by duplicating the single-agent
# env n (n=num agents) times and having the agents act independently, each one in a
# different corridor.
MultiAgentCustomRenderedCorridorEnv = make_multi_agent(
    lambda config: CustomRenderedCorridorEnv(config)
)


if __name__ == "__main__":
    parser.add_argument("--render-mode", type=str, default="rgb_array",
                        choices=["rgb_array", "human"],
                        help="Rendering mode: 'rgb_array' or 'human'")
    parser.add_argument("--render-freq", type=int, default=10,
                        help="Frequency of rendering (every N steps)")
    args = parser.parse_args()

    print("运行参数:")
    pprint(vars(args))

    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

    # The `config` arg passed into our Env's constructor (see the class' __init__ method
    # above). Feel free to change these.
    env_options = {
        "corridor_length": 10,
        "max_steps": 100,
        "num_agents": args.num_agents,  # <- only used by the multu-agent version.
    }

    env_cls_to_use = (
        CustomRenderedCorridorEnv
        if args.num_agents == 0
        else MultiAgentCustomRenderedCorridorEnv
    )

    tune.register_env("env", lambda _: env_cls_to_use(env_options))

    # Example config switching on rendering.
    base_config = (
        PPOConfig()
        # Configure our env to be the above-registered one.
        .environment("env")
        # Plugin our env-rendering (and logging) callback. This callback class allows
        # you to fully customize your rendering behavior (which workers should render,
        # which episodes, which (vector) env indices, etc..). We refer to this example
        # script here for further details:
        # https://github.com/ray-project/ray/blob/master/rllib/examples/envs/env_rendering_and_recording.py  # noqa
        .callbacks(EnvRenderCallback)
        .evaluation(
           evaluation_num_workers=1,
           evaluation_interval=1,
           evaluation_duration=5,
           evaluation_config={"record_env": "videos"},
       )
    )
    
    if args.num_agents > 0:
        base_config.multi_agent(
            policies={f"p{i}" for i in range(args.num_agents)},
            policy_mapping_fn=lambda aid, eps, **kw: f"p{aid}",
        )

    # 打印算法配置
    # print("\n算法配置:")
    # pprint(base_config.to_dict())

    run_rllib_example_script_experiment(base_config, args)

"""
1. 程序概述：
   这个程序展示了如何为自定义的 Gymnasium 强化学习环境实现一个自定义的 `render()` 方法。
   它创建了一个简单的走廊环境，并使用 PIL 库来可视化环境状态。

2. 主要组件：

   a. CustomRenderedCorridorEnv 类：
   ```python:custom_env_render_method.py
   class CustomRenderedCorridorEnv(gym.Env):
       def __init__(self, config):
           # 初始化环境
       
       def reset(self, *, seed=None, options=None):
           # 重置环境状态
       
       def step(self, action):
           # 执行动作并返回结果
       
       def render(self):
           # 自定义渲染方法
   ```
   
   这个类定义了一个简单的走廊环境，包括初始化、重置、执行动作和渲染方法。

3. 渲染方法细节：
   `render()` 方法使用 PIL 库创建一个可视化的环境表示：
   - 绘制走廊（灰色矩形）
   - 绘制智能体（蓝色矩形）
   - 绘制目标状态（绿色矩形）

4. 多智能体支持：
   程序还包含了一个多智能体版本的环境 `MultiAgentCustomRenderedCorridorEnv`。

5. 主函数：
   - 解析命令行参数
   - 注册环境
   - 配置 PPO 算法
   - 设置多智能体配置（如果适用）
   - 运行实验

6. 特殊功能：
   - 使用 RLlib 的回调类来编译最佳和最差表现的视频
   - 支持将视频记录到 WandB（Weights & Biases）账户

7. 运行说明：
   程序提供了详细的命令行参数说明，包括如何启用新的 API 堆栈、设置 WandB 键和项目等。

8. 预期结果：
   运行后，您应该能在 WandB 账户中看到训练过程中的视频，展示了最佳和最差的表现。

这个程序是学习如何创建自定义环境渲染方法的很好例子，特别适合那些需要可视化复杂环境状态的场景。
它还展示了如何将自定义环境与 RLlib 和 WandB 等工具集成，这对于实际的强化学习项目非常有用。
"""
"""
--wandb-key=9c5b48cc3eb8716aa51b2eb3d0237b4cc5b962fa --wandb-project=rllib-tuto

生成了一些.gif文件，但并未看到运动对象。
"""