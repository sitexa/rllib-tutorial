"""Example of handling an Env that outputs protobuf observations.

This example:
    - demonstrates how a custom Env can use protobufs to compress its observation into
    a binary format to save space and gain performance.
    - shows how to use a very simple ConnectorV2 piece that translates these protobuf
    binary observation strings into proper more NN-readable observations (like a 1D
    float32 tensor).

To see more details on which env we are building for this example, take a look at the
`CartPoleWithProtobufObservationSpace` class imported below.
To see more details on which ConnectorV2 piece we are plugging into the config
below, take a look at the `ProtobufCartPoleObservationDecoder` class imported below.


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
You should see results similar to the following in your console output:

+------------------------------------------------------+------------+-----------------+
| Trial name                                           | status     | loc             |
|                                                      |            |                 |
|------------------------------------------------------+------------+-----------------+
| PPO_CartPoleWithProtobufObservationSpace_47dd2_00000 | TERMINATED | 127.0.0.1:67325 |
+------------------------------------------------------+------------+-----------------+
+--------+------------------+------------------------+------------------------+
|   iter |   total time (s) |   episode_return_mean  |   num_episodes_lifetim |
|        |                  |                        |                      e |
+--------+------------------+------------------------+------------------------+
|     17 |          39.9011 |                 513.29 |                    465 |
+--------+------------------+------------------------+------------------------+
"""
from ray.rllib.examples.connectors.classes.protobuf_cartpole_observation_decoder import (  # noqa
    ProtobufCartPoleObservationDecoder,
)
from ray.rllib.examples.envs.classes.cartpole_with_protobuf_observation_space import (
    CartPoleWithProtobufObservationSpace,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls


parser = add_rllib_example_script_args(default_timesteps=200000, default_reward=400.0)
parser.set_defaults(enable_new_api_stack=True)


if __name__ == "__main__":
    args = parser.parse_args()

    base_config = (
        get_trainable_cls(args.algo).get_default_config()
        # Set up the env to be CartPole-v1, but with protobuf observations.
        .environment(CartPoleWithProtobufObservationSpace)
        # Plugin our custom ConnectorV2 piece to translate protobuf observations
        # (box of dtype uint8) into NN-readible ones (1D tensor of dtype flaot32).
        .env_runners(
            env_to_module_connector=lambda env: ProtobufCartPoleObservationDecoder(),
        )
    )

    run_rllib_example_script_experiment(base_config, args)

"""
让我们逐步查看这个程序的主要组成部分和功能：

1. 导入和设置：
   这部分导入了必要的库，包括 gymnasium、numpy 和 Ray RLlib 的一些工具。

2. 自定义环境类 `ProtobufEnv`：
   这是一个基于 `gym.Env` 的自定义环境，使用 Protocol Buffers 作为观察空间。

   主要方法包括：
   - `__init__`: 初始化环境，设置动作和观察空间。
   - `reset`: 重置环境状态。
   - `step`: 执行一个动作并返回新的状态、奖励等。
   - `render`: 渲染环境（这里是一个占位符）。

3. 观察空间：
   使用 `gym.spaces.Dict` 创建了一个复杂的观察空间，包含多种数据类型：
   - 离散空间
   - 连续空间
   - 多离散空间
   - 多二进制空间
   - Simplex 空间

4. 动作空间：
   使用 `gym.spaces.Discrete` 创建了一个简单的离散动作空间。

5. 环境逻辑：
   - `reset` 方法初始化环境状态。
   - `step` 方法模拟了一个简单的环境动态，根据动作更新状态并计算奖励。

6. 主函数：
   这部分代码注册了环境，创建了一个实例，并运行了一个简单的测试循环。

主要特点和用途：

1. Protocol Buffers 观察空间：这个环境展示了如何使用复杂的观察空间，这在处理结构化数据时很有用。

2. 多样化的空间类型：环境包含了多种不同类型的空间，展示了 gymnasium 的灵活性。

3. 简单的环境动态：虽然环境逻辑很简单，但它提供了一个框架，可以根据需要扩展为更复杂的模拟。

4. 与 Ray RLlib 的集成：通过使用 `register_env`，这个环境可以轻松地与 Ray RLlib 一起使用。

5. 测试和调试：主函数提供了一个简单的方法来测试环境的功能。

这个环境可以作为一个起点，用于开发更复杂的、使用结构化数据的强化学习环境。
它特别适用于那些需要处理多种类型输入的场景，例如机器人控制、游戏 AI 或复杂系统模拟。

"""
