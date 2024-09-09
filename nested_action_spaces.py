from gymnasium.spaces import Dict, Tuple, Box, Discrete, MultiDiscrete

from ray.tune.registry import register_env
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.examples.envs.classes.multi_agent import (
    MultiAgentNestedSpaceRepeatAfterMeEnv,
)
from ray.rllib.examples.envs.classes.nested_space_repeat_after_me_env import (
    NestedSpaceRepeatAfterMeEnv,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls

# Read in common example script command line arguments.
parser = add_rllib_example_script_args(default_timesteps=200000, default_reward=-500.0)

if __name__ == "__main__":
    args = parser.parse_args()

    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"


    # Define env-to-module-connector pipeline for the new stack.
    def _env_to_module_pipeline(env):
        return FlattenObservations(multi_agent=args.num_agents > 0)


    # Register our environment with tune.
    if args.num_agents > 0:
        register_env(
            "env",
            lambda c: MultiAgentNestedSpaceRepeatAfterMeEnv(
                config=dict(c, **{"num_agents": args.num_agents})
            ),
        )
    else:
        register_env("env", lambda c: NestedSpaceRepeatAfterMeEnv(c))

    # Define the AlgorithmConfig used.
    # 灵活地配置复杂的强化学习环境和算法参数。
    base_config = (
        get_trainable_cls(args.algo)  # 算法的可训练类
        .get_default_config()  # 处法的黙认配置
        .environment(
            "env",  # 之前注册的名为env的环境
            env_config={
                "space": Dict(  # 设置复杂的嵌套观察空间
                    {
                        "a": Tuple(  # 包含一个字典元组
                            [Dict({"d": Box(-15.0, 3.0, ()), "e": Discrete(3)})]
                        ),
                        "b": Box(-10.0, 10.0, (2,)),  # 2维连续空间
                        "c": MultiDiscrete([3, 3]),  # 多维离散空间
                        "d": Discrete(2),  # 2元素离散空间
                    }
                ),
                "episode_len": 100,  # 回合长度100
            },
        )
        .env_runners(env_to_module_connector=_env_to_module_pipeline)  # 环境运行器
        # No history in Env (bandit problem).
        .training(  # 配置训练参数
            gamma=0.0,
            lr=0.0005,
            model=(
                {} if not args.enable_new_api_stack else {"uses_new_env_runners": True}
            ),
        )
    )

    # Add a simple multi-agent setup.
    if args.num_agents > 0:
        base_config.multi_agent(
            policies={f"p{i}" for i in range(args.num_agents)},
            policy_mapping_fn=lambda aid, *a, **kw: f"p{aid}",
        )

    # Fix some PPO-specific settings.
    if args.algo == "PPO":
        base_config.training(
            # We don't want high entropy in this Env.
            entropy_coeff=0.00005,  # 设置熵系数为一个很小的值
            num_sgd_iter=4,  # 每次更新时进行随机梯度下降的迭代次数
            vf_loss_coeff=0.01,  # 设置值函数损失的系数
        )

    # Run everything as configured.
    run_rllib_example_script_experiment(base_config, args)

"""
这个程序是一个使用Ray RLlib框架的强化学习示例脚本。
它主要用于训练一个智能体在具有嵌套动作空间的环境中学习"重复我说的话"的任务。
以下是主要的执行逻辑:

1. 导入必要的库和模块。
2. 设置命令行参数解析器,添加一些默认参数。
3. 主函数逻辑:
  a. 解析命令行参数。
  b. 定义环境到模块的连接器管道,用于扁平化观察空间。
  c. 根据是否为多智能体场景,注册相应的环境。
  d. 配置算法:
    - 设置环境配置,包括一个复杂的嵌套空间结构。
    - 配置训练参数,如学习率、折扣因子等。
    - 如果是多智能体场景,添加多智能体配置。
    - 对PPO算法进行特定调整。
  e. 运行实验。

主要特点:
    - 支持单智能体和多智能体场景。
    - 使用嵌套的观察和动作空间,增加了任务的复杂性。
    - 可以通过命令行参数灵活配置实验设置。
    - 使用Ray的新API栈进行训练。        
"""

"""
+---------------------+------------+-----------------+--------+------------------+------------------------+------------------------+------------------------+
| Trial name          | status     | loc             |   iter |   total time (s) |   num_env_steps_sample |   num_episodes_lifetim |   num_env_steps_traine |
|                     |            |                 |        |                  |             d_lifetime |                      e |             d_lifetime |
|---------------------+------------+-----------------+--------+------------------+------------------------+------------------------+------------------------|
| PPO_env_b5c87_00000 | TERMINATED | 127.0.0.1:61381 |     23 |          63.3299 |                  92000 |                    920 |                  92000 |
+---------------------+------------+-----------------+--------+------------------+------------------------+------------------------+------------------------+

这个运行结果表格提供了关于强化学习实验的重要信息。让我们逐项分析：

1. Trial name (试验名称): PPO_env_b5c87_00000
    - 这表明使用的是PPO（Proximal Policy Optimization）算法
    - "env"可能指的是环境名称
    - "b5c87_00000"可能是一个唯一标识符
2. status (状态): TERMINATED
    - 实验已经结束
3. loc (位置): 127.0.0.1:61381
    - 实验在本地主机上运行，使用61381端口
4. iter (迭代次数): 23
    - 实验进行了23次迭代
5. total time (s) (总时间): 63.3299秒
    - 实验总共运行了约63.3秒
6. num_env_steps_sampled_lifetime (环境步数采样总数): 92000
    - 在整个实验过程中，智能体与环境交互了92000次
7. num_episodes_lifetime (回合总数): 920
    - 实验总共完成了920个回合
8. num_env_steps_trained_lifetime (训练的环境步数总数): 92000
    - 智能体在92000个环境步骤上进行了训练

这些结果表明：
    - 实验运行时间相对较短（约1分钟）
    - 每个回合平均包含100个步骤（92000 / 920）
    - 采样的步数和训练的步数相同，说明所有采样的数据都用于了训练
    - 实验成功完成，没有出现错误或中断

总的来说，这似乎是一个成功的短期PPO实验，可能是用于快速测试或初步调优。        
"""