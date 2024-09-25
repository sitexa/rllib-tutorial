#!/usr/bin/env python

# @OldAPIStack

import numpy as np
import os
import ray

from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import get_trainable_cls

tf1, tf, tfv = try_import_tf()

ray.init()


def train_and_export_policy_and_model(algo_name, num_steps, model_dir, ckpt_dir):
    cls = get_trainable_cls(algo_name)
    config = cls.get_default_config()
    # This Example is only for tf.
    config.framework("tf")
    # Set exporting native (DL-framework) model files to True.
    config.export_native_model_files = True
    config.env = "CartPole-v1"
    alg = config.build()
    for _ in range(num_steps):
        alg.train()

    # Export Policy checkpoint.
    alg.export_policy_checkpoint(ckpt_dir)
    # Export tensorflow keras Model for online serving
    alg.export_policy_model(model_dir)


def restore_saved_model(export_dir):
    signature_key = (
        tf1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    )
    g = tf1.Graph()
    with g.as_default():
        with tf1.Session(graph=g) as sess:
            meta_graph_def = tf1.saved_model.load(
                sess, [tf1.saved_model.tag_constants.SERVING], export_dir
            )
            print("Model restored!")
            print("Signature Def Information:")
            print(meta_graph_def.signature_def[signature_key])
            print("You can inspect the model using TensorFlow SavedModel CLI.")
            print("https://www.tensorflow.org/guide/saved_model")


def restore_policy_from_checkpoint(export_dir):
    # Load the model from the checkpoint.
    policy = Policy.from_checkpoint(export_dir)
    # Perform a dummy (CartPole) forward pass.
    test_obs = np.array([0.1, 0.2, 0.3, 0.4])
    results = policy.compute_single_action(test_obs)
    # Check results for correctness.
    assert len(results) == 3
    assert results[0].shape == ()  # pure single action (int)
    assert results[1] == []  # RNN states
    assert results[2]["action_dist_inputs"].shape == (2,)  # categorical inputs


if __name__ == "__main__":
    algo = "PPO"
    model_dir = os.path.join(ray._private.utils.get_user_temp_dir(), "model_export_dir")
    ckpt_dir = os.path.join(ray._private.utils.get_user_temp_dir(), "ckpt_export_dir")
    num_steps = 1
    train_and_export_policy_and_model(algo, num_steps, model_dir, ckpt_dir)
    restore_saved_model(model_dir)
    restore_policy_from_checkpoint(ckpt_dir)


"""
分析一下这个程序:

这是一个使用Ray RLlib框架训练和导出强化学习模型的Python脚本。主要功能包括:

1. 训练并导出策略和模型:
```python:checkpoints/cartpole_dqn_export.py
def train_and_export_policy_and_model(algo_name, num_steps, model_dir, ckpt_dir):
    # ... 训练算法并导出策略检查点和模型
```

2. 恢复保存的模型:
```python:checkpoints/cartpole_dqn_export.py
def restore_saved_model(export_dir):
    # ... 从导出目录加载并恢复SavedModel
```

3. 从检查点恢复策略:
```python:checkpoints/cartpole_dqn_export.py
def restore_policy_from_checkpoint(export_dir):
    # ... 从检查点加载策略并执行测试推理
```

4. 主函数:
```python:checkpoints/cartpole_dqn_export.py
if __name__ == "__main__":
    # ... 设置参数并调用上述函数
```

该脚本使用PPO算法训练了一个CartPole环境的模型,训练步数很少(仅1步),主要目的是演示模型的导出和加载过程。它展示了如何:

1. 训练一个简单的强化学习模型
2. 导出模型和策略检查点
3. 加载保存的模型
4. 从检查点恢复策略并进行推理

这个脚本对于理解Ray RLlib框架中模型的训练、导出和加载流程很有帮助。


说明policy_checkpoint和policy_model之间的区别:

````python:checkpoints/cartpole_dqn_export.py
    # Export Policy checkpoint.
    alg.export_policy_checkpoint(ckpt_dir)
    # Export tensorflow keras Model for online serving
    alg.export_policy_model(model_dir)
````

这两行代码分别导出了策略检查点(policy checkpoint)和策略模型(policy model)。

1. Policy Checkpoint (策略检查点):
   - `alg.export_policy_checkpoint(ckpt_dir)` 导出策略检查点。
   - 策略检查点包含了完整的训练状态,包括模型权重、优化器状态等。
   - 主要用于恢复训练或在相同环境中继续训练。
   - 通常格式为Ray特定的格式,不易于直接部署。

2. Policy Model (策略模型):
   - `alg.export_policy_model(model_dir)` 导出策略模型。
   - 策略模型只包含模型结构和权重,不包含训练状态。
   - 主要用于模型部署和推理。
   - 导出为TensorFlow SavedModel格式,更适合在生产环境中使用。

主要区别:

1. 用途:
   - Checkpoint用于恢复训练状态。
   - Model用于部署和推理。

2. 包含内容:
   - Checkpoint包含完整训练状态。
   - Model只包含模型结构和权重。

3. 格式:
   - Checkpoint通常是Ray特定格式。
   - Model是标准的TensorFlow SavedModel格式。

4. 可移植性:
   - Checkpoint主要在Ray环境中使用。
   - Model可以在任何支持TensorFlow的环境中使用。

总之,策略检查点更适合于训练过程中的保存和恢复,而策略模型更适合于模型的部署和使用。
"""
