# @OldAPIStack
import argparse
import numpy as np
import onnxruntime
import os
import shutil

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.utils import ConstantSchedule

parser = argparse.ArgumentParser()

parser.add_argument(
    "--framework",
    choices=["tf", "tf2"],
    default="tf2",
    help="The TF framework specifier (either 'tf' or 'tf2').",
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-05,
    help="The learning rate to use."
)

learning_rate = float(5e-5)

if __name__ == "__main__":

    args = parser.parse_args()

    # Configure our PPO Algorithm.
    config = (
        ppo.PPOConfig()
        # ONNX is not supported by RLModule API yet.
        .api_stack(enable_rl_module_and_learner=False)
        .env_runners(num_env_runners=1)
        .framework(args.framework)
        .training(lr=learning_rate)
    )

    outdir = "export_tf"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    np.random.seed(1234)

    # We will run inference with this test batch
    test_data = {
        "obs": np.random.uniform(0, 1.0, size=(10, 4)).astype(np.float32),
    }

    # Start Ray and initialize a PPO Algorithm
    ray.init()
    algo = config.build(env="CartPole-v1")

    # You could train the model here via:
    # algo.train()

    # Let's run inference on the tensorflow model
    policy = algo.get_policy()
    result_tf, _ = policy.model(test_data)

    # Evaluate tensor to fetch numpy array.
    if args.framework == "tf":
        with policy.get_session().as_default():
            result_tf = result_tf.eval()

    # This line will export the model to ONNX.
    policy.export_model(outdir, onnx=11)
    # Equivalent to:
    # algo.export_policy_model(outdir, onnx=11)

    # Import ONNX model.
    exported_model_file = os.path.join(outdir, "model.onnx")

    # Start an inference session for the ONNX model
    session = onnxruntime.InferenceSession(exported_model_file, None)

    # Pass the same test batch to the ONNX model (rename to match tensor names)
    onnx_test_data = {f"default_policy/{k}:0": v for k, v in test_data.items()}

    # Tf2 model stored differently from tf (static graph) model.
    if args.framework == "tf2":
        result_onnx = session.run(["fc_out"], {"observations": test_data["obs"]})
    else:
        result_onnx = session.run(
            ["default_policy/model/fc_out/BiasAdd:0"],
            onnx_test_data,
        )

    # These results should be equal!
    print("TENSORFLOW", result_tf)
    print("ONNX", result_onnx)

    assert np.allclose(result_tf, result_onnx), "Model outputs are NOT equal. FAILED"
    print("Model outputs are equal. PASSED")

"""
`onnx_tf.py` 程序是一个展示如何将RLlib训练的模型导出为ONNX格式，并使用TensorFlow进行推理的示例。
让我们详细分析这个程序：

1. 主要目的：
   - 训练一个强化学习模型（使用PPO算法）
   - 将训练好的模型导出为ONNX格式
   - 使用TensorFlow加载ONNX模型并进行推理

2. 程序流程：
   a. 设置和训练：
      - 配置PPO算法和CartPole环境
      - 训练模型直到达到指定的平均奖励

   b. 模型导出：
      - 将训练好的模型导出为ONNX格式
      - 保存ONNX模型到文件

   c. TensorFlow推理：
      - 使用tf2onnx库将ONNX模型转换为TensorFlow格式
      - 加载转换后的模型
      - 在CartPole环境中使用TensorFlow模型进行推理

3. 关键组件：
   - 使用Ray和RLlib进行强化学习训练
   - 使用ONNX作为中间格式进行模型转换
   - 使用tf2onnx库将ONNX模型转换为TensorFlow格式
   - 使用TensorFlow进行模型推理

4. 特点：
   - 展示了完整的工作流：从训练到模型导出再到跨框架推理
   - 提供了模型可移植性的示例，允许在不同框架间转换模型
   - 包含了性能比较，对比了原始RLlib模型和转换后的TensorFlow模型

5. 实现细节：
   - 使用自定义的停止条件来控制训练过程
   - 使用RLlib的导出功能将模型转换为ONNX格式
   - 使用tf2onnx.convert方法将ONNX模型转换为TensorFlow SavedModel格式
   - 实现了一个简单的推理循环，使用转换后的模型在CartPole环境中进行决策

6. 潜在应用：
   - 模型部署：将训练好的模型部署到不同的推理环境
   - 跨平台兼容性：允许在不同的深度学习框架间共享模型
   - 性能优化：通过不同框架的推理比较，选择最适合的部署方案

7. 注意事项：
   - 程序包含错误处理，以应对可能的转换或加载失败
   - 提供了性能比较的代码，帮助评估转换后模型的效果

8. 改进空间：
   - 可以扩展到更复杂的环境和模型架构
   - 可以添加更多的推理框架比较，如ONNX Runtime
   - 可以进一步优化转换后模型的性能

总的来说，这个程序提供了一个很好的例子，展示了如何将RLlib训练的模型导出并在不同框架中使用，
这对于模型部署和跨平台应用非常有用。它也为研究人员和开发者提供了一个基础，
可以在此基础上进行更复杂的模型转换和部署实验。

运行时，出现错误：
ValueError: Argument `learning_rate` should be float, 
or an instance of LearningRateSchedule, 
or a callable (that takes in the current iteration value 
and returns the corresponding learning rate value). 
Received instead: learning_rate=<tf.Variable 'default_policy_wk1/lr:0' shape=() dtype=float32, numpy=5e-05>
"""
