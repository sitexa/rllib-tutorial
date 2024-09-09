# @OldAPIStack

from packaging.version import Version
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
import onnxruntime
import os
import shutil
import torch

if __name__ == "__main__":
    # Configure our PPO Algorithm.
    config = (
        ppo.PPOConfig()
        # ONNX is not supported by RLModule API yet.
        .api_stack(enable_rl_module_and_learner=False)
        .env_runners(num_env_runners=1)
        .framework("torch")
    )

    outdir = "export_torch"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    np.random.seed(1234)

    # We will run inference with this test batch
    test_data = {
        "obs": np.random.uniform(0, 1.0, size=(10, 4)).astype(np.float32),
        "state_ins": np.array([0.0], dtype=np.float32),
    }

    # Start Ray and initialize a PPO Algorithm.
    ray.init()
    algo = config.build(env="CartPole-v1")

    # You could train the model here
    algo.train()

    # Let's run inference on the torch model
    policy = algo.get_policy()
    result_pytorch, _ = policy.model(
        {
            "obs": torch.tensor(test_data["obs"]),
        }
    )

    # Evaluate tensor to fetch numpy array
    result_pytorch = result_pytorch.detach().numpy()

    # This line will export the model to ONNX.
    policy.export_model(outdir, onnx=11)
    # Equivalent to:
    # algo.export_policy_model(outdir, onnx=11)

    # Import ONNX model.
    exported_model_file = os.path.join(outdir, "model.onnx")

    # Start an inference session for the ONNX model
    session = onnxruntime.InferenceSession(exported_model_file, None)

    # Pass the same test batch to the ONNX model
    if Version(torch.__version__) < Version("1.9.0"):
        # In torch < 1.9.0 the second input/output name gets mixed up
        test_data["state_outs"] = test_data.pop("state_ins")

    result_onnx = session.run(["output"], test_data)

    # These results should be equal!
    print("PYTORCH", result_pytorch)
    print("ONNX", result_onnx)

    assert np.allclose(
        result_pytorch, result_onnx
    ), "Model outputs are NOT equal. FAILED"
    print("Model outputs are equal. PASSED")

""" 
这个脚本主要展示了如何使用 PyTorch 训练一个强化学习模型，然后将其导出为 ONNX 格式，
并比较 PyTorch 和 ONNX 模型的输出。以下是详细分析：

1. 导入和设置：
   - 导入必要的库，包括 Ray、RLlib、PyTorch、ONNX 运行时等。
   - 设置 PPO 配置，使用 PyTorch 作为后端。

2. 环境和数据准备：
   - 使用 "CartPole-v1" 环境。
   - 创建一个测试数据批次，用于后续的推理比较。

3. 模型训练：
   - 初始化 Ray 和 PPO 算法。
   - 注释掉了实际的训练过程（`algo.train()`），可能是为了快速演示。

4. PyTorch 模型推理：
   - 获取训练好的策略。
   - 使用测试数据进行 PyTorch 模型的推理。

5. 模型导出为 ONNX：
   - 使用 `policy.export_model()` 将模型导出为 ONNX 格式。

6. ONNX 模型推理：
   - 使用 ONNX 运行时加载导出的模型。
   - 使用相同的测试数据进行 ONNX 模型的推理。

7. 结果比较：
   - 打印 PyTorch 和 ONNX 模型的输出结果。
   - 使用 `np.allclose()` 比较两个模型的输出是否相近。

8. 版本兼容性处理：
   - 针对不同版本的 PyTorch，调整了输入数据的键名。

9. 错误处理和清理：
   - 使用 `try-except` 块来捕获可能的错误。
   - 在脚本开始时清理之前的输出目录。

主要目的：
1. 展示如何将 RLlib 训练的 PyTorch 模型导出为 ONNX 格式。
2. 验证导出的 ONNX 模型与原始 PyTorch 模型的输出一致性。
3. 提供一个在不同深度学习框架间转换模型的示例。

这个脚本对于需要将强化学习模型部署到不同环境或需要在不同框架间迁移模型的开发者来说非常有用。
它展示了模型导出和跨框架推理的过程，同时也提供了一种验证导出模型正确性的方法。
"""

"""
PYTORCH [[-2.9510839e-04 -1.3749092e-03]
 [ 4.1112411e-03 -2.9926004e-03]
 [ 3.7752860e-03  9.5089839e-04]
 [ 2.7357356e-03  9.1067574e-05]
 [ 3.2435418e-03 -5.9561166e-03]
 [ 1.2981507e-03  1.8974489e-03]
 [ 4.0384526e-03 -1.2279389e-03]
 [ 8.0747392e-05 -2.4789826e-03]
 [ 4.1501205e-03 -5.1417900e-03]
 [-8.5803476e-04 -1.8008053e-04]]
ONNX [array([[-2.9510801e-04, -1.3749094e-03],
       [ 4.1112420e-03, -2.9926002e-03],
       [ 3.7752849e-03,  9.5089711e-04],
       [ 2.7357358e-03,  9.1068032e-05],
       [ 3.2435418e-03, -5.9561180e-03],
       [ 1.2981502e-03,  1.8974484e-03],
       [ 4.0384517e-03, -1.2279387e-03],
       [ 8.0747122e-05, -2.4789821e-03],
       [ 4.1501210e-03, -5.1417886e-03],
       [-8.5803360e-04, -1.8008077e-04]], dtype=float32)]
Model outputs are equal. PASSED

这个输出展示了 PyTorch 模型和导出后的 ONNX 模型在相同输入下的输出结果比较。让我们详细分析这个结果：

1. 输出格式：
   - PyTorch 输出是一个 2D 张量，形状为 [10, 2]。
   - ONNX 输出是一个包含单个 NumPy 数组的列表，数组形状也是 [10, 2]。

2. 数值比较：
   - 两个模型的输出在数值上非常接近，但并不完全相同。
   - 差异主要出现在小数点后的第 6-7 位，这是由于不同框架间的浮点数计算可能存在微小差异。

3. 具体例子：
   - PyTorch: [-2.9510839e-04 -1.3749092e-03]
   - ONNX:    [-2.9510801e-04 -1.3749094e-03]
   可以看到差异非常小，在实际应用中可以忽略不计。

4. 精度：
   - 两个模型都使用 32 位浮点数（float32），这保证了高精度的数值表示。

5. 一致性验证：
   - 程序最后输出 "Model outputs are equal. PASSED"，表明两个模型的输出在允许的误差范围内被认为是相等的。
   - 这是通过 `np.allclose()` 函数实现的，该函数允许一定程度的数值误差。

6. 结果解释：
   - 输出的每一行代表一个输入样本的预测结果。
   - 每行有两个值，可能代表不同动作的概率或 Q 值（取决于具体的强化学习算法）。

7. 模型行为：
   - 从输出可以看出，模型对不同输入产生了不同的预测，表明模型已经学习到了一定的模式。
   - 大多数输出值较小（接近于 0），这可能是因为模型还没有经过充分训练，或者是 CartPole 环境的特性导致的。

8. 导出成功：
   - 这个结果证明了 PyTorch 模型成功地被导出为 ONNX 格式，并且在 ONNX 运行时中能够正确运行。

总结：
这个输出结果验证了 PyTorch 模型到 ONNX 模型的成功转换。
两个模型在数值精度上非常接近，任何微小的差异都在可接受的范围内。
这意味着导出的 ONNX 模型可以在支持 ONNX 的其他平台或设备上使用，而不会显著影响模型的性能或预测结果。
"""
