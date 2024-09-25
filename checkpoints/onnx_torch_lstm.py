# @OldAPIStack

import numpy as np
import onnxruntime

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import add_rllib_example_script_args, check
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

torch, _ = try_import_torch()

parser = add_rllib_example_script_args()
parser.set_defaults(num_env_runners=1)


class ONNXCompatibleWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super(ONNXCompatibleWrapper, self).__init__()
        self.original_model = original_model

    def forward(self, a, b0, b1, c):
        # Convert the separate tensor inputs back into the list format
        # expected by the original model's forward method.
        b = [b0, b1]
        ret = self.original_model({"obs": a}, b, c)
        # results, state_out_0, state_out_1
        return ret[0], ret[1][0], ret[1][1]


if __name__ == "__main__":
    args = parser.parse_args()

    assert (
        not args.enable_new_api_stack
    ), "Must NOT set --enable-new-api-stack when running this script!"

    ray.init(local_mode=args.local_mode)

    # Configure our PPO Algorithm.
    config = (
        ppo.PPOConfig()
        # ONNX is not supported by RLModule API yet.
        .api_stack(
            enable_rl_module_and_learner=args.enable_new_api_stack,
            enable_env_runner_and_connector_v2=args.enable_new_api_stack,
        )
        .environment("CartPole-v1")
        .env_runners(num_env_runners=args.num_env_runners)
        .training(model={"use_lstm": True})
    )

    B = 3
    T = 5
    LSTM_CELL = 256

    # Input data for a python inference forward call.
    test_data_python = {
        "obs": np.random.uniform(0, 1.0, size=(B * T, 4)).astype(np.float32),
        "state_ins": [
            np.random.uniform(0, 1.0, size=(B, LSTM_CELL)).astype(np.float32),
            np.random.uniform(0, 1.0, size=(B, LSTM_CELL)).astype(np.float32),
        ],
        "seq_lens": np.array([T] * B, np.float32),
    }
    # Input data for the ONNX session.
    test_data_onnx = {
        "obs": test_data_python["obs"],
        "state_in_0": test_data_python["state_ins"][0],
        "state_in_1": test_data_python["state_ins"][1],
        "seq_lens": test_data_python["seq_lens"],
    }

    # Input data for compiling the ONNX model.
    test_data_onnx_input = convert_to_torch_tensor(test_data_onnx)

    # Initialize a PPO Algorithm.
    algo = config.build()

    # You could train the model here
    # algo.train()

    # Let's run inference on the torch model
    policy = algo.get_policy()
    result_pytorch, _ = policy.model(
        {
            "obs": torch.tensor(test_data_python["obs"]),
        },
        [
            torch.tensor(test_data_python["state_ins"][0]),
            torch.tensor(test_data_python["state_ins"][1]),
        ],
        torch.tensor(test_data_python["seq_lens"]),
    )

    # Evaluate tensor to fetch numpy array
    result_pytorch = result_pytorch.detach().numpy()

    # Wrap the actual ModelV2 with the torch wrapper above to make this all work with
    # LSTMs (extra `state` in- and outputs and `seq_lens` inputs).
    onnx_compatible = ONNXCompatibleWrapper(policy.model)
    exported_model_file = "model.onnx"
    input_names = [
        "obs",
        "state_in_0",
        "state_in_1",
        "seq_lens",
    ]

    # This line will export the model to ONNX.
    torch.onnx.export(
        onnx_compatible,
        tuple(test_data_onnx_input[n] for n in input_names),
        exported_model_file,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=[
            "output",
            "state_out_0",
            "state_out_1",
        ],
        dynamic_axes={k: {0: "batch_size"} for k in input_names},
    )
    # Start an inference session for the ONNX model.
    session = onnxruntime.InferenceSession(exported_model_file, None)
    result_onnx = session.run(["output"], test_data_onnx)

    # These results should be equal!
    print("PYTORCH", result_pytorch)
    print("ONNX", result_onnx[0])

    check(result_pytorch, result_onnx[0])
    print("Model outputs are equal. PASSED")

""" 
`onnx_torch_lstm.py` 这个脚本展示了如何将带有 LSTM (Long Short-Term Memory) 层的 PyTorch 模型导出为 ONNX 格式，并验证导出模型的正确性。让我们详细解析这个脚本：

1. 导入和设置：
   - 导入必要的库，包括 Ray、RLlib、PyTorch、ONNX 运行时等。
   - 设置命令行参数解析器。

2. ONNXCompatibleWrapper 类：
   - 这是一个自定义的 PyTorch 模块，用于包装原始模型。
   - 它的 forward 方法适配了 ONNX 导出所需的输入格式，处理 LSTM 的状态输入和序列长度。

3. 主程序：
   - 初始化 Ray。
   - 配置 PPO 算法，使用 LSTM 模型。
   - 设置批次大小 (B)、时间步长 (T) 和 LSTM 单元数 (LSTM_CELL)。

4. 数据准备：
   - 创建用于 Python 推理和 ONNX 会话的测试数据。
   - 包括观察值 (obs)、LSTM 状态 (state_ins) 和序列长度 (seq_lens)。

5. 模型初始化和推理：
   - 初始化 PPO 算法（没有实际训练）。
   - 使用 PyTorch 模型进行推理，获取结果。

6. ONNX 导出：
   - 使用 ONNXCompatibleWrapper 包装原始模型。
   - 使用 torch.onnx.export 将模型导出为 ONNX 格式。
   - 设置输入名称、输出名称和动态轴。

7. ONNX 推理：
   - 使用 ONNX 运行时加载导出的模型。
   - 使用相同的测试数据进行 ONNX 模型推理。

8. 结果比较：
   - 打印 PyTorch 和 ONNX 模型的输出结果。
   - 使用 check 函数比较两个模型的输出是否相等。

主要特点：
1. 处理 LSTM 模型：展示了如何处理带有状态的复杂模型。
2. 适配 ONNX 格式：使用包装器类来适配 ONNX 所需的输入输出格式。
3. 动态批次大小：通过设置动态轴，允许在推理时使用不同的批次大小。
4. 验证正确性：通过比较 PyTorch 和 ONNX 模型的输出，确保导出过程的正确性。

这个脚本对于需要将带有 LSTM 层的强化学习模型部署到不同环境的开发者特别有用。它展示了如何处理更复杂的模型结构，并确保在不同框架间的一致性。

PYTORCH [[ 0.16219775 -0.07942449]
 [ 0.11323316  0.15419194]
 [ 0.05051551  0.11130264]
 [-0.08302459  0.24144702]
 [-0.16492157  0.25229347]
 [ 0.17739937  0.21555842]
 [ 0.05350668  0.18631954]
 [-0.04083852  0.08997086]
 [-0.04131908  0.2773131 ]
 [-0.12134071  0.20679446]
 [ 0.07775923  0.11531548]
 [-0.09070934  0.19675398]
 [-0.14580925  0.18927266]
 [-0.24244787  0.20543426]
 [-0.22802067  0.20339462]]
ONNX [[ 0.1621977  -0.0794245 ]
 [ 0.11323325  0.15419188]
 [ 0.05051553  0.1113026 ]
 [-0.08302461  0.24144697]
 [-0.16492149  0.2522934 ]
 [ 0.17739947  0.21555847]
 [ 0.0535066   0.18631962]
 [-0.0408385   0.08997085]
 [-0.04131903  0.27731317]
 [-0.12134068  0.2067944 ]
 [ 0.07775921  0.11531552]
 [-0.09070938  0.19675393]
 [-0.14580926  0.18927269]
 [-0.2424478   0.20543426]
 [-0.22802067  0.20339467]]
Model outputs are equal. PASSED

这个运行结果展示了 PyTorch LSTM 模型和导出后的 ONNX 模型在相同输入下的输出比较。让我们详细分析这个结果：

1. 输出格式：
   - PyTorch 和 ONNX 输出都是 2D 数组，形状为 [15, 2]。
   - 15 行对应于批次大小 (B=3) 和时间步长 (T=5) 的乘积。
   - 每行有 2 个值，可能代表不同动作的概率或值估计。

2. 数值比较：
   - PyTorch 和 ONNX 模型的输出在数值上非常接近，几乎完全相同。
   - 差异主要出现在小数点后的第 6-7 位，这是由于不同框架间的浮点数计算可能存在微小差异。

3. 具体例子：
   - PyTorch: [ 0.16219775 -0.07942449]
   - ONNX:    [ 0.1621977  -0.0794245 ]
   可以看到差异非常小，在实际应用中可以忽略不计。

4. 值的范围：
   - 输出值的范围大约在 -0.25 到 0.28 之间。
   - 这个范围表明模型输出可能经过了某种激活函数（如 tanh）的处理。

5. 时间序列特征：
   - 可以观察到输出值随时间步的变化，这反映了 LSTM 模型捕捉时间序列特征的能力。

6. 一致性验证：
   - 程序最后输出 "Model outputs are equal. PASSED"，表明两个模型的输出在允许的误差范围内被认为是相等的。

7. LSTM 特性：
   - 输出显示了 LSTM 模型处理序列数据的能力，每个时间步的输出都受到之前时间步的影响。

8. 模型行为：
   - 输出值的变化表明模型对不同的输入状态产生了不同的响应。
   - 有些输出值接近于零，而有些则相对较大，这可能反映了模型对某些特征的敏感度。

9. 导出成功：
   - 结果证明了带有 LSTM 层的 PyTorch 模型成功地被导出为 ONNX 格式，并且在 ONNX 运行时中能够正确运行。

10. 精度保持：
    - ONNX 模型保持了与原始 PyTorch 模型几乎相同的精度，这对于模型部署和跨平台使用非常重要。

总结：
这个输出结果验证了复杂的 LSTM 模型从 PyTorch 到 ONNX 的成功转换。两个模型在数值精度上非常接近，任何微小的差异都在可接受的范围内。这意味着导出的 ONNX 模型可以在支持 ONNX 的其他平台或设备上使用，而不会影响模型的性能或预测结果。这对于需要在不同环境中部署复杂的时序强化学习模型的开发者来说是非常有价值的结果。
"""
