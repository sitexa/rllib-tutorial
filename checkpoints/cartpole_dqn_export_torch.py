import os
from typing import Union, Dict

import numpy as np
import ray
import torch
import tempfile
import logging
import argparse

from ray.rllib.policy.policy import Policy
from ray.tune.registry import get_trainable_cls

"""
这个程序是一个使用Ray RLlib框架训练和导出CartPole环境的强化学习模型的示例。

主要功能:
1. 训练一个强化学习模型(默认使用DQN算法)来解决CartPole问题。
2. 将训练好的模型导出为PyTorch格式。
3. 从导出的文件中恢复模型。

主要函数:
- train_and_export_policy_and_model: 训练模型并导出策略和模型文件。
- restore_saved_model: 从导出的文件中恢复模型。

使用方法:
1. 确保已安装所需的依赖包(ray, torch, numpy等)。
2. 运行脚本,指定所需的参数(如算法名称、训练步数等)。
3. 脚本将训练模型,并将其导出到指定目录。
4. 之后,可以使用restore_saved_model函数来加载导出的模型。

注意事项:
- 本脚本使用PyTorch作为后端框架。
- 默认使用CartPole-v1作为训练环境。
- 确保有足够的磁盘空间来存储导出的模型文件。
"""


def train_and_export_policy_and_model(algo_name: str, num_steps: int, model_dir: str, ckpt_dir: str) -> None:
    """
    训练强化学习模型并导出策略和模型。

    Args:
        algo_name (str): 要使用的算法名称。
        num_steps (int): 训练的步数。
        model_dir (str): 导出模型的目录路径。
        ckpt_dir (str): 导出检查点的目录路径。

    Raises:
        Exception: 训练或导出过程中发生的任何异常。

    """
    try:
        cls = get_trainable_cls(algo_name)
        config = cls.get_default_config()
        config.framework("torch")
        config.export_native_model_files = True
        config.env = "CartPole-v1"
        algo = config.build()
        for _ in range(num_steps):
            algo.train()

        # Export Policy checkpoint
        algo.export_policy_checkpoint(ckpt_dir)
        # Export PyTorch model for online serving
        algo.export_policy_model(model_dir)
        logging.info(f"==========Model and checkpoint exported successfully to {model_dir} and {ckpt_dir}")
    except Exception as e:
        logging.error(f"==========Error during training and exporting: {str(e)}")
        raise

    finally:
        pass


def restore_saved_model(export_dir: str) -> Union[torch.nn.Module, Dict[str, torch.Tensor]]:
    """
    从指定目录加载保存的PyTorch模型。

    Args:
        export_dir (str): 包含保存模型的目录路径。

    Returns:
        Union[torch.nn.Module, Dict[str, torch.Tensor]]: 加载的模型对象或状态字典。

    Raises:
        FileNotFoundError: 如果在指定路径找不到模型文件。

    """
    model_path = os.path.join(export_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 加载模型
    loaded_model = torch.load(model_path, map_location=torch.device('cpu'))

    logging.info("==========Model restored!")

    # 检查加载的对象类型
    if isinstance(loaded_model, torch.nn.Module):
        # 如果加载的是整个模型，获取其状态字典
        state_dict = loaded_model.state_dict()
        logging.info(f"==========Loaded a full model. Model state dict keys: {state_dict.keys()}")
    elif isinstance(loaded_model, dict):
        # 如果加载的已经是状态字典
        state_dict = loaded_model
        logging.info(f"==========Loaded a state dict. State dict keys: {state_dict.keys()}")
    else:
        logging.warning(f"==========Unexpected type loaded: {type(loaded_model)}")
        state_dict = None

    if state_dict:
        logging.info("==========You can now use this state_dict to initialize your PyTorch model.")
    else:
        logging.warning("==========Unable to extract state dict from loaded object.")

    return loaded_model  # 返回加载的对象，以便进一步检查或使用


def restore_policy_from_checkpoint(export_dir: str) -> None:
    """
    从检查点恢复策略并执行测试推理。

    Args:
        export_dir (str): 包含策略检查点的目录路径。

    Returns:
        None

    Raises:
        AssertionError: 如果结果不符合预期格式。
    """
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

    # Log the results of the policy restoration and forward pass
    logging.info("==========Policy restored from checkpoint")
    logging.info(f"==========Test observation: {test_obs}")
    logging.info(f"==========Results of compute_single_action:")
    logging.info(f"  Action: {results[0]}")
    logging.info(f"  RNN states: {results[1]}")
    logging.info(f"  Action distribution inputs shape: {results[2]['action_dist_inputs'].shape}")

    # Additional logging for verification
    logging.info("==========Assertions passed:")
    logging.info(f"  Length of results is 3: {len(results) == 3}")
    logging.info(f"  Action is a scalar: {results[0].shape == ()}")
    logging.info(f"  RNN states is an empty list: {results[1] == []}")
    logging.info(f"  Action distribution inputs shape is (2,): {results[2]['action_dist_inputs'].shape == (2,)}")


def continue_training_from_checkpoint(ckpt_dir: str, num_steps: int, new_model_dir: str, new_ckpt_dir: str) -> None:
    """
    从检查点恢复策略并继续训练。

    Args:
        ckpt_dir (str): 包含原始检查点的目录路径。
        num_steps (int): 继续训练的步数。
        new_model_dir (str): 新模型导出的目录路径。
        new_ckpt_dir (str): 新检查点导出的目录路径。

    Raises:
        Exception: 训练或导出过程中发生的任何异常。
    """
    try:
        # 从检查点恢复策略
        policy = Policy.from_checkpoint(ckpt_dir)
        
        # 获取算法配置
        config = policy.config
        logging.info(f"恢复的策略配置: {config}")
        
        # 创建一个新的配置对象
        algo_name = config.get("algo_class", "PPO") 
        logging.info(f"使用的算法: {algo_name}")
        
        algo_cls = get_trainable_cls(algo_name)
        new_config = algo_cls.get_default_config()
        
        # 更新新配置
        for key, value in config.items():
            if key != "framework" and hasattr(new_config, key):
                setattr(new_config, key, value)
                
        # 设置框架和导出选项
        new_config.framework("torch")
        new_config.export_native_model_files = True
        
        # 创建算法实例
        algo = algo_cls(config=new_config)
        
        # 设置算法的策略权重
        policy_weights = policy.get_weights()
        if "default_policy" in policy_weights:
            algo.get_policy("default_policy").set_weights(policy_weights["default_policy"])
        else:
            logging.warning("无法找到 'default_policy' 的权重，跳过权重设置")

        
        # 继续训练
        for _ in range(num_steps):
            algo.train()
        
        # 导出新的策略检查点
        algo.export_policy_checkpoint(new_ckpt_dir)
        # 导出新的PyTorch模型
        algo.export_policy_model(new_model_dir)
        
        logging.info(f"==========继续训练完成。新模型和检查点已导出到 {new_model_dir} 和 {new_ckpt_dir}")
    except Exception as e:
        logging.error(f"==========继续训练过程中出错: {str(e)}")
        raise
    finally:
        pass


if __name__ == "__main__":
    # 设置日志级别为INFO
    logging.basicConfig(level=logging.INFO)
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Train and export PPO model for CartPole")
    parser.add_argument("--num_steps", type=int, default=1, help="Number of training steps")
    parser.add_argument("--model_dir", type=str, default=os.path.join(tempfile.gettempdir(), "model_export_dir"),
                        help="Directory to export the model")
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(tempfile.gettempdir(), "ckpt_export_dir"),
                        help="Directory to export the checkpoint")
    parser.add_argument("--continue_training", action="store_true", help="是否从检查点继续训练")
    parser.add_argument("--continue_steps", type=int, default=1, help="继续训练的步数")
    parser.add_argument("--new_model_dir", type=str, default=os.path.join(tempfile.gettempdir(), "new_model_export_dir"),
                        help="新模型导出目录")
    parser.add_argument("--new_ckpt_dir", type=str, default=os.path.join(tempfile.gettempdir(), "new_ckpt_export_dir"),
                        help="新检查点导出目录")
    
    args = parser.parse_args()

    # 初始化Ray
    with ray.init():
        # 训练并导出策略和模型
        train_and_export_policy_and_model("PPO", args.num_steps, args.model_dir, args.ckpt_dir)
        
        # 恢复保存的模型
        restore_saved_model(args.model_dir)
        
        # 从检查点恢复策略
        restore_policy_from_checkpoint(args.ckpt_dir)

        # 如果指定了继续训练，则从检查点继续训练
        if args.continue_training:
            continue_training_from_checkpoint(args.ckpt_dir, args.continue_steps, args.new_model_dir, args.new_ckpt_dir)
            
            # 验证新训练的模型和检查点
            restore_saved_model(args.new_model_dir)
            restore_policy_from_checkpoint(args.new_ckpt_dir)
