"""
训练并评估单一模型的脚本
"""

import argparse

from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--task', type=str,
                        default='traffic_state_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='GRU', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='METR_LA', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--saved_model', type=str2bool,
                        default=True, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='whether re-train model if the model is trained before')
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # 增加其他可选的参数
    add_general_args(parser)
    # 解析参数 - 使用 parse_known_args 以允许未知参数（如 input_window, output_window）
    args, unknown_args = parser.parse_known_args()
    dict_args = vars(args)
    # 处理未知参数（格式: --key value）
    for i in range(0, len(unknown_args), 2):
        if i + 1 < len(unknown_args):
            key = unknown_args[i].lstrip('--')
            value = unknown_args[i + 1]
            # 尝试转换为 int 或 float，否则保持为字符串
            try:
                if '.' in value:
                    dict_args[key] = float(value)
                else:
                    dict_args[key] = int(value)
            except ValueError:
                # 尝试布尔值（使用 str2bool 函数保持一致性）
                try:
                    dict_args[key] = str2bool(value)
                except (ValueError, argparse.ArgumentTypeError):
                    # 不是布尔值，保持为字符串
                    dict_args[key] = value
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
              config_file=args.config_file, saved_model=args.saved_model,
              train=args.train, other_args=other_args)
