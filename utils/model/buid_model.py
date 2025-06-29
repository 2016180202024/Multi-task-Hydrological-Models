import importlib
import os
import numpy as np
import torch
from torch import nn

from configs.data_config.dataset_config import DataShapeConfig
from configs.data_config.project_config import ProjectConfig


def read_data_shape_by_model_name(dir_name):
    data_shape = str.split(str.split(dir_name, '@')[1], '_')[0]
    len_shape = str.split(str.split(data_shape, ',')[0][1:], '-')
    size_shape = str.split(str.split(data_shape, ',')[1][:-1], '-')
    past_len, pred_len = int(len_shape[0]), int(len_shape[1])
    src_size, pred_size = int(size_shape[0]), int(size_shape[1])
    return {
        'past_len': past_len,
        'pred_len': pred_len,
        'src_size': src_size,
        'pred_size': pred_size
    }


def build_model_with_name(used_model):
    # 先修改model_config
    model_congfigs = importlib.import_module(f"configs.model_config.{used_model}_config")
    ModelConfig = getattr(model_congfigs, f"{used_model}Config")
    model_config = ModelConfig()
    # 后新建model
    model = build_model_with_name_config(used_model, model_config)
    return model


def build_model_with_name_datashape(used_model, datashape, features_name):
    # 先修改model_config
    model_congfigs = importlib.import_module(f"configs.model_config.{used_model}_config")
    ModelConfig = getattr(model_congfigs, f"{used_model}Config")
    data_config = DataShapeConfig(datashape['past_len'], datashape['pred_len'], features_name)
    model_config = ModelConfig(data_config)
    # 后新建model
    model = build_model_with_name_config(used_model, model_config)
    return model


def build_model_with_name_config(used_model, model_config):
    # 后新建model
    models = importlib.import_module("models")
    Model = getattr(models, used_model)
    model = Model(model_config)
    return model


def load_model_by_path(model_path):
    dir_name = os.path.basename(os.path.dirname(model_path))
    # 加载模型的输入数据类型
    data_shape = str.split(str.split(dir_name, '@')[1], '_')[0]
    len_shape = str.split(str.split(data_shape, ',')[0][1:], '-')
    size_shape = str.split(str.split(data_shape, ',')[1][:-1], '-')
    past_len, pred_len = int(len_shape[0]), int(len_shape[1])
    all_len = past_len + pred_len
    src_size, pred_size = int(size_shape[0]), int(size_shape[1])
    # 加载模型
    model_name = str.split(dir_name, '_')[0]
    datashape = {'src_len': all_len, 'src_size': src_size, 'past_len': past_len,
                 'pred_len': pred_len, 'tgt_size': pred_size}
    model = build_model_with_name_datashape(model_name, datashape)
    model.state_dict = torch.load(model_path)
    if ProjectConfig.multi_gpu:
        model = nn.DataParallel(model)
    return model, datashape


# 为所有特征列标准化
def normalization(feature: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    shape = feature.shape
    mean = mean.reshape((1, 1, len(mean))).repeat(shape[0], axis=0).repeat(shape[1], axis=1)
    std = std.reshape((1, 1, len(std))).repeat(shape[0], axis=0).repeat(shape[1], axis=1)
    feature = feature - mean
    # 避免某些特征（glacier_extent和wetlands_extent）的std为0
    feature = np.divide(feature, std, out=feature, where=(std != 0))
    return feature
