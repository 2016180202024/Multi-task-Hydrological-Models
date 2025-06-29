import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from configs.data_config.dataset_config import DataShapeConfig
from configs.data_config.project_config import ProjectConfig
from configs.train_config.pretrain_config import PretrainConfig
from data.train_data.dataset import CamelsDataset
from utils.model.test_full import test_full
from utils.model.tools import seed_torch

warnings.filterwarnings("ignore")


def pretrain_test(past_len, pred_len, feature_name, model_name):
    device = ProjectConfig.device
    num_workers = ProjectConfig.num_workers
    dataset_num_worker = ProjectConfig.dataset_num_worker
    prefetch_factor = ProjectConfig.prefetch_factor

    datashape_config = DataShapeConfig(past_len, pred_len, feature_name)
    past_len = datashape_config.past_len
    pred_len = datashape_config.pred_len
    src_size = datashape_config.src_size
    use_baseflow = datashape_config.use_baseflow
    use_signatures = datashape_config.use_signatures
    out_features_index = datashape_config.out_features_index
    streamflow_size = datashape_config.streamflow_size
    signatures_size = datashape_config.signatures_size
    streamflow_index = out_features_index[0]
    signatures_index = out_features_index[1]
    streamflow_columns = datashape_config.streamflow_columns
    signatures_columns = datashape_config.signatures_columns

    pretrain_config = PretrainConfig('test', model_name, datashape_config)
    seed = pretrain_config.seed
    data_root = pretrain_config.data_root
    saving_root = Path(pretrain_config.saving_root)
    basin_root = pretrain_config.basin_root
    decode_mode = pretrain_config.decode_mode
    batch_size = pretrain_config.batch_size
    best_model = pretrain_config.model

    print("pid:", os.getpid())
    seed_torch(seed=seed)
    if (saving_root / 'log_test.csv').exists():
        print(f'Already test in {saving_root}!')
        return
    print(saving_root)
    # Model
    best_path = list(saving_root.glob(f"(max_sf_nse)*.pkl"))
    assert (len(best_path) == 1)
    best_path = best_path[0]
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    # Train mean and std to normalize data
    train_means = np.loadtxt(saving_root / "train_means.csv", dtype="float32")
    train_stds = np.loadtxt(saving_root / "train_stds.csv", dtype="float32")
    train_x_mean = train_means[:src_size]
    train_y_mean = train_means[src_size:]
    train_x_std = train_stds[:src_size]
    train_y_std = train_stds[src_size:]

    # Dataset
    with open(basin_root, 'rb') as f:
        basin_test = list(pickle.load(f))

    dataset_test = CamelsDataset(data_root, basin_test, past_len, pred_len,
                                 use_baseflow, use_signatures, streamflow_index, signatures_index,
                                 device, dataset_num_worker, 'test',
                                 x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers,
                             prefetch_factor=prefetch_factor, shuffle=False)
    # Testing
    test_full(best_model, decode_mode, loader_test, device, saving_root,
              streamflow_size, signatures_size, streamflow_columns, signatures_columns)


if __name__ == '__main__':
    feature_list = ['streamflow', 'baseflow', 'baseflow_signatures']
    for past_len in [90]:
        for feature in feature_list:
            pred_len = 30
            if past_len == 15:
                pred_len = 15
            pretrain_test(past_len, pred_len, feature, 'LSTMMSVS2S')
