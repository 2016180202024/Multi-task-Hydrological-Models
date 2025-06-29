import os
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from configs.data_config.dataset_config import DataShapeConfig
from configs.data_config.project_config import ProjectConfig
from configs.train_config.pretrain_config import PretrainConfig
from data.train_data.dataset import CamelsDataset
from utils.model.tools import seed_torch
from utils.model.train_full import train_full


def pretrain(past_len, pred_len, feature, model_name):
    device = ProjectConfig.device
    num_workers = ProjectConfig.num_workers
    dataset_num_worker = ProjectConfig.dataset_num_worker
    use_board = ProjectConfig.use_board
    prefetch_factor = ProjectConfig.prefetch_factor
    pin_memory = ProjectConfig.pin_memory
    use_train_eval = ProjectConfig.use_train_eval

    datashape_config = DataShapeConfig(past_len, pred_len, feature)
    past_len = datashape_config.past_len
    pred_len = datashape_config.pred_len
    use_baseflow = datashape_config.use_baseflow
    use_signatures = datashape_config.use_signatures
    streamflow_index = datashape_config.out_features_index[0]
    signatures_index = datashape_config.out_features_index[1]
    streamflow_size = datashape_config.streamflow_size
    signatures_size = datashape_config.signatures_size

    pretrain_config = PretrainConfig('train', model_name, datashape_config)
    seed = pretrain_config.seed
    data_root = pretrain_config.data_root
    saving_root = Path(pretrain_config.saving_root)
    basin_root = pretrain_config.basin_root
    decode_mode = pretrain_config.decode_mode
    n_epochs = pretrain_config.n_epochs
    batch_size = pretrain_config.batch_size
    loss_func = pretrain_config.loss_func
    model = pretrain_config.model
    optimizer = pretrain_config.optimizer
    scheduler = pretrain_config.scheduler

    print("pid:", os.getpid())
    seed_torch(seed=seed)
    saving_root.mkdir(exist_ok=True, parents=True)
    if (saving_root / 'tb_log').exists() and len(os.listdir(saving_root / 'tb_log')) >= 300:
        print(f'Already train in {saving_root}!')
        return
    print(saving_root)
    torch.autograd.set_detect_anomaly(True)

    with open(basin_root, 'rb') as f:
        basin_train = list(pickle.load(f))
    with open(basin_root.replace('train_', 'val_'), 'rb') as f:
        basin_val = list(pickle.load(f))

    dataset_train = CamelsDataset(data_root, basin_train, past_len, pred_len,
                                  use_baseflow, use_signatures, streamflow_index, signatures_index,
                                  device, dataset_num_worker)
    dataset_val = CamelsDataset(data_root, basin_val, past_len, pred_len,
                                use_baseflow, use_signatures, streamflow_index, signatures_index,
                                device, dataset_num_worker,
                                'val', dataset_train.get_means()[0], dataset_train.get_means()[1],
                                dataset_train.get_stds()[0], dataset_train.get_stds()[1])
    # We use the feature means/stds of the training data for normalization in val and test stage
    train_x_mean, train_y_mean = dataset_train.get_means()
    train_x_std, train_y_std = dataset_train.get_stds()
    # Saving training mean and training std
    train_means = np.concatenate((train_x_mean, train_y_mean), axis=0)
    train_stds = np.concatenate((train_x_std, train_y_std), axis=0)
    np.savetxt(saving_root / "train_means.csv", train_means)
    np.savetxt(saving_root / "train_stds.csv", train_stds)
    with open(saving_root / "y_stds_dict.pickle", "wb") as f:
        pickle.dump(dataset_train.y_stds_dict, f)
    # dataset_train, dataset_val = random_split(
    #     dataset_train,
    #     [round(0.8 * len(dataset_train)), round(0.2 * len(dataset_train))],
    #     generator=torch.Generator().manual_seed(seed)
    # )

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory)

    # model.load_state_dict(torch.load(
    #     r'D:\experiment\model\60\Transformer_NAR_[64-4-4-256-0.1]@[30-30,53-1]_bfFalse_snFalse@n200_bs1024_lr0.001@seed1234\(max_sf_nse)_113_0.5689130425453186.pkl', map_location=device))

    # Training and Validation
    train_full(model, decode_mode, loader_train, loader_val, optimizer,
               scheduler, loss_func, n_epochs, device, saving_root, use_board, use_train_eval,
               use_baseflow, use_signatures, streamflow_size, signatures_size)


def train_for_models():
    days = [30, 60, 90, 120]
    predict_len = 5
    models = ['Transformer', 'LSTMMSVS2S']
    # models = ['LSTMMSVS2S']
    features = ['streamflow', 'baseflow', 'baseflow_signatures']
    for day in days:
        for feature in features:
            for model in models:
                print(f'[train_for_models] {day} {feature} {model}')
                pretrain(day - predict_len, predict_len, feature, model)
                # pretrain_test(day - predict_len, predict_len, feature, model)


def train_for_train_days():
    days = [60+5*i for i in range(21)]
    predict_len = 5
    models = ['LSTMMSVS2S']
    features = ['streamflow', 'baseflow_signatures']
    for day in days:
        if day % 30 == 0:
            continue
        for feature in features:
            for model in models:
                pretrain(day - predict_len, predict_len, feature, model)
                # pretrain_test(day - predict_len, predict_len, feature, model)


def train_for_predict_days():
    past_day = 120
    pred_days = [i for i in range(5, 45, 5)]
    models = ['LSTMMSVS2S']
    features = ['streamflow', 'baseflow_signatures']
    for pred_day in pred_days:
        for feature in features:
            for model in models:
                pretrain(past_day, pred_day, feature, model)
                # pretrain_test(day - predict_len, predict_len, feature, model)


def train_for_features():
    days = [60]
    # days = [90]
    predict_len = 5
    models = ['LSTMMSVS2S']
    features = ['runoff_ratio', 'q_mean_5_95', 'stream_elas', 'fdc_slope',
                'BFI_5', 'BFI_60', 'hfd_mean', 'high_q_freq_dur', 'low_q_freq_dur']
    for day in days:
        for feature in features:
            for model in models:
                pretrain(day - predict_len, predict_len, feature, model)
                # pretrain_test(day - predict_len, predict_len, feature, model)


if __name__ == '__main__':
    # train_for_models()
    # train_for_train_days()
    # train_for_predict_days()
    train_for_features()
