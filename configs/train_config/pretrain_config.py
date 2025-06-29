import os.path
import importlib

import torch
from torch import nn

from configs.data_config.path_config import PathConfig
from configs.data_config.dataset_config import DataShapeConfig
from configs.data_config.project_config import ProjectConfig
from utils.model import streamflow_loss
from utils.model.lr_strategies import SchedulerFactory


class PretrainConfig:

    def __init__(self, stage, used_model, datashape_config: DataShapeConfig):
        self.stage = stage  # TODO train test
        # Random seed config
        self.seed = 1234
        device = ProjectConfig.device

        # training config
        scale_factor = 1  # TODO: the bath_size bigger, the learning_rate larger.
        self.n_epochs = 200  # TODO: origin 200
        self.batch_size = 4096 // scale_factor  # TODO
        self.learning_rate = 0.001 / scale_factor  # TODO: lr=0.0002
        self.learning_rate_weights = 0.001  # TODO

        # use model config
        self.used_model = used_model  # TODO Transformer LSTMMSVS2S

        # 先修改model_config
        model_congfigs = importlib.import_module(f"configs.model_config.{self.used_model}_config")
        ModelConfig = getattr(model_congfigs, f"{self.used_model}Config")
        model_config = ModelConfig(datashape_config)
        # 后新建model
        models = importlib.import_module("models")
        Model = getattr(models, self.used_model)
        model = Model(model_config)
        if ProjectConfig.multi_gpu:
            model = nn.DataParallel(model, device_ids=ProjectConfig.device_ids)
        self.model = model.to(device)

        # loss function config
        self.weights = [1, 1, 1]
        if not datashape_config.use_baseflow:
            self.weights.pop()
        if not datashape_config.use_signatures:
            self.weights.pop()
        if datashape_config.use_baseflow or datashape_config.use_signatures:
            self.loss_func = streamflow_loss.StreamflowLoss(self.model, self.weights).to(device)
        else:
            self.loss_func = streamflow_loss.StreamflowLoss().to(device)

        # Optimizer, Scheduler config
        scheduler_paras = {"scheduler_type": "warm_up", "last_epoch": -1,
                           "warm_up_epochs": self.n_epochs * 0.25, "decay_rate": 0.99}
        if datashape_config.use_baseflow or datashape_config.use_signatures:
            self.optimizer = torch.optim.AdamW([{'params': self.model.parameters(), 'lr': self.learning_rate},
                                               {'params': self.loss_func.weights, 'lr': self.learning_rate_weights}])
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = SchedulerFactory.get_scheduler(self.optimizer, **scheduler_paras)
        # scheduler_paras = {"scheduler_type": "none", "last_epoch": -1,}
        # scheduler_paras = {"scheduler_type": "exp_decay", "last_epoch": -1,
        #                    "decay_epoch": n_epochs * 0.5, "decay_rate": 0.99}
        # scheduler_paras = {"scheduler_type": "cos_anneal", "last_epoch": -1,
        #                    "cos_anneal_t_max": 32}

        # save message config
        self.learning_config_info = f"n{self.n_epochs}_bs{self.batch_size}_lr{self.learning_rate}_lrw{self.learning_rate_weights}"
        self.decode_mode = ModelConfig.decode_mode
        self.saving_message = f"{model_config.model_info}@{datashape_config.data_shape_info}" \
                              f"@{self.learning_config_info}@seed{self.seed}"

        # input and output path config
        days = datashape_config.past_len + datashape_config.pred_len
        self.data_root = os.path.join(PathConfig.train_path, str(days))
        self.basin_root = os.path.join(PathConfig.model_conf_path, f'{stage}_gauge_list.pkl')
        self.saving_root = os.path.join(PathConfig.model_path, str(days), self.saving_message)
        # if stage == 'test':
        #     self.saving_root = os.path.join(PathConfig.model_path, stage, self.saving_message)
