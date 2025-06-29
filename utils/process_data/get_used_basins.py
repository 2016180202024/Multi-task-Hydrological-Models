import pickle
from pathlib import Path

from configs.data_config.path_config import PathConfig


def get_used_basins():
    model_conf_path = Path(PathConfig.model_conf_path) / 'used_basin.pkl'
    with open(model_conf_path, 'rb') as file:
        used_basins = list(pickle.load(file))
    return used_basins
