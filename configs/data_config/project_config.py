import os
import torch


# Project root, computing resources
class ProjectConfig:
    # multi_gpu = True
    gpu_count = torch.cuda.device_count()
    multi_gpu = (gpu_count > 1)  # TODO: multi gpu to run
    device_ids = []
    if multi_gpu:
        device_ids = [i for i in range(gpu_count)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids)[1:-1]
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    num_workers = 8  # TODO: number of threads for loading data
    dataset_num_worker = 8
    prefetch_factor = 4  # TODO
    pin_memory = False  # TODO
    use_board = True  # TODO: production environment is False
    use_train_eval = False
