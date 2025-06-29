import concurrent.futures
import os
from bisect import bisect_right
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset


class CamelsDataset(Dataset):
    def __init__(self, root_path: str, basin_list: list, past_len: int, pred_len: int,
                 withBaseflow, withSignatures, streamflow_index, signatures_index, device='cuda:0', num_worker=4,
                 stage='train', x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        # 设置数据
        self.root_path = root_path
        self.basin_list = np.array(basin_list)
        self.past_len = past_len
        self.pred_len = pred_len
        self.withBaseflow = withBaseflow
        self.withSignatures = withSignatures
        self.streamflow_index = streamflow_index
        self.signatures_index = signatures_index
        self.device = device
        self.num_worker = num_worker
        # 初始化
        self.length_dict = dict()
        self.x_dict = dict()
        self.y_dict = dict()
        self.stage = stage
        self.y_stds_dict = dict()
        self.length = 0
        self.index_ls = list()
        if y_stds_dict is not None:
            self.y_stds_dict = y_stds_dict
        if self.stage == 'train':
            self.x_mean = None
            self.x_std = None
            self.y_mean = None
            self.y_std = None
        elif self.stage == 'test' or self.stage == 'val':
            self.x_mean = x_mean
            self.y_mean = y_mean
            self.x_std = x_std
            self.y_std = y_std
        else:
            raise RuntimeError("Stage is wrong!.")
        # 加载数据
        print(f'Loading {self.stage} data!')
        self._load_data()
        # 标准化数据
        print(f'Normalizing {self.stage} data!')
        self._normalize_data()

    def __getitem__(self, idx):
        # 二分查找到idx数据所属的哪个basin_idx，再计算流域内的相对local_idx
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx] - 1
        basin = self.basin_list[basin_idx]
        # 提取某流域的输入输出数据
        x_seq = self.x_dict[basin][local_idx, :, :]
        y_seq_past = self.y_dict[basin][local_idx, :self.past_len, :]
        y_seq_pred = self.y_dict[basin][local_idx, self.past_len:self.past_len + self.pred_len, :]
        # y_stds_dict[basin]是为了计算nse，需要每个流域的流量std，在val和test时没用
        if self.stage == 'train':
            y_std = self.y_stds_dict[basin]
        else:
            y_std = basin
        return x_seq, y_seq_past, y_seq_pred, y_std

    def __len__(self):
        return self.length

    def _load_data(self):
        basin_lists = np.array(self.basin_list)
        np.random.shuffle(basin_lists)
        use_chunk_mean = True
        if self.num_worker > len(basin_lists):
            self.num_worker = len(basin_lists)
        if 100 * self.num_worker >= len(basin_lists):
            # 如果basin_lists数量太少，就不能采用集合mean_std的形式
            use_chunk_mean = False
        basin_lists = np.array_split(basin_lists, self.num_worker)
        future_list = []
        # 多线程加载数据到内存
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_worker) as executor:
            for index in range(self.num_worker):
                future = executor.submit(self.load_data_basins,
                                         self, f'Thread_{index}', basin_lists[index], use_chunk_mean)
                future_list.append(future)
        # 当为train时，计算数据整体mean和std
        if self.stage == 'train':
            if use_chunk_mean:
                input_mean, input_std, output_mean, output_std = [], [], [], []
                for future in future_list:
                    input_mean.append(future.result()[0])
                    input_std.append(future.result()[1])
                    output_mean.append(future.result()[2])
                    output_std.append(future.result()[3])
                input_mean = np.concatenate(input_mean, axis=0)
                input_std = np.concatenate(input_std, axis=0)
                output_mean = np.concatenate(output_mean, axis=0)
                output_std = np.concatenate(output_std, axis=0)
                self.x_mean = np.nanmean(input_mean, axis=0)
                self.x_std = np.sqrt(np.nanmean(np.square(input_std), axis=0))
                self.y_mean = np.nanmean(output_mean, axis=0)
                self.y_std = np.sqrt(np.nanmean(np.square(output_std), axis=0))
            else:
                input_data, output_data = [], []
                for future in future_list:
                    input_data.append(future.result()[0])
                    output_data.append(future.result()[1])
                input_data = np.concatenate(input_data, axis=0)
                output_data = np.concatenate(output_data, axis=0)
                self.x_mean = np.nanmean(input_data, axis=0)
                self.x_std = np.nanstd(input_data, axis=0)
                self.y_mean = np.nanmean(output_data, axis=0)
                self.y_std = np.nanstd(output_data, axis=0)
        # 处理数据索引
        length = 0
        index_ls = [0]
        for basin in self.basin_list:
            basin_len = self.length_dict[basin]
            length += basin_len
            index_ls.append(index_ls[-1] + basin_len)
        self.length = length
        self.index_ls = index_ls

    def _normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basin_list.tolist()):
            # print(f"[{basin}]: normalizing data %.4f" % (idx / len(self.basin_list)))
            x_norm = self.normalization(self.x_dict[basin], self.x_mean, self.x_std)
            y_norm = self.normalization(self.y_dict[basin], self.y_mean, self.y_std)
            self.x_dict[basin] = x_norm.astype(np.float32)
            self.y_dict[basin] = y_norm.astype(np.float32)

    def get_means(self):
        return self.x_mean, self.y_mean

    def get_stds(self):
        return self.x_std, self.y_std

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'x':
            feature = feature * self.x_std + self.x_mean
        elif variable == 'y':
            feature = feature * self.y_std + self.y_mean
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    @staticmethod
    def load_data_basins(cls, thread_num, basins, use_chunk_mean):
        basin_number = len(basins)
        input_data = None
        output_data = None
        for idx, basin in enumerate(basins):
            try:
                # print(f"[{thread_num}-{basin}]: loading data %.4f" % (idx / basin_number), flush=True)
                camels_name = basin.split('_')[0]
                dirpath = os.path.join(cls.root_path, camels_name)
                if os.path.exists(os.path.join(dirpath, cls.stage)):
                    dirpath = os.path.join(dirpath, cls.stage)
                filepath = os.path.join(dirpath, f'{basin}.mat')
                data = loadmat(filepath)
                # 处理nan数据
                data = cls.drop_nan_data(basin, data, cls.withBaseflow, cls.withSignatures)
                if data is None:  # static data contains nan value
                    cls.basin_list = np.delete(cls.basin_list, np.where(cls.basin_list == basin))
                    continue
                # 处理x_dict数据
                forcing_static = np.array(data['forcing_static']).flatten()
                forcing_timeseries = np.array(data['forcing_timeseries'])
                forcing_timeseries = cls.concat_timeseries_static(forcing_timeseries, forcing_static)
                # 处理y_dict数据
                streamflow_timeseries = np.array(data['streamflow_timeseries'])
                streamflow_timeseries = streamflow_timeseries[:, :, cls.streamflow_index]
                streamflow_static = np.array(data['streamflow_static']).flatten()
                streamflow_static = streamflow_static[cls.signatures_index]
                streamflow_timeseries = cls.concat_timeseries_static(streamflow_timeseries, streamflow_static)
                # 保存样本数据，分basin
                cls.x_dict[basin] = forcing_timeseries.astype(np.float32)
                cls.y_dict[basin] = streamflow_timeseries.astype(np.float32)
                # 保存每个basin的样本数length
                cls.length_dict[basin] = streamflow_timeseries.shape[0]
                # 保存每个basin的mean和std
                streamflow_timeseries = cls.flatten_3d_array(streamflow_timeseries, cls.pred_len)
                cls.y_stds_dict[basin] = np.nanstd(streamflow_timeseries, axis=0)
                # stage==test，只计算y_stds_dict（用于nse计算）
                if cls.stage == 'train':
                    forcing_timeseries = cls.flatten_3d_array(forcing_timeseries, cls.pred_len)
                    if input_data is None:
                        input_data = forcing_timeseries
                        output_data = streamflow_timeseries
                    else:
                        input_data = np.concatenate([input_data, forcing_timeseries], axis=0)
                        output_data = np.concatenate([output_data, streamflow_timeseries], axis=0)
            except Exception:
                # print(f"[{basin}] contains error!")
                cls.x_dict.pop(basin, 0)
                cls.y_dict.pop(basin, 0)
                cls.length_dict.pop(basin, 0)
                cls.y_stds_dict.pop(basin, 0)
                cls.basin_list = np.delete(cls.basin_list, np.where(cls.basin_list == basin))
                # traceback.print_exc()
                continue
        if not use_chunk_mean:  # False为不使用chunk mean，直接返回全部data
            return input_data, output_data
        input_mean = np.nanmean(input_data, axis=0).reshape(1, -1)
        input_std = np.nanstd(input_data, axis=0).reshape(1, -1)
        output_mean = np.nanmean(output_data, axis=0).reshape(1, -1)
        output_std = np.nanstd(output_data, axis=0).reshape(1, -1)
        return input_mean, input_std, output_mean, output_std

    @staticmethod
    def drop_nan_data(basin, data, with_baseflow, with_signatures):
        streamflow_static = data['streamflow_static']
        forcing_static = data['forcing_static']
        if ((with_signatures and np.isnan(streamflow_static).any())
                or np.isnan(forcing_static).any()):
            # print(f"[{basin}] static data contains nan value!")
            return None
        streamflow_timeseries = data['streamflow_timeseries']
        forcing_timeseries = data['forcing_timeseries']
        indexs = []
        for index in range(streamflow_timeseries.shape[0]):
            if (np.isnan(streamflow_timeseries[index, :, 0]).any()
                    or (with_baseflow and np.isnan(streamflow_timeseries[index, :, 1:]).any())
                    or np.isnan(forcing_timeseries[index, :, :]).any()):
                indexs.append(index)
                # print(f"[{basin}-{index}] timeseries data contains nan value!")
        if len(indexs) > 0:
            data['streamflow_timeseries'] = np.delete(streamflow_timeseries, indexs, axis=0)
            data['forcing_timeseries'] = np.delete(forcing_timeseries, indexs, axis=0)
        return data

    @staticmethod
    # 拼接timeseries数据和static数据，需要扩容static维度
    def concat_timeseries_static(timeseries: np.ndarray, static: np.ndarray) -> np.ndarray:
        shape = timeseries.shape
        static = static.reshape((1, 1, len(static))).repeat(shape[0], axis=0).repeat(shape[1], axis=1)
        return np.concatenate((timeseries, static), axis=2)

    @staticmethod
    # 为所有特征列标准化
    def normalization(feature: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        shape = feature.shape
        mean = mean.reshape((1, 1, len(mean))).repeat(shape[0], axis=0).repeat(shape[1], axis=1)
        std = std.reshape((1, 1, len(std))).repeat(shape[0], axis=0).repeat(shape[1], axis=1)
        feature = feature - mean
        # 避免某些特征（glacier_extent和wetlands_extent）的std为0
        feature = np.divide(feature, std, out=feature, where=(std != 0))
        return feature

    @staticmethod
    # 将三维的第一维flatten到二维矩阵的第一维或第二维，axis=0为行拼接
    def flatten_3d_array(array_3d: np.ndarray, pred_len) -> np.ndarray:
        array_2d = array_3d[0, :, :]
        for i in range(1, array_3d.shape[0]):
            array_2d = np.concatenate(
                (array_2d, array_3d[i, -pred_len:, :]), axis=0)
        return array_2d


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4]).reshape((1, 4, 1)).repeat(3, axis=0).repeat(5, axis=2)
    f = a[0]
    for i in range(1, a.shape[0]):
        f = np.concatenate([f, a[i]], axis=1)

    b = np.array([[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                  [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                  [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]])
    d = b[:, :, -1:]
    print()
