# All metrics needs obs and sim shape: (batch_size, pred_len, streamflow_size)

import numpy as np
import torch


# 先计算成二维矩阵，行坐标为天，纵坐标为数据
# 再按行坐标平均mean计算特征统计
class CalcEvalIndex:
    def __init__(self, obs, sim, calc_median=True):
        self.obs, self.sim = self.drop_nan(obs, sim)
        self.mean_obs = np.mean(self.obs, axis=0)
        self.mean_sim = np.mean(self.sim, axis=0)
        self.std_obs = np.std(self.obs, axis=0)
        self.std_sim = np.std(self.sim, axis=0)
        self.calc_median = calc_median

    def calc_nse(self):
        denominator = np.mean((self.obs - self.mean_obs) ** 2, axis=0)
        diff = (self.sim - self.obs) ** 2
        mean = np.mean(diff, axis=0)
        mean = 1 - mean / denominator
        median = None
        if self.calc_median:
            median = np.median(diff, axis=0)
            median = 1 - median / denominator
        return mean, median

    def calc_R(self):
        diff_obs = self.obs - self.mean_obs
        diff_sim = self.sim - self.mean_sim
        denominator = (np.sqrt(np.mean(diff_sim ** 2, axis=0)) *
                       np.sqrt(np.mean(diff_obs ** 2, axis=0)))
        mean = np.mean(diff_obs * diff_sim, axis=0) / denominator
        median = None
        if self.calc_median:
            median = np.median(diff_obs * diff_sim, axis=0) / denominator
        return mean, median

    def calc_kge(self):
        beta = self.mean_sim / self.mean_obs
        alpha = self.std_sim / self.std_obs
        denominator = self.std_obs * self.std_sim
        diff = (self.obs - self.mean_obs) * (self.sim - self.mean_sim)
        mean = np.mean(diff, axis=0) / denominator
        mean = 1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (mean - 1) ** 2)
        median = None
        if self.calc_median:
            median = np.median(diff, axis=0) / denominator
            median = 1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (median - 1) ** 2)
        return mean, median

    # alpha是从高到低的n%的top
    def calc_tpe(self, alpha):
        shape = self.obs.shape
        mean_all = np.empty((shape[1], shape[2]), dtype=np.float32)
        median_all = np.empty((shape[1], shape[2]), dtype=np.float32)
        top = int(shape[0] * alpha / 100.0)
        for index in range(shape[2]):
            obs_temp = self.obs[:, :, index]
            sim_temp = self.sim[:, :, index]
            sort_index = np.argsort(obs_temp, axis=0)
            obs_sort = np.take_along_axis(obs_temp, sort_index, axis=0)
            sim_sort = np.take_along_axis(sim_temp, sort_index, axis=0)
            obs_t = obs_sort[-top:, :]
            sim_t = sim_sort[-top:, :]
            denominator = np.mean(obs_t, axis=0)
            bias = np.abs(sim_t - obs_t)
            mean = np.mean(bias, axis=0) / denominator
            mean_all[:, index] = mean
            if self.calc_median:
                median = np.median(bias, axis=0) / denominator
                median_all[:, index] = median
        return mean_all, median_all

    def calc_fdc_fms(self, m1: float = 0.2, m2: float = 0.7) -> float:
        """[summary]

        Parameters
        ----------
        obs : np.ndarray
            Array containing the discharge observations
        sim : np.ndarray
            Array containing the discharge simulations
        m1 : float, optional
            Lower bound of the middle section. Has to be in range(0,1), by default 0.2
        m2 : float, optional
            Upper bound of the middle section. Has to be in range(0,1), by default 0.2

        Returns
        -------
        float
            Bias of the middle slope of the flow duration curve (Yilmaz 2018).

        Raises
        ------
        RuntimeError
            If `obs` and `sim` don't have the same length
        RuntimeError
            If `m1` is not in range(0,1)
        RuntimeError
            If `m2` is not in range(0,1)
        RuntimeError
            If `m1` >= `m2`
        """
        # make sure that metric is calculated over the same dimension
        obs = self.obs
        sim = self.sim

        if obs.shape != sim.shape:
            raise RuntimeError("obs and sim must be of the same length.")

        if (m1 <= 0) or (m1 >= 1):
            raise RuntimeError("m1 has to be in the range (0,1)")

        if (m2 <= 0) or (m2 >= 1):
            raise RuntimeError("m1 has to be in the range (0,1)")

        if m1 >= m2:
            raise RuntimeError("m1 has to be smaller than m2")

        # for numerical reasons change 0s to 1e-6
        sim[sim == 0] = 1e-6
        obs[obs == 0] = 1e-6

        # sort both in descending order
        obs = -np.sort(-obs, axis=0)
        sim = -np.sort(-sim, axis=0)

        # calculate fms part by part
        qsm1 = np.log(sim[np.round(m1 * len(sim)).astype(int)] + 1e-6)
        qsm2 = np.log(sim[np.round(m2 * len(sim)).astype(int)] + 1e-6)
        qom1 = np.log(obs[np.round(m1 * len(obs)).astype(int)] + 1e-6)
        qom2 = np.log(obs[np.round(m2 * len(obs)).astype(int)] + 1e-6)

        fms = ((qsm1 - qsm2) - (qom1 - qom2)) / (qom1 - qom2 + 1e-6)

        return fms * 100

    def calc_fdc_fhv(self, h: float = 0.02) -> float:
        """Peak flow bias of the flow duration curve (Yilmaz 2018).

        Parameters
        ----------
        obs : np.ndarray
            Array containing the discharge observations
        sim : np.ndarray
            Array containing the discharge simulations
        h : float, optional
            Fraction of the flows considered as peak flows. Has to be in range(0,1), by default 0.02

        Returns
        -------
        float
            Bias of the peak flows

        Raises
        ------
        RuntimeError
            If `obs` and `sim` don't have the same length
        RuntimeError
            If `h` is not in range(0,1)
        """
        # make sure that metric is calculated over the same dimension
        obs = self.obs
        sim = self.sim

        if obs.shape != sim.shape:
            raise RuntimeError("obs and sim must be of the same length.")

        if (h <= 0) or (h >= 1):
            raise RuntimeError("h has to be in the range (0,1)")

        # sort both in descending order
        obs = -np.sort(-obs, axis=0)
        sim = -np.sort(-sim, axis=0)

        # subset data to only top h flow values
        obs = obs[:np.round(h * len(obs)).astype(int)]
        sim = sim[:np.round(h * len(sim)).astype(int)]

        fhv = np.sum(sim - obs, axis=0) / (np.sum(obs, axis=0) + 1e-6)

        return fhv * 100

    def calc_fdc_flv(self, l: float = 0.7) -> float:
        """[summary]

        Parameters
        ----------
        obs : np.ndarray
            Array containing the discharge observations
        sim : np.ndarray
            Array containing the discharge simulations
        l : float, optional
            Upper limit of the flow duration curve. E.g. 0.7 means the bottom 30% of the flows are
            considered as low flows, by default 0.7

        Returns
        -------
        float
            Bias of the low flows.

        Raises
        ------
        RuntimeError
            If `obs` and `sim` don't have the same length
        RuntimeError
            If `l` is not in the range(0,1)
        """
        # make sure that metric is calculated over the same dimension
        obs = self.obs
        sim = self.sim

        if obs.shape != sim.shape:
            raise RuntimeError("obs and sim must be of the same length.")

        if (l <= 0) or (l >= 1):
            raise RuntimeError("l has to be in the range (0,1)")

        # for numerical reasons change 0s to 1e-6
        sim[sim == 0] = 1e-6
        obs[obs == 0] = 1e-6

        # sort both in descending order
        obs = -np.sort(-obs, axis=0)
        sim = -np.sort(-sim, axis=0)

        # subset data to only top h flow values
        obs = obs[np.round(l * len(obs)).astype(int):]
        sim = sim[np.round(l * len(sim)).astype(int):]

        # transform values to log scale
        obs = np.log(obs + 1e-6)
        sim = np.log(sim + 1e-6)

        # calculate flv part by part
        qsl = np.sum(sim - sim.min(), axis=0)
        qol = np.sum(obs - obs.min(), axis=0)

        flv = -1 * (qsl - qol) / (qol + 1e-6)

        return flv * 100

    # alpha是从高到低的n%的top
    def calc_tpe_1D(self, alpha):
        shape = self.obs.shape
        top = int(shape[0] * alpha / 100.0)
        sort_index = np.argsort(self.obs, axis=0)
        obs_sort = np.take_along_axis(self.obs, sort_index, axis=0)
        sim_sort = np.take_along_axis(self.sim, sort_index, axis=0)
        obs_t = obs_sort[-top:]
        sim_t = sim_sort[-top:]
        denominator = np.mean(obs_t, axis=0)
        bias = np.abs(sim_t - obs_t)
        mean = np.mean(bias, axis=0) / denominator
        median = None
        if self.calc_median:
            median = np.median(bias, axis=0) / denominator
        return mean, median

    def calc_bias(self):
        bias = self.sim - self.obs
        mean = np.mean(bias, axis=0)
        mean = mean / self.mean_obs
        median = None
        if self.calc_median:
            median = np.median(bias, axis=0)
            median = median / self.mean_obs
        return mean, median

    def calc_rmse(self):
        mse = (self.obs - self.sim) ** 2
        mean = np.sqrt(np.mean(mse, axis=0))
        median = None
        if self.calc_median:
            median = np.sqrt(np.median(mse, axis=0))
        return mean, median

    def calc_nrmse(self):
        mse = (self.obs - self.sim) ** 2
        diff = np.max(self.obs, axis=0) - np.min(self.obs, axis=0)
        mean = np.sqrt(np.mean(mse, axis=0)) / diff
        median = None
        if self.calc_median:
            median = np.sqrt(np.median(mse, axis=0)) / diff
        return mean, median

    def drop_nan(self, array1: np.array, array2: np.array):
        index_array = []
        for index in range(array1.shape[0]):
            if np.isnan(array1[index]).any() or np.isnan(array2[index]).any():
                index_array.append(index)
        return (np.delete(array1, index_array, axis=0),
                np.delete(array2, index_array, axis=0))


def calc_nse(obs, sim):
    mean_obs = np.nanmean(obs, axis=0)
    denominator = np.nanmean((obs - mean_obs) ** 2, axis=0)
    diff = (sim - obs) ** 2
    nse = 1 - np.mean(diff, axis=0) / denominator
    return nse


def calc_nrmse(obs, sim):
    obs = np.nanmean(obs, axis=1)
    sim = np.nanmean(sim, axis=1)
    rmse = np.sqrt(np.nanmean((obs - sim) ** 2, axis=0))
    max = np.max(obs, axis=0)
    min = np.min(obs, axis=0)
    nrmse = np.divide(rmse, max - min, out=np.zeros_like(rmse), where=(max - min != 0))
    rmse[np.isinf(rmse)] = 0
    return nrmse


if __name__ == '__main__':
    # obs1 = np.load(r'C:\Users\admini\Desktop\obs.npy')
    # sim1 = np.load(r'C:\Users\admini\Desktop\sim.npy')
    # cal = CalcEvalIndex(obs=obs1, sim=sim1)
    # rmse_mean, rmse_median = cal.calc_rmse()
    # nse_mean, nse_median = cal.calc_nse()
    # kge_mean, kge_median = cal.calc_kge()
    # bias_mean, bias_median = cal.calc_bias()
    # tpe5_mean, tpe5_median = cal.calc_tpe(5)
    rmse = np.array([1, 2, 3, 4, np.inf, np.nan])
    rmse[np.isinf(rmse)] = 0
    rmse[np.isnan(rmse)] = 0
    rmse[rmse is np.nan] = 0
    print(rmse)
