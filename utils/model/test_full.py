import time
import numpy as np
import pandas as pd
from utils.model.metrics import CalcEvalIndex
from utils.model.eval import predict_by_dataloader_and_model
from utils.model.tools import count_parameters


def test_full(test_model, decode_mode, loader_test, device, saving_root,
              streamflow_size, signatures_size, streamflow_columns, signatures_columns):
    t1 = time.time()
    print(f"Parameters count:{count_parameters(test_model)}")
    log_file = saving_root / f"log_test.csv"
    with open(log_file, "wt") as f:
        f.write(f"parameters_count:{count_parameters(test_model)}\n")
    obs, pred = predict_by_dataloader_and_model(test_model, loader_test, decode_mode, device)
    # Calculate nse after rescale (But if you take the same mean and std, it's equivalent before and after)
    obs = loader_test.dataset.local_rescale(obs.numpy(), variable='y')
    pred = loader_test.dataset.local_rescale(pred.numpy(), variable='y')
    # eval output data
    eval_data = pd.DataFrame()
    eval_data['days'] = np.arange(0, obs.shape[1] + 1, 1)
    eval_data.iloc[0, 0] = 'mean'
    # streamflow eval
    obs_streamflow, pred_streamflow = (
        obs[:, :, :streamflow_size], pred[:, :, :streamflow_size],
    )
    obs_streamflow[obs_streamflow < 0] = 0
    pred_streamflow[pred_streamflow < 0] = 0
    # streamflow eval columns
    eval_class = ['mean', 'median']
    eval_stream_index = ['rmse', 'nse', 'kge', 'fhv', 'fms', 'flv', 'bias', 'tpe5', 'R']
    eval_stream_index = concat_columns(eval_class, eval_stream_index, streamflow_columns)
    # streamflow eval value
    cal_stream = CalcEvalIndex(obs=obs_streamflow, sim=pred_streamflow)
    rmse_mean, rmse_median = cal_stream.calc_rmse()
    nse_mean, nse_median = cal_stream.calc_nse()
    kge_mean, kge_median = cal_stream.calc_kge()
    fhv_mean = cal_stream.calc_fdc_fhv()
    fms_mean = cal_stream.calc_fdc_fms()
    flv_mean = cal_stream.calc_fdc_flv()
    bias_mean, bias_median = cal_stream.calc_bias()
    tpe5_mean, tpe5_median = cal_stream.calc_tpe(5)
    R_mean, R_median = cal_stream.calc_R()
    eval_stream_data = np.concatenate(
        (mean_array(rmse_mean), mean_array(nse_mean), mean_array(kge_mean),
         mean_array(fhv_mean), mean_array(fms_mean), mean_array(flv_mean),
         mean_array(bias_mean), mean_array(tpe5_mean), mean_array(R_mean),
         mean_array(rmse_median), mean_array(nse_median), mean_array(kge_median),
         mean_array(fhv_mean), mean_array(fms_mean), mean_array(flv_mean),
         mean_array(bias_median), mean_array(tpe5_median), mean_array(R_median)), axis=1)
    eval_data[eval_stream_index] = eval_stream_data
    # signatures eval
    if len(signatures_columns) > 0:
        obs_signatures, pred_signatures = (
            obs[:, :, -signatures_size:], pred[:, :, -signatures_size:]
        )
        # streamflow eval columns
        eval_sign_index = ['rmse', 'kge', 'bias', 'R']
        eval_sign_index = concat_columns(eval_class, eval_sign_index, signatures_columns)
        # streamflow eval value
        cal_sign = CalcEvalIndex(obs=obs_signatures, sim=pred_signatures)
        nmse_mean, nmse_median = cal_sign.calc_nrmse()
        kge_mean, kge_median = cal_sign.calc_kge()
        bias_mean, bias_median = cal_sign.calc_bias()
        R_mean, R_median = cal_sign.calc_R()
        eval_sign_data = np.concatenate(
            (mean_array(nmse_mean), mean_array(kge_mean),
             mean_array(bias_mean), mean_array(R_mean),
             mean_array(nmse_median), mean_array(kge_median),
             mean_array(bias_median), mean_array(R_median)), axis=1)
        eval_data[eval_sign_index] = eval_sign_data
    # eval output
    eval_data.to_csv(log_file, index=False)
    t2 = time.time()
    print(f"Testing used time: {(t2 - t1) / 60} min")


def concat_columns(columns1, columns2, columns3):
    columns = []
    for column1 in columns1:
        for column2 in columns2:
            for column3 in columns3:
                column = f"{column3}_{column2}_{column1}"
                columns.append(column)
    return columns


def mean_array(array):
    return np.concatenate((np.mean(array, axis=0, keepdims=True), array), axis=0)
