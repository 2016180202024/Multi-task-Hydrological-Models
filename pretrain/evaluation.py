import os
import shutil
from pathlib import Path

import pandas as pd

from configs.data_config.path_config import PathConfig
from utils.model.metrics import CalcEvalIndex
from utils.process_data.get_used_basins import get_used_basins


def evaluate(used_basins, dir_path):
    print('--------------------------------------')
    print(f'[{os.path.basename(dir_path)}] is evaluating!')
    out_dir = Path(dir_path) / 'predict'
    if not out_dir.exists():
        return
    out_file = Path(dir_path) / 'predict' / 'evaluation.csv'
    # 如果有已处理的gauge，读取然后跳过这些gauge
    columns = ['rmse', 'nse', 'kge', 'fhv', 'fms', 'flv', 'bias', 'tpe5', 'R']
    eval_data = pd.DataFrame(columns=(['gauge_id'] + columns)).set_index('gauge_id')
    exists_gauge = []
    if out_file.exists():
        eval_data = pd.read_csv(out_file).set_index('gauge_id')
        exists_gauge = eval_data.index.tolist()
    # 计算所有len
    all_length = len(used_basins)
    temp_length = 0
    # 遍历每一个camels和gauge
    try:
        for gauge_id in used_basins:
            data_dict = {}
            temp_length += 1
            if gauge_id in exists_gauge:
                print(f'[{temp_length}/{all_length}] [{gauge_id}] is exists!')
                continue
            print(f'[{temp_length}/{all_length}] [{gauge_id}]')
            camels = gauge_id.split('_')[0]
            data_file_path = Path(dir_path) / 'predict' / camels / f'{gauge_id}_pred.csv'
            guage_data = pd.read_csv(data_file_path)
            obs_streamflow, pred_streamflow = guage_data['streamflow'], guage_data['streamflow_pred']
            cal_stream = CalcEvalIndex(obs=obs_streamflow, sim=pred_streamflow, calc_median=False)
            data_dict['rmse'], _ = cal_stream.calc_rmse()
            data_dict['nse'], _ = cal_stream.calc_nse()
            data_dict['kge'], _ = cal_stream.calc_kge()
            data_dict['fhv'] = cal_stream.calc_fdc_fhv()
            data_dict['fms'] = cal_stream.calc_fdc_fms()
            data_dict['flv'] = cal_stream.calc_fdc_flv()
            data_dict['bias'], _ = cal_stream.calc_bias()
            data_dict['tpe5'], _ = cal_stream.calc_tpe_1D(5)
            data_dict['R'], _ = cal_stream.calc_R()
            eval_data.loc[gauge_id] = data_dict
    except Exception as e:
        print(e)
    finally:
        eval_data.to_csv(out_file, index_label='gauge_id')


def camels_evaluate():
    used_basins = get_used_basins()
    dir_path = Path(PathConfig.model_path)
    for day_dir in dir_path.glob('*'):
        if not Path(day_dir).name.isdigit():
            continue
        day = int(Path(day_dir).name)
        for pred_day in range(5, 35, 5):
            for model_dir in Path(day_dir).glob(f'*[{day-pred_day}-{pred_day},*'):
                evaluate(used_basins, model_dir)


def camels_evaluate1():
    used_basins = get_used_basins()
    root_path = Path(PathConfig.model_path)
    pred_day = 5
    for day in [30, 60, 90, 120]:
        for model_dir in Path(root_path / str(day)).glob(f'*[{day-pred_day}-{pred_day},*'):
            evaluate(used_basins, model_dir)


def rename_dict():
    root_path = Path(PathConfig.model_path)
    for r_dir_path in root_path.glob('*'):
        for dir_path in r_dir_path.glob('*@seed1234'):
            if 'bfFalse_snFalse' in str(dir_path):
                new_dir_path = str(dir_path).replace('bfFalse_snFalse', 'streamflow')
                Path(dir_path).rename(new_dir_path)
                break
            if 'bfTrue_snFalse' in str(dir_path):
                new_dir_path = str(dir_path).replace('bfTrue_snFalse', 'baseflow')
                Path(dir_path).rename(new_dir_path)
                break
            if 'bfTrue_snTrue' in str(dir_path):
                new_dir_path = str(dir_path).replace('bfTrue_snTrue', 'baseflow_signatures')
                Path(dir_path).rename(new_dir_path)
                break


def sum_eval_result():
    dir_path = Path(PathConfig.model_path)
    out_path = dir_path / 'output'
    for day_dir in dir_path.glob('*'):
        if not Path(day_dir).name.isdigit():
            continue
        day = int(Path(day_dir).name)
        # for pred_day in range(5, 35, 5):
        for pred_day in [5]:
            for model_dir in Path(day_dir).glob(f'*[{day-pred_day}-{pred_day},*'):
                out_dir = out_path / str(day) / model_dir.name
                out_dir.mkdir(exist_ok=True, parents=True)
                eval_path = model_dir / 'predict' / 'evaluation.csv'
                if eval_path.exists():
                    out_file = out_dir / 'evaluation.csv'
                    shutil.copy(eval_path, out_file)
                shap_path = model_dir / 'predict' / 'shap.plk'
                if shap_path.exists():
                    out_file = out_dir / 'shap.plk'
                    shutil.copy(shap_path, out_file)
    shutil.make_archive(
        base_name=str(out_path),  # 输出文件的路径（不含扩展名）
        format='zip',  # 压缩格式
        root_dir=out_path  # 要压缩的文件夹路径
    )


if __name__ == '__main__':
    # rename_dict()
    # camels_evaluate()
    sum_eval_result()
