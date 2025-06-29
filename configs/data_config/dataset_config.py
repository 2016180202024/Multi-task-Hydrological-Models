# Define the shape of data
import numpy as np


class DataShapeConfig:

    def __init__(self, past_len, pred_len, features_name):
        # 输入输出时间序列长度
        # past_len [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
        # 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
        self.past_len = past_len  # TODO Length of formerly known runoff sequence
        self.pred_len = pred_len  # TODO Length of runoff sequence to be predicted
        self.tgt_len = past_len + pred_len
        self.src_len = past_len + pred_len
        # 输入特征长度
        self.dynamic_input_size = 16
        self.static_input_size = 37
        self.src_size = self.dynamic_input_size + self.static_input_size  # input attributes size
        # 输出特征
        self.features_name = features_name  # TODO
        self.forcing_columns = ['total_precipitation', 'potential_evaporation', 'snow_depth_water_equivalent',
                                'temperature_2m', 'temperature_2m_max', 'dewpoint_temperature_2m',
                                'surface_pressure', 'u_component_of_wind_10m', 'v_component_of_wind_10m',
                                'surface_net_solar_radiation', 'surface_net_thermal_radiation']
        streamflow_columns = np.array(['streamflow', 'baseflow_5', 'baseflow_60'])
        signatures_columns = np.array([
            'q_mean', 'runoff_ratio', 'stream_elas', 'fdc_slope', 'BFI_5', 'BFI_60', 'hfd_mean',
            'q_5', 'q_95', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'low_q_dur', 'zero_q_freq'
        ])
        features_index_dict = {
            'streamflow': [[0], []],
            'baseflow': [[i for i in range(3)], []],
            'baseflow_signatures': [[i for i in range(3)], [i for i in range(14)]],
            'runoff_ratio': [[i for i in range(3)], [i for i in range(14) if i not in [1]]],
            'q_mean_5_95': [[i for i in range(3)], [i for i in range(14) if i not in [0, 7, 8]]],
            'stream_elas': [[i for i in range(3)], [i for i in range(14) if i not in [2]]],
            'fdc_slope': [[i for i in range(3)], [i for i in range(14) if i not in [3]]],
            'BFI_5': [[i for i in range(3) if i not in [1]], [i for i in range(14) if i not in [4]]],
            'BFI_60': [[i for i in range(3) if i not in [2]], [i for i in range(14) if i not in [5]]],
            'hfd_mean': [[i for i in range(3)], [i for i in range(14) if i not in [6]]],
            'high_q_freq_dur': [[i for i in range(3)], [i for i in range(14) if i not in [9, 10]]],
            'low_q_freq_dur': [[i for i in range(3)], [i for i in range(14) if i not in [11, 12, 13]]],
        }
        self.out_features_index = features_index_dict[features_name]
        self.streamflow_columns = streamflow_columns[self.out_features_index[0]]
        self.signatures_columns = signatures_columns[self.out_features_index[1]]
        self.streamflow_size = len(self.out_features_index[0])
        self.signatures_size = len(self.out_features_index[1])
        # 输出特征长度
        self.tgt_size = self.streamflow_size + self.signatures_size
        self.use_baseflow = self.streamflow_size > 1  # TODO: whether to use baseflow
        self.use_signatures = self.signatures_size > 0  # TODO: whether to use static signature

        self.data_shape_info = f"[{past_len}-{pred_len},{self.src_size}-{self.tgt_size}]_{features_name}"
