from configs.data_config.dataset_config import DataShapeConfig


class LSTMMSVS2SConfig:
    model_name = "LSTMMSVS2S"
    decode_mode = None
    seq_len_e1 = 0
    output_len_e1 = 0
    input_size_e1 = 0
    hidden_size_e1 = 128
    seq_len_e2 = 0
    input_size_e2 = 0
    hidden_size_e2 = 128
    output_len_d = 0
    output_size = 0
    input_size_d = 256
    hidden_size_d = 128
    dropout_rate = 0.2

    def __init__(self, datashape_config: DataShapeConfig):
        self.seq_len_e1 = datashape_config.src_len
        self.output_len_e1 = datashape_config.pred_len
        self.input_size_e1 = datashape_config.src_size
        self.seq_len_e2 = datashape_config.past_len
        self.input_size_e2 = datashape_config.tgt_size
        self.output_len_d = datashape_config.pred_len
        self.output_size = datashape_config.tgt_size

    model_info = (f"{model_name}_"
                  f"[hs1_{hidden_size_e1},hs2_{hidden_size_e2},"
                  f"hs3_{hidden_size_d},dr_{dropout_rate}]")
