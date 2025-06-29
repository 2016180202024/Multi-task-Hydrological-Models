from configs.data_config.dataset_config import DataShapeConfig


class CNNLSTMConfig:
    model_name = "CNNLSTM"
    decode_mode = None
    seq_len_e = 0
    output_len_e = 0
    input_size_e = 0
    hidden_size_e = 128
    output_len_d = 0
    output_size = 0
    input_size_d = 256
    hidden_size_d = 128
    dropout_rate = 0.2
    conv_kernel_size = 1
    conv_size = 0

    def __init__(self, datashape_config: DataShapeConfig):
        self.seq_len_e = datashape_config.src_len
        self.output_len_e = datashape_config.pred_len
        self.input_size_e = datashape_config.src_size
        self.output_len_d = datashape_config.pred_len
        self.output_size = datashape_config.tgt_size
        self.conv_size = datashape_config.signatures_size

    model_info = (f"{model_name}_"
                  f"[hs1_{hidden_size_e},hs3_{hidden_size_d},dr_{dropout_rate}]")
