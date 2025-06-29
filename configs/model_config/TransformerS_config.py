from configs.data_config.dataset_config import DataShapeConfig


class TransformerSConfig:
    model_name = "TransformerS"
    decode_mode = "NAR"  # NAR or AR
    d_model = 64  # embedding size, d_model matters a lot
    n_heads = 4  # number of heads in multi-head attention
    assert d_model % n_heads == 0
    d_head = d_model // n_heads  # dimension of each head in multi-head attention
    n_encoder_layers = 4  # number of encoder layer
    n_decoder_layers = 4  # number of decoder layer
    assert n_encoder_layers == n_decoder_layers  # we assert they are equal here
    d_ff = 256  # feedforward dimension
    dropout_rate = 0.1  # dropout rate

    def __init__(self, datashape_config: DataShapeConfig):
        self.src_len = datashape_config.src_len
        self.tgt_len = datashape_config.past_len + datashape_config.pred_len
        self.pred_len = datashape_config.pred_len
        self.src_size = datashape_config.src_size
        self.tgt_size = datashape_config.tgt_size

    model_info = (f"{model_name}_{decode_mode}_"
                  f"[{d_model}-{n_heads}-{n_encoder_layers}-{d_ff}-{dropout_rate}]")
