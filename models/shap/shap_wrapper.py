import torch
from torch import nn


class ShapWrapper(nn.Module):
    def __init__(self, model, model_type, past_len, pred_len, selector,
                 x_time_len, x_static_len, y_time_len, y_static_len):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.past_len = past_len
        self.pred_len = pred_len
        self.selector = selector
        self.x_time_len = x_time_len
        self.x_static_len = x_static_len
        self.y_time_len = y_time_len
        self.y_static_len = y_static_len

    def forward(self, flat_input):
        """
        输入格式: [batch_size, all_days * input_dim + past_days * output_dim]
        输出格式: [batch_size, pred_days * output_dim]
        """
        # mask掉不需要输出shap的特征
        flat_input = self.selector(flat_input)
        x_day = self.past_len + self.pred_len
        y_day = self.past_len if 'LSTM' in self.model_type else self.past_len + self.pred_len
        samples_num = flat_input.shape[0]
        # 切分静态和动态部分
        x_time_index, x_index = x_day * self.x_time_len, x_day * self.x_time_len + self.x_static_len
        y_time_index, y_index = x_index + y_day * self.y_time_len, x_index + y_day * self.y_time_len + self.y_static_len
        x_seq_time = flat_input[:, :x_time_index].reshape(samples_num, x_day, self.x_time_len)
        x_seq_static = flat_input[:, x_time_index:x_index].reshape(samples_num, 1, self.x_static_len).repeat((1, x_seq_time.shape[1], 1))
        y_seq_time = flat_input[:, x_index:y_time_index].reshape(samples_num, y_day, self.y_time_len)
        y_seq_static = flat_input[:, y_time_index:y_index].reshape(samples_num, 1, self.y_static_len).repeat((1, y_seq_time.shape[1], 1))
        # 输入模型的数据
        x_seq = torch.cat([x_seq_time, x_seq_static], dim=2)
        y_seq = torch.cat([y_seq_time, y_seq_static], dim=2)
        # 输出模型的数据
        output = self.model(x_seq, y_seq)
        out_time = output[:, :, :self.y_time_len].mean(dim=1).reshape(samples_num, -1)
        out_static = output[:, :, self.y_time_len:].mean(dim=1).reshape(samples_num, -1)
        output = torch.cat([out_time, out_static], dim=1)
        return output
