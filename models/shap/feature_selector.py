import torch


class FeatureSelector:
    def __init__(self, baseline_data, feature_mask):
        """
        baseline_data: 背景数据均值 [n_features]
        feature_mask: 布尔掩码 [n_features]
        """
        if not isinstance(baseline_data, torch.Tensor):
            baseline_data = torch.FloatTensor(baseline_data)
        self.baseline_data = baseline_data
        self.mask = torch.BoolTensor(feature_mask).to(device=self.baseline_data.device)

    def __call__(self, x):
        """
        输入x: [batch_size, seq_len, n_features]
        输出: 仅目标特征可变，其他特征固定为基线值
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        baseline_data = self.baseline_data.repeat(x.shape[0], 1)
        # 替换目标特征为输入值
        masked_x = torch.where(
            self.mask.reshape(1, -1),
            x.to(device=self.baseline_data.device),
            baseline_data
        )
        return masked_x
