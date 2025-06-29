import torch

from utils.model.gradnorm import GradNorm


class StreamflowLoss(torch.nn.Module):
    """Calculate (batch-wise) Streamflow Loss.

    Contains the NSE Loss of timeseries streamflow and the RMSE loss of static signatures.

    Parameters:
    -----------
    weight : float
        The weight of MSE loss of static signatures to NSE Loss of timeseries streamflow.
    """

    def __init__(self, model: torch.nn.Module = None,
                 weights: torch.tensor = None,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.gradnorm = None
        self.weights = torch.tensor([1], dtype=torch.float32, device=device)
        if weights is not None:
            # 采用gradnorm
            self.weights = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32, device=device))
            self.gradnorm = GradNorm(model)

    def update_weights(self, loss):
        if self.gradnorm is not None:
            if self.gradnorm.get_init_task_loss() is None:
                self.gradnorm.update_init_task_loss(loss.clone().detach().to(self.device))
            self.gradnorm.forward(loss, self.weights)

    def forward(self, sf_pred: torch.Tensor = None, sf_true: torch.Tensor = None, sf_stds: torch.Tensor = None,
                bf_pred: torch.Tensor = None, bf_true: torch.Tensor = None, bf_stds: torch.Tensor = None,
                sg_pred: torch.Tensor = None, sg_true: torch.Tensor = None):
        """
        Parameters
        ----------
        sf_pred : torch.Tensor
            Tensor containing the streamflow prediction [batch_size, tgt_len, streamflow_size].
        sf_true : torch.Tensor
            Tensor containing the true streamflow values
        sf_stds : torch.Tensor
            Tensor containing the streamflow std (calculate over training period) of each sample
        bf_pred : torch.Tensor
            Tensor containing the baseflow prediction [batch_size, tgt_len, streamflow_size].
        bf_true : torch.Tensor
            Tensor containing the true baseflow values
        bf_stds : torch.Tensor
            Tensor containing the baseflow std (calculate over training period) of each sample
        sg_pred : torch.Tensor
            Tensor containing the signatures prediction [batch_size, signatures_len, signatures_size].
        sg_true : torch.Tensor
            Tensor containing the true signatures values

        Returns
        -------
        torch.Tenor
            The (batch-wise) Streamflow Loss
        """
        sf_loss = self.calc_nse(sf_pred, sf_true, sf_stds)
        loss = [sf_loss]
        if bf_pred is not None:
            bf_loss = self.calc_nse(bf_pred, bf_true, bf_stds)
            loss.append(0.1 * bf_loss)
        if sg_pred is not None:
            sg_loss = self.calc_rmse(sg_pred, sg_true)
            loss.append(0.1 * sg_loss)
        loss = torch.stack(loss)
        return loss

    def calc_nse(self, pred, true, stds, eps: float = 0.1):
        nse_error = (pred - true) ** 2
        nse_weights = 1 / (stds + eps) ** 2
        nse_weights = nse_weights.reshape(nse_weights.shape[0], 1, nse_weights.shape[1])
        nse_weights = nse_weights.repeat(1, pred.shape[1], 1)
        streamflow_loss = torch.nanmean(nse_weights * nse_error)
        return streamflow_loss

    def calc_rmse(self, pred, true):
        pred = pred.nanmean(dim=1, keepdim=False)
        true = true.nanmean(dim=1, keepdim=False)
        rmse_error = torch.sqrt(torch.nanmean((pred - true) ** 2))
        return rmse_error

    def get_weights_numpy(self):
        return self.weights.clone().detach().cpu().numpy()

    def get_weights(self):
        return self.weights
