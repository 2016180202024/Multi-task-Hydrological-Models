import gc
import time
import numpy as np
import torch
from utils.model.eval import eval_model
from utils.model.logs import BestModelLog, BoardWriter
from utils.model.tools import count_parameters


def train_full(model, decode_mode, train_loader, val_loader,
               optimizer, scheduler, loss_func, n_epochs, device,
               saving_root, using_board, use_train_eval,
               use_baseflow, use_signatures, streamflow_size, signatures_size):
    print(f"Parameters count:{count_parameters(model)}")
    # log_train.csv
    log_file = saving_root / "log_train.csv"
    with open(log_file, "wt") as f:
        f.write(f"parameters_count:{count_parameters(model)}\n")
        f.write("epoch,train_loss,train_sf_nse,train_bf_nse[0],train_bf_nse[1],train_sg_mse,"
                "val_sf_nse,val_bf_nse[0],val_bf_nse[1],val_sg_mse,weight[0],weight[1],weight[2]\n")
    # tensorboard
    if using_board:
        tb_root = saving_root / "tb_log"
        writer = BoardWriter(tb_root)
    else:
        writer = None
    # model.pkl
    min_loss = BestModelLog(model, saving_root, "min_loss", high_better=False)
    max_sf_nse = BestModelLog(model, saving_root, "max_sf_nse", high_better=True)
    max_bf_nse = BestModelLog(model, saving_root, "max_bf_nse", high_better=True)
    min_sg_mse = BestModelLog(model, saving_root, "min_sg_mse", high_better=False)
    # Train
    for i in range(n_epochs):
        t1 = time.time()
        print(f"Training progress: {i + 1} / {n_epochs}")
        # 训练损失
        gc.collect()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 loss_func, decode_mode, device, streamflow_size, signatures_size)
        train_loss = np.sum(train_loss)
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        loss_weights = loss_func.get_weights_numpy()
        weights[:len(loss_weights)] = loss_weights
        # 评价模型
        train_sf_nse, train_bf_nse, train_sg_mse = 0, [0, 0], 0
        if use_train_eval:
            train_sf_nse, train_bf_nse, train_sg_mse = (
                eval_model(model, train_loader, decode_mode, device, streamflow_size, signatures_size))
        val_sf_nse, val_bf_nse, val_sg_mse = (
            eval_model(model, val_loader, decode_mode, device, streamflow_size, signatures_size))
        for v_i in range(2-len(val_bf_nse)):
            val_bf_nse.append(0)
        if writer is not None:
            writer.write_board(f"train_loss", metric_value=np.sum(train_loss), epoch=i)
            writer.write_board(f"val_sf_nse", metric_value=val_sf_nse, epoch=i)
            writer.write_board("weight[0]", metric_value=weights[0], epoch=i)
            if use_train_eval:
                writer.write_board(f"train_sf_nse", metric_value=train_sf_nse, epoch=i)
                writer.write_board("train_bf_nse[0]", metric_value=train_bf_nse[0], epoch=i)
                writer.write_board("train_bf_nse[1]", metric_value=train_bf_nse[1], epoch=i)
                writer.write_board("train_sg_mse", metric_value=train_sg_mse, epoch=i)
            if use_baseflow:
                writer.write_board(f"val_bf_nse[0]", metric_value=val_bf_nse[0], epoch=i)
                writer.write_board(f"val_bf_nse[1]", metric_value=val_bf_nse[1], epoch=i)
                writer.write_board("weight[1]", metric_value=weights[1], epoch=i)
            if use_signatures:
                writer.write_board(f"val_sg_mse", metric_value=val_sg_mse, epoch=i)
                writer.write_board("weight[2]", metric_value=weights[2], epoch=i)
        with open(log_file, "at") as f:
            f.write(f"{i},{np.sum(train_loss)},{train_sf_nse},{train_bf_nse[0]},{train_bf_nse[1]},{train_sg_mse},"
                    f"{val_sf_nse},{val_bf_nse[0]},{val_bf_nse[1]},{val_sg_mse},"
                    f"{weights[0]},{weights[1]},{weights[2]}\n")
        min_loss.update(model, np.sum(train_loss), i)
        max_sf_nse.update(model, val_sf_nse, i)
        if use_baseflow:
            max_bf_nse.update(model, np.mean(val_bf_nse), i)
        if use_signatures:
            min_sg_mse.update(model, val_sg_mse, i)
        t2 = time.time()
        print(f"train_loss:{np.sum(train_loss):.4f}, val_sf_nse:{val_sf_nse:.4f}, "
              f"val_bf_nse:{np.mean(val_bf_nse):.4f}, val_sg_mse:{val_sg_mse:.4f}, "
              f"weight[0]:{weights[0]:.4f}, weight[1]:{weights[1]:.4f}, weight[2]:{weights[2]:.4f}")
        print(f"Training used time: {(t2 - t1) / 60} min")


# Train utilities
def train_epoch(model, data_loader, optimizer, scheduler,
                loss_func, decode_mode, device, streamflow_size, signatures_size):
    """
    Train model for a single epoch.
    :param model: A torch.nn.Module implementing the Transformer model.
    :param data_loader: A PyTorch DataLoader, providing the trainings data in mini batches.
    :param optimizer: One of PyTorch optimizer classes.
    :param scheduler: scheduler of learning rate.
    :param loss_func: The loss function to minimize.
    :param decode_mode: decoding mode in Transformer.
    :param device: device for data and models
    :param signatures_size:
    :param streamflow_size:
    """
    # set model to train mode (important for dropout)
    model.train()
    train_loss = None
    cnt = 0
    for x_seq, y_seq_past, y_seq_future, y_stds in data_loader:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        x_seq, y_seq_past, y_seq_future = x_seq.to(device), y_seq_past.to(device), y_seq_future.to(device)
        batch_size = y_seq_past.shape[0]
        past_len = y_seq_past.shape[1]
        pred_len = y_seq_future.shape[1]
        tgt_len = past_len + pred_len
        tgt_size = y_seq_future.shape[2]

        if decode_mode == "NAR":
            enc_inputs = x_seq
            dec_inputs = torch.zeros((batch_size, tgt_len, tgt_size)).to(device)
            dec_inputs[:, :-pred_len, :] = y_seq_past
            # get model predictions
            y_hat = model(enc_inputs, dec_inputs)
            y_hat = y_hat[:, -pred_len:, :]
        elif decode_mode == "AR":
            enc_inputs = x_seq
            dec_inputs = torch.cat((y_seq_past, y_seq_future), dim=1)
            # get model predictions
            y_hat = model(enc_inputs, dec_inputs)
            y_hat = y_hat[:, -pred_len - 1:-1, :]
        else:  # Model is not Transformer
            y_hat = model(x_seq, y_seq_past)

        # calculate loss
        # need y_stds of each basin to calculate NSELoss
        y_stds = y_stds.to(device)
        # 仅有streamflow
        if streamflow_size == 1 and signatures_size == 0:
            loss = loss_func(y_hat, y_seq_future, y_stds)
        # 有streamflow和baseflow
        elif streamflow_size > 1 and signatures_size == 0:
            loss = loss_func(y_hat[:, :, :1], y_seq_future[:, :, :1], y_stds[:, :1],
                             y_hat[:, :, 1:], y_seq_future[:, :, 1:], y_stds[:, 1:])
        # 有streamflow和signatures
        elif streamflow_size == 1 and signatures_size > 0:
            loss = loss_func(y_hat[:, :, :1], y_seq_future[:, :, :1], y_stds[:, :1],
                             None, None, None,
                             y_hat[:, :, -signatures_size:], y_seq_future[:, :, -signatures_size:])
        # 有streamflow、baseflow和signatures
        else:
            loss = loss_func(y_hat[:, :, :1], y_seq_future[:, :, :1], y_stds[:, :1],
                             y_hat[:, :, 1:streamflow_size], y_seq_future[:, :, 1:streamflow_size], y_stds[:, 1:streamflow_size],
                             y_hat[:, :, -signatures_size:], y_seq_future[:, :, -signatures_size:])
        # get all loss
        weights = loss_func.get_weights()
        tasks_loss = torch.mul(loss, weights)
        # calculate gradients
        torch.sum(tasks_loss).backward(retain_graph=True)
        # update the weights
        loss_func.update_weights(tasks_loss)
        # step forward
        optimizer.step()
        # record train loss list
        train_loss_temp = tasks_loss.clone().detach().cpu().numpy()
        train_loss = train_loss_temp if train_loss is None else train_loss + train_loss_temp
        cnt += 1
    # renormalize weights
    weights = loss_func.get_weights()
    normalize_coeff = len(weights.data) / torch.sum(weights.data, dim=0)
    loss_func.get_weights().data = weights.data * normalize_coeff
    # step forward learning rate
    scheduler.step()
    return train_loss / cnt
