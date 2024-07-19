import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from timeit import default_timer

from models.fno.fno import FNO2d
from utils.dataloader import DatasetType, FluidDataset


def run_training(
    data_file,
    lookback_steps,
    epochs,
    learning_rate,
    device,
    batch_size,
    scheduler_step,
    scheduler_gamma,
    model_update,
    continue_training=False,
    base_data_path="data/dataset/",
    base_model_path="models/fno/pretrained/"
):

    print(f"Device: {device}, epochs: {epochs}, learning rate: {learning_rate}, scheduler step: {scheduler_step}, scheduler gamma: {scheduler_gamma}")

    # --------------------------------------------------------------#
    # Load data
    # --------------------------------------------------------------#

    data_path = base_data_path + data_file

    train_data = FluidDataset(file_name=data_path, dataset_type=DatasetType.TRAIN)
    val_data = FluidDataset(file_name=data_path, dataset_type=DatasetType.VALIDATION)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------------------#
    # Model
    # --------------------------------------------------------------#

    model = FNO2d(in_channels=2, out_channels=2).to(device)
    model_name = "FNO_" + data_file
    model_path = base_model_path + model_name + ".pt"

    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=scheduler_step, gamma=scheduler_gamma
    )
    loss_fn = nn.MSELoss()

    start_epoch = 0
    loss_val_min = np.infty

    train_losses = []
    val_losses = []

    if continue_training:
        print("Restoring model from file...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.train()

        # Load optimizer state dict
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        for state in optimiser.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint["epoch"]
        loss_val_min = checkpoint["loss"]
        train_losses = checkpoint["training_losses"]
        val_losses = checkpoint["validation_losses"]


    # --------------------------------------------------------------#
    # Training
    # --------------------------------------------------------------#

    for epoch in range(start_epoch, epochs):

        model.train()
        start_time = default_timer()

        # Training

        total_train_loss = 0
        train_iters = 0

        for data, grid in train_loader:

            # data: [b, x1, x2, t, u]
            # grid: [b, x1, x2, dims]

            total_steps = data.shape[-2]
            num_predictions = total_steps - lookback_steps

            data = data.to(device)
            grid = grid.to(device)

            for start_step in range(num_predictions):

                train_iters += 1
                current_step = start_step + lookback_steps

                x = data[..., start_step:current_step, :].flatten(-2)
                y = data[..., current_step : current_step + 1, :]
                y_pred = model(x, grid)

                loss = loss_fn(y_pred, y)
                optimiser.zero_grad()
                loss.backward()
                total_train_loss += loss.item()
                optimiser.step()

        scheduler.step()

        # Validation

        if epoch % model_update == 0:
            with torch.no_grad():

                total_val_loss = 0
                val_iters = 0

                for data, grid in val_loader:

                    total_steps = data.shape[-2]
                    num_predictions = total_steps - lookback_steps

                    data = data.to(device)
                    grid = grid.to(device)

                    for start_step in range(num_predictions):

                        val_iters += 1
                        current_step = start_step + lookback_steps

                        x = data[..., start_step:current_step, :].flatten(-2)
                        y = data[..., current_step : current_step + 1, :]
                        y_pred = model(x, grid)

                        loss = loss_fn(y_pred, y)
                        total_val_loss += loss.item()

        # Logging

        stop_time = default_timer()
        train_losses.append(total_train_loss / train_iters)
        val_losses.append(total_val_loss / val_iters)
        print(f"Epoch: {epoch}, time: {stop_time - start_time:.5f}, train loss: {train_losses[-1]:.6f}, val loss: {val_losses[-1]:.6f}")

        # Store model

        if val_losses[-1] < loss_val_min:
            loss_val_min = val_losses[-1]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "loss": loss_val_min,
                    "training_losses": train_losses,
                    "validation_losses": val_losses
                },
                model_path,
            )


    # --------------------------------------------------------------#
    # Plots
    # --------------------------------------------------------------#    

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    x_plot = list(range(len(train_losses)))
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("MSE Loss")
    ax[0].plot(x_plot, train_losses, color="#636EFA")
    ax[0].plot(x_plot, val_losses, color="#EF553B")
    ax[0].legend(["Training Loss", "Validation Loss"])
    ax[0].grid()

    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("MSE Loss")
    ax[1].plot(x_plot[2:], train_losses[2:], color="#636EFA")
    ax[1].plot(x_plot[2:], val_losses[2:], color="#EF553B")
    ax[1].legend(["Training Loss", "Validation Loss"])
    ax[1].grid()

    fig.tight_layout()
    plt.show()
