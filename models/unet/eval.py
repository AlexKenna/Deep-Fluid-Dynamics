import pickle
import seaborn
import xarray
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.unet.unet import UNet2d
from utils.dataloader import DatasetType, FluidDataset


def run_evaluation(
    data_file,
    lookback_steps,
    device,
    batch_size,
    base_data_path="data/dataset/",
    base_model_path="models/fno/pretrained/"
):

    # --------------------------------------------------------------#
    # Load data
    # --------------------------------------------------------------#

    data_path = base_data_path + data_file
    test_data = FluidDataset(file_name=data_path, dataset_type=DatasetType.TEST)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


    # --------------------------------------------------------------#
    # Model
    # --------------------------------------------------------------#

    model = nn.DataParallel(UNet2d(in_channels=2, out_channels=2))
    model_name = "UNet_" + data_file
    model_path = base_model_path + model_name + ".pt"

    print("Restoring model from file...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    

    # --------------------------------------------------------------#
    # Evaluate Test Errors
    # --------------------------------------------------------------#

    test_errors = None

    with torch.no_grad():

        for data, grid in test_loader:

            # data: [b, x1, x2, t, u]
            # grid: [b, x1, x2, dims]

            total_steps = data.shape[-2]
            num_predictions = total_steps - lookback_steps

            if test_errors is None:
                test_errors = np.zeros(num_predictions)

            data = data.to(device)
            grid = grid.to(device)
            preds = data[..., :lookback_steps, :]

            for start_step in range(num_predictions):

                current_step = start_step + lookback_steps

                x = preds[..., -lookback_steps:, :].flatten(-2)
                y = data[..., current_step : current_step + 1, :]

                y_pred = model(x, grid)
                preds = torch.concat([preds, y_pred], dim=-2)

                loss = nn.functional.mse_loss(y_pred, y)
                test_errors[start_step] += loss.item() * data.shape[0]

    test_errors /= len(test_loader.dataset)


    # --------------------------------------------------------------#
    # Plot Test Errors
    # --------------------------------------------------------------#

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4))

    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean MSE Loss")

    ax.plot(list(range(len(test_errors))), test_errors)
    ax.grid()

    plt.show()


    # --------------------------------------------------------------#
    # Evaluate Example Output
    # --------------------------------------------------------------#

    data, grid = next(iter(test_loader))
    data = data.to(device)[0].unsqueeze(0)
    grid = grid.to(device)[0].unsqueeze(0)
    preds = data[..., :lookback_steps, :]

    total_steps = data.shape[-2]
    num_predictions = total_steps - lookback_steps

    for start_step in range(num_predictions):

        current_step = start_step + lookback_steps

        x = preds[..., -lookback_steps:, :].flatten(-2)
        y = data[..., current_step : current_step + 1, :]

        y_pred = model(x, grid)
        preds = torch.concat([preds, y_pred], dim=-2)


    # --------------------------------------------------------------#
    # Format Example Data
    # --------------------------------------------------------------#

    def vorticity(ds):
        return (ds.v.differentiate("x") - ds.u.differentiate("y")).rename("vorticity")
    
    with open(f"{base_data_path}{data_file}.pkl", "rb") as handle:
        info = pickle.load(handle)

    # Construct the data for the
    true = data.permute(0, 4, 3, 1, 2).squeeze(0, 1).cpu().detach().numpy()
    preds = preds.permute(0, 4, 3, 1, 2).squeeze(0, 1).cpu().detach().numpy()

    pred_ds = xarray.Dataset(
        {
            "u": (("time", "x", "y"), preds[0, ...]),
            "v": (("time", "x", "y"), preds[1, ...]),
        },
        coords={
            "time": info["details"]["dt"] 
                    * info["details"]["inner_steps"]
                    * np.arange(info["details"]["outer_steps"])
        }
    )

    true_ds = xarray.Dataset(
        {
            "u": (("time", "x", "y"), true[0, ...]),
            "v": (("time", "x", "y"), true[1, ...]),
        },
        coords={
            "time": info["details"]["dt"] 
                    * info["details"]["inner_steps"]
                    * np.arange(info["details"]["outer_steps"])
        }
    )

    pred_vorticity = pred_ds.pipe(vorticity)
    true_vorticity = true_ds.pipe(vorticity)

    
    # --------------------------------------------------------------#
    # Plot Vorticity Example Output
    # --------------------------------------------------------------#

    print("Vorticity output:")

    vmin = min(pred_vorticity.min().item(), true_vorticity.min().item())
    vmax = min(pred_vorticity.max().item(), true_vorticity.max().item())

    ncols = 6
    fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(12, 4), layout="compressed")

    # Plot true vorticity
    for col in range(ncols):
        idx = col * int(((true_vorticity.shape[0] - 1) / (ncols - 1)))
        ax[0][col].set_title(f"t={true_vorticity[idx]['time'].item()}")
        ax[0][col].imshow(
            true_vorticity[idx], cmap=seaborn.cm.icefire,
            vmin=vmin, vmax=vmax
        )
        ax[0][col].axis("off")

    # Plot predicted vorticity
    for col in range(1, ncols):
        idx = col * int(((pred_vorticity.shape[0] - 1) / (ncols - 1)))
        im = ax[1][col].imshow(
            pred_vorticity[idx], cmap=seaborn.cm.icefire,
            vmin=vmin, vmax=vmax
        )
        ax[1][col].axis("off")

    ax[0][0].set_title("Initial Condition")
    ax[1][0].axis("off")
    ax[1][0].text(0.5, 0.5, "Prediction", ha="center", va="center", fontsize=12)

    fig.colorbar(im, ax=ax)
    plt.show()


    # --------------------------------------------------------------#
    # Plot X-Velocity Example Output
    # --------------------------------------------------------------#

    print("x-velocity output:")

    vmin = min(pred_ds["u"].min().item(), true_ds["u"].min().item())
    vmax = min(pred_ds["u"].max().item(), true_ds["u"].max().item())

    ncols = 6
    fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(12, 4), layout="compressed")

    for col in range(ncols):
        idx = col * int(((true_ds["u"].shape[0] - 1) / (ncols - 1)))
        ax[0][col].set_title(f"t={true_ds['u'][idx]['time'].item()}")
        ax[0][col].imshow(true_ds["u"][idx], cmap=seaborn.cm.icefire, vmin=vmin, vmax=vmax)
        ax[0][col].axis("off")

    for col in range(1, ncols):
        idx = col * int(((pred_ds["u"].shape[0] - 1) / (ncols - 1)))
        im = ax[1][col].imshow(pred_ds["u"][idx], cmap=seaborn.cm.icefire, vmin=vmin, vmax=vmax)
        ax[1][col].axis("off")

    ax[0][0].set_title("Initial Condition")
    ax[1][0].axis("off")
    ax[1][0].text(0.5, 0.5, "Prediction", ha="center", va="center", fontsize=12)

    fig.colorbar(im, ax=ax)
    plt.show()


    # --------------------------------------------------------------#
    # Plot Y-Velocity Example Output
    # --------------------------------------------------------------#

    print("y-velocity output:")

    vmin = min(pred_ds["v"].min().item(), true_ds["v"].min().item())
    vmax = min(pred_ds["v"].max().item(), true_ds["v"].max().item())

    ncols = 6
    fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(12, 4), layout="compressed")

    for col in range(ncols):
        idx = col * int(((true_ds["v"].shape[0] - 1) / (ncols - 1)))
        ax[0][col].set_title(f"t={true_ds['v'][idx]['time'].item()}")
        ax[0][col].imshow(true_ds["v"][idx], cmap=seaborn.cm.icefire, vmin=vmin, vmax=vmax)
        ax[0][col].axis("off")

    for col in range(1, ncols):
        idx = col * int(((pred_ds["v"].shape[0] - 1) / (ncols - 1)))
        im = ax[1][col].imshow(pred_ds["v"][idx], cmap=seaborn.cm.icefire, vmin=vmin, vmax=vmax)
        ax[1][col].axis("off")

    ax[0][0].set_title("Initial Condition")
    ax[1][0].axis("off")
    ax[1][0].text(0.5, 0.5, "Prediction", ha="center", va="center", fontsize=12)

    fig.colorbar(im, ax=ax)
    plt.show()

