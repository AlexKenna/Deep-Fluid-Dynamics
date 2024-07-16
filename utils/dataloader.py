import numpy as np
import h5py
import pickle
import torch
from torch.utils.data import Dataset


class FluidDataset(Dataset):
    def __init__(self, filename: str, train: bool = True):
        self.filename = filename
        self.train = train

    def __len__(self):
        return 1500 if self.train else 500

    def __getitem__(self, idx):
        starting_pos = 0 if self.train else 1500

        with h5py.File(f"{self.filename}.h5", "r") as datafile:

            # Convert data to correct format
            data = torch.tensor(datafile["dataset"][starting_pos + idx])
            data = data.type(torch.float)
            data = data.permute(2, 3, 0, 1)

            # Construct grid
            with open(f"{self.filename}.pkl", 'rb') as handle:
                info = pickle.load(handle)
                grid = np.array(np.meshgrid(info["coords"]["x"], info["coords"]["y"]))
                grid = torch.tensor(grid).type(torch.float)
                grid = grid.permute(1, 2, 0)

            return data, grid
