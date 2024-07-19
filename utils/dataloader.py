import numpy as np
import h5py
import pickle

from enum import Enum

import torch
from torch.utils.data import Dataset


class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATION = 3


class FluidDataset(Dataset):
    def __init__(self, file_name: str, dataset_type: DatasetType = DatasetType.TRAIN):
        self.file_name = file_name
        self.dataset_type = dataset_type

        self._train_len = 1500
        self._val_len = 250
        self._test_len = 250

    def __len__(self):
        if self.dataset_type == DatasetType.TRAIN:
            return self._train_len
        elif self.dataset_type == DatasetType.VALIDATION:
            return self._val_len
        elif self.dataset_type == DatasetType.TEST:
            return self._test_len

    def __getitem__(self, idx):

        if self.dataset_type == DatasetType.TRAIN:
            starting_pos = 0
        elif self.dataset_type == DatasetType.VALIDATION:
            starting_pos = self._train_len
        elif self.dataset_type == DatasetType.TEST:
            starting_pos = self._train_len + self._val_len

        with h5py.File(f"{self.file_name}.h5", "r") as datafile:

            # Convert data to correct format
            data = torch.tensor(datafile["dataset"][starting_pos + idx])
            data = data.type(torch.float)
            data = data.permute(2, 3, 1, 0)

            # Construct grid
            with open(f"{self.file_name}.pkl", "rb") as handle:
                info = pickle.load(handle)
                grid = np.array(np.meshgrid(info["coords"]["x"], info["coords"]["y"]))
                grid = torch.tensor(grid).type(torch.float)
                grid = grid.permute(1, 2, 0)

            return data, grid
