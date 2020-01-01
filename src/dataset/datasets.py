# src/dataset/datasets.py

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SumStatDataset(Dataset):
    """
    The summary statistics dataset object.

    Parameters
    ----------
    data : torch.Tensor
        Data to be wrapped as Dataset.

    Attributes
    ----------
    X : torch.Tensor
        Torch tensor containing the data.
    data_len : int
        The number of data rows.
    features : int
        The number of features (columns).
    """

    def __init__(self, data: torch.Tensor) -> None:
        self.X: torch.Tensor = data
        self.data_len: int = self.X.shape[0]
        self.features: int = self.X.shape[1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve item in Dataset by index.

        Parameters
        ----------
        idx : int
            The index to extract.

        Returns
        -------
        tuple of torch.Tensor
            Returns a tuple of tensors (sample and target). In this instance, the target is a dummy
            tensor as the custom "loss" function does not use it.
        """
        return self.X[idx], torch.tensor([8])  # dummy target

    def __len__(self) -> int:
        """
        Return the length of the data (number of rows).

        Returns
        -------
        int
            Number of rows in data.
        """
        return self.data_len
