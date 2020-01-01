# src/data_loader/data_loaders.py

from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.base import BaseDataLoader
from src.dataset import SumStatDataset


class SumStatDataLoader(BaseDataLoader):
    """
    Summary statistics data loader.

    Parameters
    ----------
    data : torch.Tensor
        Torch tensor data to be converted to a Dataset object and loaded.
    batch_size : int
        Batch size (default 1)
    shuffle : bool
        Whether or not to shuffle the loaded data (default False)
    validation_split : float or int
        Proportion or number of batch to be used for validation (default 0).
    num_workers : int
        Number of workers to utilize (default 0).

    Attributes
    ----------
    data : torch.Tensor
        Raw data loaded into a torch Tensor.

    dataset : torch.utils.data
        Data processed into a Dataset object.
    """

    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int = 1,
        shuffle: bool = False,
        validation_split: Union[float, int] = 0.0,
        num_workers: int = 0,
    ) -> None:
        self.data: torch.Tensor = data
        self.dataset: Dataset = SumStatDataset(data=data)

        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=validation_split,
            num_workers=num_workers,
        )
