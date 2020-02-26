# base/base_data_loader.py

from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate  # type: ignore
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders.

    Parameters
    ----------
    dataset : torch.Dataset
        Dataset to be loaded.
    batch_size : int
        Batch size (default 1).
    shuffle : bool
        Whether or not to shuffle the loaded data (default False).
    validation_split : float or int
        Proportion or number of batch to be used for validation (default 0).
    num_workers : int
        Number of workers to utilize (default 0).
    collate_fn : typing.Callable
        Mini-batch merge function (default torch.utils.data.dataloader.default_collate)

    Attributes
    ----------
    batch_idx : int
        Current batch index.
    init_kwargs : dict of {str, Any}:
        Torch DataLoader keyword arguments.
    n_samples : int
        Number of samples.
    sampler : torch.utils.data.sampler.Sampler of int or None
        Training set sampler for batches.
    shuffle : bool
        Whether or not to shuffle the loaded data.
    valid_sampler : torch.utils.data.sampler.Sampler of int or None
        Validation set sampler for batches.
    validation_split : float or int
        The split proportion or the exact number of samples of the validation set.

    Methods
    -------
    split_validation()
        Loads the validation data if the data is split into training and validation sets.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        validation_split: Union[float, int] = 0.0,
        num_workers: int = 0,
        collate_fn: Callable = default_collate,
        test_set: bool = False,
    ) -> None:
        self.validation_split: Union[float, int] = validation_split
        self.shuffle: bool = shuffle
        self.batch_idx: int = 0
        self.n_samples: int = len(dataset)

        # Setting the training/validation set samplers. Both samplers are `None` if validation split
        # is set to 0.
        samplers: Optional[
            Tuple[
                Optional[Union[DistributedSampler, SubsetRandomSampler]],
                Optional[Union[DistributedSampler, SubsetRandomSampler]],
            ]
        ] = self._split_sampler(self.validation_split, test_set)

        # Guard against _split_sampler throwing an error.
        self.sampler: Optional[Union[DistributedSampler, SubsetRandomSampler]] = samplers[
            0
        ] if samplers else None
        self.valid_sampler: Optional[Union[DistributedSampler, SubsetRandomSampler]] = samplers[
            1
        ] if samplers else None

        # Keyword arguments for the torch DataLoader.
        self.init_kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }

        # Make mypy happy, guard against the sampler being `None`.
        if self.sampler:
            self.init_kwargs["sampler"] = self.sampler

        super().__init__(**self.init_kwargs)

    def _split_sampler(
        self, split: Union[float, int], test_set: bool = False
    ) -> Union[
        NoReturn,
        Tuple[
            Optional[Union[DistributedSampler, SubsetRandomSampler]],
            Optional[Union[DistributedSampler, SubsetRandomSampler]],
        ],
    ]:
        """
        Randomly splits data indices into training and validation sets and create the
        corresponding sampler objects. Throws error if the validation set size is configured to be
        larger than the entire dataset.

        Parameters
        ----------
        split : float or int
            The split proportion or the exact number of samples of the validation set.

        Returns
        -------
        tuple
            The training and validation sampler objects, in that order. A tuple of `None` if split
            value is 0.

        Raises
        ------
        AssertionError
            If validation set size is larger than entire dataset.
        """
        if split == 0.0:
            return None, None

        # Array of all sample indices.
        idx_full: np.ndarray = np.arange(self.n_samples)
        np.random.seed(0)  # fixed for reproducibility
        np.random.shuffle(idx_full)

        # If split value is int, assert it is not greater than the entire dataset.
        if isinstance(split, int):
            assert split > 0
            assert (
                split < self.n_samples
            ), "Validation set size is configured to be larger than entire dataset."
            len_valid: int = split  # number of validation samples

        else:
            len_valid = int(self.n_samples * split)

        # Sample indices for the training and validation sets.
        valid_idx: np.ndarray = idx_full[0:len_valid]
        train_idx: np.ndarray = np.delete(idx_full, np.arange(0, len_valid))

        # Training and validation set sampler objects.
        train_sampler: Union[DistributedSampler, SubsetRandomSampler] = SubsetRandomSampler(
            train_idx
        ) if not test_set else DistributedSampler(train_idx)
        valid_sampler: Union[DistributedSampler, SubsetRandomSampler] = SubsetRandomSampler(
            valid_idx
        ) if not test_set else DistributedSampler(train_idx)

        # Turn off shuffle option which is mutually exclusive with sampler option.
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self) -> Optional[DataLoader]:
        """
        Loads the validation data if the data is split.

        Returns
        -------
        torch.utils.data.DataLoader or None
            A torch `DataLoader` object with the validation `SubsetRandomSampler` object and
            keyword arguments if the validation sampler is present.
        """
        if self.valid_sampler is None:
            return None
        else:
            valid_init_kwargs: Dict[str, Any] = self.init_kwargs
            valid_init_kwargs["sampler"] = self.valid_sampler
            return DataLoader(**valid_init_kwargs)
