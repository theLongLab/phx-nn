# base/base_model.py

from abc import abstractmethod
from typing import Iterable, NoReturn, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models.

    Methods
    -------
    forward()
        Forward pass on neural network. Throws an error if not implemented in child class.
    """

    # mypy error stems from the PyTorch source code side.
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic abstract method.

        Parameters
        ----------
        *inputs : torch.Tensor
            Tensors representing the transformed input data.

        Returns
        -------
        torch.Tensor
            The model output tensor.

        Raises
        ------
        NotImplementedError
            If not implemented in child class.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Prints model summary and the number of trainable parameters.
        """
        p: torch.Tensor
        model_parameters: Iterable[torch.Tensor] = filter(
            lambda p: p.requires_grad, self.parameters()
        )
        params: int = sum([np.prod(p.size()) for p in model_parameters])

        return super().__str__() + "\nTrainable parameters: {}".format(params)
