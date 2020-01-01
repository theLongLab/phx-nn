# test.py

import json
from pathlib import Path
import pickle
import sys
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from optuna import create_study, Study, Trial
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import ray
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
from xgboost import XGBRegressor

from src.logging import DictLogger
from src.model import PoolHapXNet


def main(config: Mapping[str, Any]) -> None:
    loss_fn: XGBRegressor
    with open(config["loss"], "rb") as inner_model:
        loss_fn = pickle.load(inner_model)

    X_train: torch.Tensor = torch.from_numpy(
        pd.read_csv(config["training_data"]).drop(["sim"], axis=1).values
    )

    X_test: torch.Tensor = torch.from_numpy(
        pd.read_csv(config["testing_data"]).drop(["sim"], axis=1).values
    )

    model: pl.LightningModule = PoolHapXNet(
        model_hparams={},
        train_data=X_train,
        val_data=X_test,
        loss_fn=loss_fn,
        **config["test_data_loader"]
    )

    checkpoint_callback: ModelCheckpoint = ModelCheckpoint()

    trainer: pl.Trainer = pl.Trainer(checkpoint_callback=checkpoint_callback)

    trainer.fit(model)
    trainer.test()

    # Save
    trained_phx_nn_fpath: Path = Path()
    with trained_phx_nn_fpath.open("wb") as phx_nn_pkl:
        pickle.dump(model, phx_nn_pkl)

if __name__ == "__main__":
    pass

    # main()
