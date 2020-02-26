# test.py

from argparse import Namespace
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
import torch
import torch.nn as nn
from xgboost import XGBRegressor

from src.model import PoolHapXNet
from src.utils import early_stop_callback, test_checkpoint_callback

TEST_CHECKPOINT_DIR: Path = Path(Path(sys.argv[2]).absolute().parent, "testing")
TEST_LOGS_DIR: Path = Path(TEST_CHECKPOINT_DIR.parent, "test_logs")
TEST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main(
    config: Mapping[str, Any], model_hparams: Mapping[str, Any], warm_start_fpath: Optional[Path]
) -> None:
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
        model_hparams=model_hparams,
        train_data=X_train,
        val_data=X_test,
        loss_fn=loss_fn,
        **config["test_data_loader"]
    )

    if warm_start_fpath:
        warm_start_checkpoint = torch.load(
            warm_start_fpath, map_location=lambda storage, loc: storage
        )

        try:
            model.load_state_dict(warm_start_checkpoint["state_dict"])
        except RuntimeError:
            new_state_dict = {}
            for k, v in warm_start_checkpoint["state_dict"].items():
                new_state_dict[k[7:]] = v

            model.load_state_dict(new_state_dict)

    trainer: pl.Trainer = pl.Trainer(
        default_save_path=TEST_LOGS_DIR,
        checkpoint_callback=test_checkpoint_callback(TEST_CHECKPOINT_DIR),
        early_stop_callback=early_stop_callback(),
        gpus=[3],  # use 4th
        min_nb_epochs=config["min_epochs"],
        max_nb_epochs=config["epochs"],
        # distributed_backend="ddp",
        use_amp=True,
        # show_progress_bar=False,
    )

    trainer.fit(model)

    best_checkpoint = torch.load(
        next(TEST_CHECKPOINT_DIR.iterdir()), map_location=lambda storage, loc: storage
    )
    model.load_state_dict(best_checkpoint["state_dict"])
    trainer.test(model)

    # Save
    torch.save(model.state_dict(), Path(TEST_CHECKPOINT_DIR.parent, "phxnn.pth"))


if __name__ == "__main__":
    with Path(sys.argv[1]).open() as config_f:
        config: Mapping[str, Any] = json.load(config_f)

    with Path(sys.argv[2]).open() as arch_f:
        model_hparams: Mapping[str, Any] = json.load(arch_f)

    warm_start_fpath: Optional[Path]
    try:
        warm_start_fpath = Path(sys.argv[3])
    except IndexError:
        warm_start_fpath = None

    main(config=config, model_hparams=model_hparams, warm_start_fpath=warm_start_fpath)
