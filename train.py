# train.py

from datetime import datetime
from itertools import cycle
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Generator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from optuna import create_study, Study, Trial
import pandas as pd
import pytorch_lightning as pl
import ray
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from xgboost import XGBRegressor

from src.logging import DictLogger
from src.model import HParamNet, LightningHParamNet
from src.utils import early_stop_callback, train_checkpoint_callback


# Boilerplate to set up checkpoint directory structure and initialize ray.
CHECKPOINT_DIR_NAME: str = datetime.now().strftime("%Y%m%d_%H%M")
TRAIN_CHECKPOINT_DIR: Path = Path(Path.cwd(), "saved", CHECKPOINT_DIR_NAME, "training")
TRAIN_LOGS_DIR: Path = Path(TRAIN_CHECKPOINT_DIR, "train_logs")
TRAIN_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LOGS_DIR.mkdir(parents=True, exist_ok=True)

ray.init(num_cpus=32, num_gpus=4)


@ray.remote(num_cpus=4, num_gpus=0.5)  # fractional GPU to fit 2 CV rounds on a single GPU
def _proc(
    est: nn.Module,
    cvtrain_data: torch.Tensor,
    cvval_data: torch.Tensor,
    loss_fn: BaseEstimator,
    data_loader_args: Mapping[str, Any],
    logger: pl.logging.LightningLoggerBase,
    checkpoint_callback: pl.callbacks.ModelCheckpoint,
    epochs: int,
) -> float:
    """
    A single k-fold cross validation fold training loop.

    Parameters
    ----------
    est : torch.nn.Module
        Neural network model.
    cvtrain_data : torch.Tensor
        Cross validation training set.
    cvval_data : torch.Tensor
        Cross validation validation set (testing set in the neural network).
    loss_fn : sklearn.base.BaseEstimator
        GBDT custom loss function.
    data_loader_args : dict-like
        Keyword arguments to pass to Dataloader.
    logger : pl.logging.LightningLoggerBase
        Logger object.
    checkpoint_callback pl.callbacks.ModelCheckpoint
        Checkpoint callback object.
    epochs : int
        Number of epochs.

    Returns
    -------
    float
        The score for a single k-fold cross validation fold.
    """

    # NN model object.
    model: pl.LightningModule = LightningHParamNet(
        est=est,
        cvtrain_data=cvtrain_data,
        cvval_data=cvval_data,
        loss_fn=loss_fn,
        **data_loader_args
    )

    # PyTorch Lightning trainer instance.
    trainer: pl.Trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback(),
        # early_stop_callback=None,
        gpus=1,
        min_nb_epochs=50,
        max_nb_epochs=epochs,
        # distributed_backend="ddp",
        use_amp=True,
    )
    trainer.fit(model)

    checkpoint = torch.load(
        next(Path(TRAIN_CHECKPOINT_DIR, logger.version).iterdir()),
        map_location=lambda storage, loc: storage,
    )
    model.load_state_dict(checkpoint["state_dict"])
    trainer.test(model)
    logger.save()

    return logger.metrics[-1]["mean_test_loss"]


def objective(
    trial: Trial,
    cv_rounds: Sequence[Tuple],
    data: torch.Tensor,
    loss_fn: BaseEstimator,
    data_loader_args: Mapping[str, Any],
    hparam_space: Mapping[str, Sequence],
    epochs: int,
) -> float:
    """
    Optuna trial objective function.

    Parameters
    ----------
    trial : optuna.Trial
        Current trial object.
    cv_rounds : list-like of tuples
        Train/validation data indices of k-fold cross validation folds.
    data : torch.Tensor
        Training data.
    loss_fn : sklearn.base.BaseEstimator
        Trained GBDT model object.
    data_loader_args : dict-like
        Keyword arguments to be passed into dataloader object.
    hparam_space : dict-like
        Hyperparameter search space for optuna trial suggestions.
    epochs : int
        Number of epochs per neural network training loop.

    Returns
    -------
    float
        Mean cross validation fold score.
    """

    # Number of folds
    folds: int = len(cv_rounds)

    # Manual cross-validation with the same hparam search estimator.
    est: nn.Module = HParamNet(trial=trial, hparam_space=hparam_space)
    cv_results: List[float] = ray.get(
        [
            _proc.remote(
                est=est,
                cvtrain_data=data[cv_rounds[i][0]],
                cvval_data=data[cv_rounds[i][1]],
                loss_fn=loss_fn,
                data_loader_args=data_loader_args,
                logger=DictLogger("trial_{}_{}".format(trial.number, i), TRAIN_LOGS_DIR),
                checkpoint_callback=train_checkpoint_callback(
                    Path(TRAIN_CHECKPOINT_DIR, "trial_{}_{}".format(trial.number, i))
                ),
                epochs=epochs,
            )
            for i in range(folds)
        ]
    )

    return np.mean(cv_results)


def main(config: Mapping[str, Any], seed: Optional[int]) -> None:
    """
    Hyperparameter tuning training loop.

    Parameters
    ----------
    config : dict-like
        Configuration properties in a dictionary.
    seed : int or None
        Seed to specify; random if not specified.
    """

    # Load inner GBDT model pickle file as the loss function.
    loss_fn: XGBRegressor
    with open(config["loss"], "rb") as inner_model:
        loss_fn = pickle.load(inner_model)

    # Load training data into a float32 torch tensor.
    X: torch.Tensor = torch.from_numpy(
        pd.read_csv(config["training_data"]).drop(["sim"], axis=1).values
    ).float()

    # Set k-fold cross validation indices.
    kf: KFold = KFold(n_splits=config["folds"], shuffle=True, random_state=seed)
    cv_rounds: List[Tuple[np.ndarray, np.ndarray]] = []

    train_idx: np.ndarray
    val_idx: np.ndarray
    for train_idx, val_idx in kf.split(X):
        cv_rounds.append((train_idx, val_idx))

    # Optuna study loop.
    study: Study = create_study()
    trial: Trial
    study.optimize(
        lambda trial: objective(  # lambda to pass parameters into the objective fn
            trial=trial,
            cv_rounds=cv_rounds,
            data=X,
            loss_fn=loss_fn,
            data_loader_args=config["data_loader"],
            hparam_space=config["hparam_space"],
            epochs=config["epochs"],
        ),
        n_trials=config["trials"],
    )

    # Save study, best trial, and best NN architecture hparams to disk.
    study_fpath: Path = Path(TRAIN_CHECKPOINT_DIR, "optuna_study.pkl")
    best_trial_fpath: Path = Path(TRAIN_CHECKPOINT_DIR, "best_trial.pkl")
    best_arch_fpath: Path = Path(TRAIN_CHECKPOINT_DIR, "best_arch.json")

    with best_trial_fpath.open("wb") as best_trial_pkl:
        pickle.dump(study.best_trial, best_trial_pkl)

    with study_fpath.open("wb") as study_pkl:
        pickle.dump(study, study_pkl)

    with best_arch_fpath.open("w") as best_arch:
        json.dump(study.best_trial.params, best_arch)

    print("Best Trial:")
    print("Value: {}".format(study.best_trial.value))
    print("Params: ")
    for k, v in study.best_trial.params.items():
        print("    {}: {}".format(k, v))


if __name__ == "__main__":
    with Path(sys.argv[1]).open() as config_f:
        config: Mapping[str, Any] = json.load(config_f)

    seed: Optional[int] = None
    try:
        seed = int(sys.argv[2])
    except IndexError:
        pass

    main(config=config, seed=seed)
