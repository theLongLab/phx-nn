# src/model/model.py

from collections import OrderedDict
from pathlib import Path
import pickle
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    NoReturn,
    Sequence,
    Tuple,
    Union,
)

from adabound import AdaBound
from optuna import Trial
import pytorch_lightning as pl
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from src.base import BaseDataLoader, BaseModel
from src.data_loader import SumStatDataLoader
from src.model.loss import phx_param_map


GPU_COUNT: int = torch.cuda.device_count()
HEADS: Tuple[str, ...] = (
    "num_gap_window",
    "inpool_gap_supp_min",
    "allpool_gap_supp_min",
    "l1_region_size_min",
    "l1_region_size_max",
    "l2_region_size_min",
    "l2_region_size_max",
    "l34_region_mis_tol",
    "l56_region_mis_tol",
    "l78_region_mis_tol",
    "est_indv_perpool",
    "aem_max_l",
    "bfs_mis_tol",
    "aem_conv_cutoff",
    "aem_zero_cutoff",
    "aem_regional_crosspool_freq_cutoff",
    "aem_regional_hapsetsize_min",
    "aem_regional_hapsetsize_max",
    "regr_one_vec_weight",
    "regr_hap_vc_weight",
    "regr_hap_11_weight",
    "regr_regional_hapsetsize_max",
    "regr_gamma_min",
    "regr_n_gamma",
    "regr_mis_tol",
    "regr_coverage_weight",
    "regr_distance_max_weight",
    "regr_max_regions",
)
INPUT_SIZE: int = 40


class HParamNet(BaseModel):
    def __init__(self, trial: Trial, hparam_space: Mapping[str, Sequence]) -> None:
        super().__init__()
        self.layers: List[nn.Module] = []
        self.batchnorms: List[nn.Module] = []
        self.dropouts: List[nn.Module] = []

        # Order of PHX params for inner model input.
        self.heads: Tuple[str, ...] = HEADS
        final_output_dim: int = self.__optimize_layers(trial=trial, hparam_space=hparam_space)
        self._batch_size: int = trial.suggest_categorical("batch_size", hparam_space["batch_size"])
        self._lr: float = trial.suggest_uniform("lr", hparam_space["lr"][0], hparam_space["lr"][1])
        self._final_lr: float = trial.suggest_uniform(
            "final_lr", hparam_space["final_lr"][0], hparam_space["final_lr"][1]
        )
        self.__build_model(final_output_dim)

    def __optimize_layers(self, trial: Trial, hparam_space: Mapping[str, Sequence]) -> int:
        # Optimize the number of layers, hidden units in each layer and dropout rate.
        n_layers: int = trial.suggest_int(
            "n_layers", hparam_space["n_layers"][0], hparam_space["n_layers"][1]
        )
        input_dropout_rate: float = trial.suggest_loguniform(
            "input_dropout_rate", hparam_space["dropout_rate"][0], 0.2
        )
        dropout_rate: float = trial.suggest_loguniform(
            "dropout_rate", input_dropout_rate, hparam_space["dropout_rate"][1]
        )

        input_dim: int = INPUT_SIZE  # number of input summary statistics
        i: int
        for i in range(n_layers):
            output_dim: int = trial.suggest_int(
                "n_units_l{}".format(i), hparam_space["n_units_l"][0], hparam_space["n_units_l"][1],
            )
            self.layers.append(nn.Linear(input_dim, output_dim))

            if i != n_layers - 1:
                self.batchnorms.append(nn.BatchNorm1d(num_features=output_dim))
                self.dropouts.append(nn.Dropout(dropout_rate))

            # Input layer needs a lower dropout rate.
            if i == 0:
                self.dropouts[-1] = nn.Dropout(input_dropout_rate)

            input_dim = output_dim

        return output_dim

    def __build_model(self, final_output_dim: int) -> None:
        # Assign layers as class variables (PyTorch requirement).
        layer: nn.Module
        for i, layer in enumerate(self.layers):
            layer_name: str = "fc{}".format(i)
            setattr(self, layer_name, layer)
            nn.init.xavier_normal_(getattr(self, layer_name).weight)  # init weight

        # Assign batchnorm actions as class variables (Pytorch requirement)
        batchnorm: nn.Module
        for i, batchnorm in enumerate(self.batchnorms):
            setattr(self, "bn{}".format(i), batchnorm)

        # Assign dropout actions as class variables (PyTorch requirement).
        dropout: nn.Module
        for i, dropout in enumerate(self.dropouts):
            setattr(self, "dropout{}".format(i), dropout)

        # Assign heads as class variables (PyTorch requirement).
        head: str
        for i, head in enumerate(self.heads):
            setattr(self, head, nn.Linear(final_output_dim, 1))
            nn.init.xavier_normal_(getattr(self, head).weight)

    def forward(self, X):
        x: torch.Tensor = X
        for layer, batchnorm, dropout in zip(self.layers[:-1], self.batchnorms, self.dropouts):
            x = F.leaky_relu(layer(x))
            x = batchnorm(x)
            x = dropout(x)

        for layer in self.layers[-1:]:
            x = F.leaky_relu(layer(x))

        # Clamp values of PHX parameters at each head.
        num_gap_window: torch.Tensor = torch.clamp(torch.round(self.num_gap_window(x)).int(), 1, 5)
        inpool_gap_supp_min: torch.Tensor = torch.sigmoid(self.inpool_gap_supp_min(x))
        allpool_gap_supp_min: torch.Tensor = torch.sigmoid(self.allpool_gap_supp_min(x))

        l1_region_size_min: torch.Tensor = torch.clamp(
            torch.round(self.l1_region_size_min(x)).int(), 3, 13
        )
        l1_region_size_max: torch.Tensor = torch.clamp(
            torch.round(self.l1_region_size_max(x)).int(), max(l1_region_size_min).item() + 1, 16
        )
        l2_region_size_min: torch.Tensor = l1_region_size_min
        l2_region_size_max: torch.Tensor = l1_region_size_max
        l34_region_mis_tol: torch.Tensor = torch.clamp(
            torch.round(self.l34_region_mis_tol(x)).int(), 0, 5
        )
        l56_region_mis_tol: torch.Tensor = torch.clamp(
            torch.round(self.l56_region_mis_tol(x)).int(), 1, 6
        )
        l78_region_mis_tol: torch.Tensor = torch.clamp(
            torch.round(self.l78_region_mis_tol(x)).int(), 2, 7
        )

        est_indv_perpool: torch.Tensor = torch.clamp(
            torch.round(self.est_indv_perpool(x)).int(), 1000, 1000000
        )

        aem_max_l: torch.Tensor = torch.clamp(torch.round(self.aem_max_l(x)).int(), 0, 6)
        tens: torch.Tensor
        for tens in aem_max_l:
            tens += 1 if tens.item() % 2 == 0 else 0

        bfs_mis_tol: torch.Tensor = torch.clamp(torch.round(self.bfs_mis_tol(x)).int(), 0, 10)

        aem_conv_cutoff: torch.Tensor = torch.clamp(self.aem_conv_cutoff(x), 0, 1e-4)
        aem_zero_cutoff: torch.Tensor = torch.clamp(self.aem_zero_cutoff(x), 0, 1e-6)
        aem_regional_crosspool_freq_cutoff: torch.Tensor = torch.clamp(
            self.aem_regional_crosspool_freq_cutoff(x), 0, 0.05
        )
        aem_regional_hapsetsize_min: torch.Tensor = torch.clamp(
            torch.round(self.aem_regional_hapsetsize_min(x)).int(), 1, 10
        )
        aem_regional_hapsetsize_max: torch.Tensor = torch.clamp(
            torch.round(self.aem_regional_hapsetsize_max(x)).int(),
            max(11, max(aem_regional_hapsetsize_min).item() + 4),
            100,
        )

        regr_one_vec_weight: torch.Tensor = torch.clamp(self.regr_one_vec_weight(x), 1, 10)
        regr_hap_vc_weight: torch.Tensor = torch.clamp(self.regr_hap_vc_weight(x), 1, 10)
        regr_hap_11_weight: torch.Tensor = torch.clamp(self.regr_hap_11_weight(x), 1, 10)
        regr_regional_hapsetsize_max: torch.Tensor = torch.clamp(
            torch.round(self.regr_regional_hapsetsize_max(x)).int(), 11, 100
        )
        regr_gamma_min: torch.Tensor = torch.sigmoid(self.regr_gamma_min(x)) / 4
        regr_n_gamma: torch.Tensor = torch.clamp(torch.round(self.regr_n_gamma(x)).int(), 2, 10)
        regr_mis_tol: torch.Tensor = torch.clamp(torch.round(self.regr_mis_tol(x)).int(), 8, 20)
        regr_coverage_weight: torch.Tensor = torch.clamp(self.regr_coverage_weight(x), 0.5, 2.5)
        regr_distance_max_weight: torch.Tensor = torch.clamp(self.regr_distance_max_weight(x), 1, 5)
        regr_max_regions: torch.Tensor = torch.clamp(
            torch.round(self.regr_max_regions(x)).int(), 2, 3
        )

        output: List[torch.Tensor] = []
        for head in self.heads:
            output.append(eval(head).float())

        return tuple(output)


class LightningHParamNet(pl.LightningModule):
    def __init__(
        self,
        est: nn.Module,
        cvtrain_data: torch.Tensor,
        cvval_data: torch.Tensor,
        loss_fn: BaseEstimator,
        shuffle: bool = False,
        validation_split: Union[float, int] = 0.0,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        # Avoid overriding `LightningModule` attributes (e.g. self.model)
        self._model: nn.Module = est
        self.loss_fn: XGBRegressor = loss_fn
        self._dataloader_args: Dict[str, Any] = {
            "data": cvtrain_data,
            "batch_size": self._model._batch_size,
            "shuffle": shuffle,
            "validation_split": validation_split,
            "num_workers": num_workers,
        }

        self._cvval_data: torch.Tensor = cvval_data

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self._model(X)

    def training_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        X: torch.Tensor
        y: torch.Tensor
        X, y = batch

        output: Tuple[torch.Tensor, ...] = self.forward(X)
        loss: torch.Tensor = phx_param_map(output=output, gbtree=self.loss_fn)
        return {"loss": loss}

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        X: torch.Tensor
        y: torch.Tensor
        X, y = batch

        output: Tuple[torch.Tensor, ...] = self.forward(X)
        val_loss: torch.Tensor = phx_param_map(output=output, gbtree=self.loss_fn)
        return {"val_loss": val_loss}

    def validation_end(self, outputs: Sequence[Mapping]) -> Dict[str, Dict[str, float]]:
        x: Dict[str, torch.Tensor]
        mean_val_loss: Union[Any, torch.Tensor] = sum(  # Union Any to make mypy happy
            x["val_loss"].clone().detach() for x in outputs
        ) / len(outputs)

        return {"log": {"mean_val_loss": mean_val_loss.item()}}

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        X: torch.Tensor
        y: torch.Tensor
        X, y = batch

        output: Tuple[torch.Tensor, ...] = self.forward(X)
        test_loss: torch.Tensor = phx_param_map(output=output, gbtree=self.loss_fn)
        return {"test_loss": test_loss}

    def test_end(self, outputs: Sequence[Mapping]) -> Dict[str, Dict[str, float]]:
        x: Dict[str, torch.Tensor]
        mean_test_loss: Union[Any, torch.Tensor] = sum(  # Union Any to make mypy mappy
            x["test_loss"].clone().detach() for x in outputs
        ) / len(outputs)

        return {"log": {"mean_test_loss": mean_test_loss.item()}}

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[StepLR]]:
        self.opt: Optimizer = AdaBound(
            self._model.parameters(), lr=self._model._lr, final_lr=self._model._final_lr
        )
        self.lr_sch: StepLR = StepLR(self.opt, 50)
        return [self.opt], [self.lr_sch]

    def on_epoch_end(self) -> None:
        self.lr_sch.step()

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        self._train_dataloader: BaseDataLoader = SumStatDataLoader(**self._dataloader_args)
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self) -> Optional[DataLoader]:
        return self._train_dataloader.split_validation()

    @pl.data_loader
    def test_dataloader(self) -> DataLoader:
        return SumStatDataLoader(
            data=self._cvval_data,
            batch_size=512,
            shuffle=False,
            validation_split=0.0,
            num_workers=4 * GPU_COUNT,
        )


class PoolHapXNet(pl.LightningModule):
    def __init__(
        self,
        model_hparams: Mapping[str, Any],
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        loss_fn: BaseEstimator,
        shuffle: bool = False,
        validation_split: Union[float, int] = 0.0,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.layers: List[nn.Module] = []
        self.batchnorms: List[nn.Module] = []
        self.dropouts: List[nn.Module] = []
        self.heads: Tuple[str, ...] = HEADS
        final_output_dim: int = self.__set_layers(model_hparams)
        self.__build_model(final_output_dim)

        self.loss_fn: XGBRegressor = loss_fn
        self._dataloader_args: Dict[str, Any] = {
            "data": train_data,
            "batch_size": model_hparams["batch_size"],
            "shuffle": shuffle,
            "validation_split": validation_split,
            "num_workers": num_workers,
            # "test_set": True,
        }

        self._lr: float = model_hparams["lr"]
        self._final_lr: float = model_hparams["final_lr"]
        self._val_data: torch.Tensor = val_data

    def __set_layers(self, model_hparams: Mapping[str, Any]) -> int:
        # Optimize the number of layers, hidden units in each layer and dropout rate.
        n_layers: int = model_hparams["n_layers"]
        input_dropout_rate: float = model_hparams["input_dropout_rate"]
        dropout_rate: float = model_hparams["dropout_rate"]

        input_dim: int = INPUT_SIZE  # number of input summary statistics
        i: int
        for i in range(n_layers):
            output_dim: int = model_hparams["n_units_l{}".format(i)]
            self.layers.append(nn.Linear(input_dim, output_dim))

            if i != n_layers - 1:
                self.batchnorms.append(nn.BatchNorm1d(num_features=output_dim))
                self.dropouts.append(nn.Dropout(dropout_rate))

            # Input layer needs a lower dropout rate.
            if i == 0:
                self.dropouts[-1] = nn.Dropout(input_dropout_rate)

            input_dim = output_dim

        return output_dim

    def __build_model(self, final_output_dim: int) -> None:
        # Assign layers as class variables (PyTorch requirement).
        layer: nn.Module
        for i, layer in enumerate(self.layers):
            layer_name: str = "fc{}".format(i)
            setattr(self, layer_name, layer)
            nn.init.xavier_normal_(getattr(self, layer_name).weight)  # init weight

        # Assign batchnorm actions as class variables (Pytorch requirement)
        batchnorm: nn.Module
        for i, batchnorm in enumerate(self.batchnorms):
            setattr(self, "bn{}".format(i), batchnorm)

        # Assign dropout actions as class variables (PyTorch requirement).
        dropout: nn.Module
        for i, dropout in enumerate(self.dropouts):
            setattr(self, "dropout{}".format(i), dropout)

        # Assign heads as class variables (PyTorch requirement).
        head: str
        for i, head in enumerate(self.heads):
            setattr(self, head, nn.Linear(final_output_dim, 1))
            nn.init.xavier_normal_(getattr(self, head).weight)

    def forward(self, X):
        x: torch.Tensor = X
        for layer, batchnorm, dropout in zip(self.layers[:-1], self.batchnorms, self.dropouts):
            x = F.leaky_relu(layer(x))
            x = batchnorm(x)
            x = dropout(x)

        for layer in self.layers[-1:]:
            x = F.leaky_relu(layer(x))

        # Clamp values of PHX parameters at each head.
        num_gap_window: torch.Tensor = torch.clamp(torch.round(self.num_gap_window(x)).int(), 1, 5)
        inpool_gap_supp_min: torch.Tensor = torch.sigmoid(self.inpool_gap_supp_min(x))
        allpool_gap_supp_min: torch.Tensor = torch.sigmoid(self.allpool_gap_supp_min(x))

        l1_region_size_min: torch.Tensor = torch.clamp(
            torch.round(self.l1_region_size_min(x)).int(), 3, 13
        )
        l1_region_size_max: torch.Tensor = torch.clamp(
            torch.round(self.l1_region_size_max(x)).int(), max(l1_region_size_min).item() + 1, 16
        )
        l2_region_size_min: torch.Tensor = l1_region_size_min
        l2_region_size_max: torch.Tensor = l1_region_size_max
        l34_region_mis_tol: torch.Tensor = torch.clamp(
            torch.round(self.l34_region_mis_tol(x)).int(), 0, 5
        )
        l56_region_mis_tol: torch.Tensor = torch.clamp(
            torch.round(self.l56_region_mis_tol(x)).int(), 1, 6
        )
        l78_region_mis_tol: torch.Tensor = torch.clamp(
            torch.round(self.l78_region_mis_tol(x)).int(), 2, 7
        )

        est_indv_perpool: torch.Tensor = torch.clamp(
            torch.round(self.est_indv_perpool(x)).int(), 1000, 1000000
        )

        aem_max_l: torch.Tensor = torch.clamp(torch.round(self.aem_max_l(x)).int(), 0, 6)
        tens: torch.Tensor
        for tens in aem_max_l:
            tens += 1 if tens.item() % 2 == 0 else 0

        bfs_mis_tol: torch.Tensor = torch.clamp(torch.round(self.bfs_mis_tol(x)).int(), 0, 10)

        aem_conv_cutoff: torch.Tensor = torch.clamp(self.aem_conv_cutoff(x), 0, 1e-4)
        aem_zero_cutoff: torch.Tensor = torch.clamp(self.aem_zero_cutoff(x), 0, 1e-6)
        aem_regional_crosspool_freq_cutoff: torch.Tensor = torch.clamp(
            self.aem_regional_crosspool_freq_cutoff(x), 0, 0.05
        )
        aem_regional_hapsetsize_min: torch.Tensor = torch.clamp(
            torch.round(self.aem_regional_hapsetsize_min(x)).int(), 1, 10
        )
        aem_regional_hapsetsize_max: torch.Tensor = torch.clamp(
            torch.round(self.aem_regional_hapsetsize_max(x)).int(),
            max(11, max(aem_regional_hapsetsize_min).item() + 4),
            100,
        )

        regr_one_vec_weight: torch.Tensor = torch.clamp(self.regr_one_vec_weight(x), 1, 10)
        regr_hap_vc_weight: torch.Tensor = torch.clamp(self.regr_hap_vc_weight(x), 1, 10)
        regr_hap_11_weight: torch.Tensor = torch.clamp(self.regr_hap_11_weight(x), 1, 10)
        regr_regional_hapsetsize_max: torch.Tensor = torch.clamp(
            torch.round(self.regr_regional_hapsetsize_max(x)).int(), 11, 100
        )
        regr_gamma_min: torch.Tensor = torch.sigmoid(self.regr_gamma_min(x)) / 4
        regr_n_gamma: torch.Tensor = torch.clamp(torch.round(self.regr_n_gamma(x)).int(), 2, 10)
        regr_mis_tol: torch.Tensor = torch.clamp(torch.round(self.regr_mis_tol(x)).int(), 8, 20)
        regr_coverage_weight: torch.Tensor = torch.clamp(self.regr_coverage_weight(x), 0.5, 2.5)
        regr_distance_max_weight: torch.Tensor = torch.clamp(self.regr_distance_max_weight(x), 1, 5)
        regr_max_regions: torch.Tensor = torch.clamp(
            torch.round(self.regr_max_regions(x)).int(), 2, 3
        )

        output: List[torch.Tensor] = []
        for head in self.heads:
            output.append(eval(head).float())

        return output

    def training_step(
        self, batch: Tuple, batch_idx: int
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        X: torch.Tensor
        y: torch.Tensor
        X, y = batch

        output: Tuple[torch.Tensor, ...] = self.forward(X)
        loss: torch.Tensor = phx_param_map(output=output, gbtree=self.loss_fn)

        tqdm_dict: Dict[str, torch.Tensor] = {"train_loss": loss}
        log_output: OrderedDict = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return log_output

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        X: torch.Tensor
        y: torch.Tensor
        X, y = batch

        output: Tuple[torch.Tensor, ...] = self.forward(X)
        val_loss: torch.Tensor = phx_param_map(output=output, gbtree=self.loss_fn)
        return {"val_loss": val_loss}

    def validation_end(self, outputs: Sequence[Mapping]) -> Dict[str, Dict[str, float]]:
        x: Dict[str, torch.Tensor]
        mean_val_loss: Union[Any, torch.Tensor] = sum(  # Union Any to make mypy happy
            x["val_loss"].clone().detach() for x in outputs
        ) / len(outputs)

        tqdm_dict: Dict[str, float] = {"mean_val_loss": mean_val_loss.item()}
        return {"progress_bar": tqdm_dict, "log": {"mean_val_loss": mean_val_loss.item()}}

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        X: torch.Tensor
        y: torch.Tensor
        X, y = batch

        output: Tuple[torch.Tensor, ...] = self.forward(X)
        test_loss: torch.Tensor = phx_param_map(output=output, gbtree=self.loss_fn)
        return {"test_loss": test_loss}

    def test_end(self, outputs: Sequence[Mapping]) -> Dict[str, Dict[str, float]]:
        x: Dict[str, torch.Tensor]
        mean_test_loss: Union[Any, torch.Tensor] = sum(  # Union Any to make mypy happy
            x["test_loss"].clone().detach() for x in outputs
        ) / len(outputs)
        tqdm_dict: Dict[str, float] = {"mean_test_loss": mean_test_loss.item()}

        print("\n\n==\nMean Test Loss: {}\n==\n\n".format(mean_test_loss.item()))
        return {"progress_bar": tqdm_dict, "log": {"mean_test_loss": mean_test_loss.item()}}

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[StepLR]]:
        self.opt: Optimizer = AdaBound(self.parameters(), lr=self._lr, final_lr=self._final_lr)
        self.lr_sch: StepLR = StepLR(self.opt, 50)
        return [self.opt], [self.lr_sch]

    def on_epoch_end(self) -> None:
        self.lr_sch.step()

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        self._train_dataloader: BaseDataLoader = SumStatDataLoader(**self._dataloader_args)
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self) -> Optional[DataLoader]:
        return self._train_dataloader.split_validation()

    @pl.data_loader
    def test_dataloader(self) -> DataLoader:
        return SumStatDataLoader(
            data=self._val_data,
            batch_size=512,
            shuffle=False,
            validation_split=0.0,
            num_workers=4 * GPU_COUNT,
            # test_set=True,
        )
