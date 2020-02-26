# inference.py

import json
from pathlib import Path
import pickle
import sys
from typing import Any, List, Mapping, Optional

from numpy import ndarray
import pandas as pd
import pytorch_lightning as pl
import torch
from xgboost import XGBRegressor

from src.model import PoolHapXNet

FEATURE_NAMES: List[str] = [
    "Num_Gap_Window",
    "In-pool_Gap_Support_Min",
    "All-pool_Gap_Support_Min",
    "Level_1_Region_Size_Min",
    "Level_1_Region_Size_Max",
    "Level_2_Region_Size_Min",
    "Level_2_Region_Size_Max",
    "Level_3_4_Region_Mismatch_Tolerance",
    "Level_5_6_Region_Mismatch_Tolerance",
    "Level_7_8_Region_Mismatch_Tolerance",
    "Est_Ind_PerPool",
    "AEM_Maximum_Level",
    "BFS_Mismatch_Tolerance",
    "AEM_Convergence_Cutoff",
    "AEM_Zero_Cutoff",
    "AEM_Regional_Cross_Pool_Freq_Cutoff",
    "AEM_Regional_HapSetSize_Min",
    "AEM_Regional_HapSetSize_Max",
    "Regression_One_Vector_Weight",
    "Regression_Hap_VC_Weight",
    "Regression_Hap_11_Weight",
    "Regression_Regional_HapSetSize_Max",
    "Regression_Gamma_Min",
    "Regression_n_Gamma",
    "Regression_Mismatch_Tolerance",
    "Regression_Coverage_Weight",
    "Regression_Distance_Max_Weight",
    "Regression_Maximum_Regions",
]

INT_FEATS: List[str] = [
    "Num_Gap_Window",
    "Level_1_Region_Size_Min",
    "Level_1_Region_Size_Max",
    "Level_2_Region_Size_Min",
    "Level_2_Region_Size_Max",
    "Level_3_4_Region_Mismatch_Tolerance",
    "Level_5_6_Region_Mismatch_Tolerance",
    "Level_7_8_Region_Mismatch_Tolerance",
    "Est_Ind_PerPool",
    "AEM_Maximum_Level",
    "BFS_Mismatch_Tolerance",
    "AEM_Convergence_Cutoff",
    "AEM_Regional_HapSetSize_Min",
    "AEM_Regional_HapSetSize_Max",
    "Regression_Regional_HapSetSize_Max",
    "Regression_n_Gamma",
    "Regression_Mismatch_Tolerance",
    "Regression_Maximum_Regions",
]


def main(
    config: Mapping[str, Any],
    model_hparams: Mapping[str, Any],
    nn_state_dict_fpath: Path,
    data_fpath: Path,
    output_fpath: Path,
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

    best_state_dict = torch.load(nn_state_dict_fpath)
    model.load_state_dict(best_state_dict)

    data: pd.DataFrame = pd.read_csv(data_fpath)
    data_sims: pd.DataFrame = pd.DataFrame(data["sim"])
    data_sims.columns = ["Project_Name"]

    data_tens: torch.Tensor = torch.from_numpy(data.drop(["sim"], axis=1).values).float()

    model.eval()
    output_tens: ndarray = torch.cat(model(data_tens), 1).clone().detach().cpu().numpy()

    output: pd.DataFrame = pd.concat(
        [
            data_sims,
            pd.DataFrame(output_tens, columns=FEATURE_NAMES),
            pd.DataFrame(
                data=[0.9 for _ in range(len(data.index))], columns=["Regression_Gamma_Max"]
            ),
        ],
        axis=1,
    )

    int_feat: str
    for int_feat in INT_FEATS:
        output[int_feat] = output[int_feat].astype(int)

    output.to_csv(output_fpath, sep="\t", index=False)


if __name__ == "__main__":
    with Path(sys.argv[1]).open() as config_f:
        config: Mapping[str, Any] = json.load(config_f)

    with Path(sys.argv[2]).open() as arch_f:
        model_hparams: Mapping[str, Any] = json.load(arch_f)

    nn_state_dict_fpath: Path = Path(sys.argv[3])
    data_fpath: Path = Path(sys.argv[4])
    output_fpath: Path = Path(sys.argv[5])

    main(
        config=config,
        model_hparams=model_hparams,
        nn_state_dict_fpath=nn_state_dict_fpath,
        data_fpath=data_fpath,
        output_fpath=output_fpath,
    )
