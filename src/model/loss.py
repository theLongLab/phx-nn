# src/model/loss.py

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

from pandas import DataFrame
from sklearn.base import BaseEstimator
import torch
from xgboost import XGBRegressor

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


def phx_param_map(output: Tuple[torch.Tensor, ...], gbtree: BaseEstimator) -> torch.Tensor:
    """
    PoolHapX parameter map serving as the PHX-NN cost function.

    Parameters
    ----------
    output : torch.Tensor
        The torch output tensor from the neural network.
    gbtree : sklearn.base.BaseEstimator
        The pre-trained xgboost regressor model (using scikit-learn API).

    Returns
    -------
    torch.Tensor
        A torch tensor containing the **negative** xgboost regressor output. To be **minimized**.
    """
    output_data = torch.cat(output, 1).clone().detach().cpu().numpy()
    gbtree = deepcopy(gbtree)
    df: DataFrame = DataFrame(data=output_data, columns=FEATURE_NAMES)
    result: torch.Tensor = -torch.mean(
        torch.from_numpy(gbtree.predict(df)).float().requires_grad_(True)
    )

    del gbtree
    return result
