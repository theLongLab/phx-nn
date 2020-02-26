# data_split.py

from pathlib import Path
import sys
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def main(test_size: float, seed: Optional[int]) -> None:
    """
    Splits the data based on test set proportion.
    """
    processed_dpath: Path = Path("data", "processed")
    raw_data: pd.DataFrame = pd.read_csv(Path("data", "raw", "raw_sum_stats.csv"))

    processed_train_data: pd.DataFrame
    processed_test_data: pd.DataFrame
    processed_train_data, processed_test_data = train_test_split(
        raw_data, test_size=test_size, random_state=seed
    )

    processed_train_data.to_csv(Path(processed_dpath, "processed_sum_stats_train.csv"), index=False)
    processed_test_data.to_csv(Path(processed_dpath, "processed_sum_stats_test.csv"), index=False)


if __name__ == "__main__":
    test_size: float = float(sys.argv[1])
    seed: Optional[int] = None
    try:
        seed = int(sys.argv[2])
    except IndexError:
        pass

    main(test_size=test_size, seed=seed)
