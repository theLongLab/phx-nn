# src/logging/logger.py

import json
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl


class DictLogger(pl.logging.LightningLoggerBase):
    """
    Dictionary logger.

    Parameters
    ----------
    version : str
        The version to attach to this logger.

    Methods
    -------
    log_metrics(step_num=None)
        Append metric to the metric list.
    """

    def __init__(self, version: str, train_logs_dir: Path) -> None:
        super().__init__()
        self.metrics: List[str] = []
        self._version: str = version
        self._save_path: Path = Path(train_logs_dir, "{}_final_metrics.log".format(self._version))

    @pl.logging.rank_zero_only
    def log_metrics(self, metric: str, step_num: Optional[int] = None) -> None:
        """
        Append metric to the metric list.

        Parameters
        ----------
        metric : str
            Metric in question.

        step_num : int or None
            Optional step number, not utilized here.
        """
        self.metrics.append(metric)

    def save(self) -> None:
        with self._save_path.open("w") as log:
            json.dump(self.metrics, log)

    @property
    def version(self) -> str:
        """
        Return logger version as a string.
        """
        return self._version
