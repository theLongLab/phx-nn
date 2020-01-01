# src/logging/logger.py

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

    def __init__(self, version: str) -> None:
        super().__init__()
        self.metrics: List[str] = []
        self._version: str = version

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

    @property
    def version(self) -> str:
        """
        Return logger version as a string.
        """
        return self._version
