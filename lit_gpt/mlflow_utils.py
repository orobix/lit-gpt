import logging
import re
from argparse import Namespace
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from lightning.fabric.utilities.logger import (
    _add_prefix,
    _convert_params,
    _flatten_dict,
)
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn
from typing_extensions import override

log = logging.getLogger(__name__)


class CustomMLFlowLogger(MLFlowLogger):
    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: str | None = None,
        tracking_uri: str | None = ...,
        tags: Dict[str, Any] | None = None,
        save_dir: str | None = "./mlruns",
        log_model: bool | Literal["all"] = False,
        prefix: str = "",
        artifact_location: str | None = None,
        run_id: str | None = None,
        synchronous: bool = True,
    ):
        super().__init__(
            experiment_name,
            run_name,
            tracking_uri,
            tags,
            save_dir,
            log_model,
            prefix,
            artifact_location,
            run_id,
        )
        self.synchronous = synchronous

    @override
    @rank_zero_only
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
    ) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        from mlflow.entities import Metric

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        metrics_list: List[Metric] = []

        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                log.warning(f"Discarding metric with string value {k}={v}.")
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
            if k != new_k:
                rank_zero_warn(
                    "MLFlow only allows '_', '/', '.' and ' ' special characters in metric name."
                    f" Replacing {k} with {new_k}.",
                    category=RuntimeWarning,
                )
                k = new_k
            metrics_list.append(
                Metric(key=k, value=v, timestamp=timestamp_ms, step=step or 0)
            )

        self.experiment.log_batch(
            run_id=self.run_id, metrics=metrics_list, synchronous=self.synchronous
        )

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:  # type: ignore[override]
        params = _convert_params(params)
        params = _flatten_dict(params)

        from mlflow.entities import Param

        # Truncate parameter values to 250 characters.
        # TODO: MLflow 1.28 allows up to 500 characters: https://github.com/mlflow/mlflow/releases/tag/v1.28.0
        params_list = [Param(key=k, value=str(v)[:250]) for k, v in params.items()]

        # Log in chunks of 100 parameters (the maximum allowed by MLflow).
        for idx in range(0, len(params_list), 100):
            self.experiment.log_batch(
                run_id=self.run_id,
                params=params_list[idx : idx + 100],
                synchronous=self.synchronous,
            )
