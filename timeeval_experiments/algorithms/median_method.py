from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.data_types import TrainingType, InputDimensionality
from .common import SKIP_PULL, DEFAULT_TIMEOUT


def median_method(params: Any = None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="MedianMethod-docker",
        main=DockerAdapter(
            image_name="mut:5000/akita/median_method",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
