from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.data_types import TrainingType, InputDimensionality
from .common import SKIP_PULL, DEFAULT_TIMEOUT

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for k-means
def post_kmeans(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 20)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


def kmeans(params: Any = None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="KMeans-docker",
        main=DockerAdapter(
            image_name="mut:5000/akita/kmeans",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_kmeans,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
