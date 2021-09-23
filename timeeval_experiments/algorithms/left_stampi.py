from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for left_stampi
def post_left_stampi(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 50)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


_left_stampi_parameters = {
 "anomaly_window_size": {
  "defaultValue": 50,
  "description": "Size of the sliding windows",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "n_init_train": {
  "defaultValue": 100,
  "description": "Fraction of data used to warmup streaming.",
  "name": "n_init_train",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 }
}


def left_stampi(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Left STAMPi",
        main=DockerAdapter(
            image_name="mut:5000/akita/left_stampi",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_left_stampi,
        params=_left_stampi_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )