from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for TARZAN
def post_tarzan(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 50)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


_tarzan_parameters = {
 "alphabet_size": {
  "defaultValue": 10,
  "description": "Number of symbols used for discretization by SAX (performance parameter)",
  "name": "alphabet_size",
  "type": "int"
 },
 "anomaly_window_size": {
  "defaultValue": 20,
  "description": "Size of the sliding window. Equal to the discord length!",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def tarzan(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="TARZAN",
        main=DockerAdapter(
            image_name="mut:5000/akita/tarzan",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_tarzan,
        params=_tarzan_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
