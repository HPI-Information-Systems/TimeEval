from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import FullParameterGrid

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for DeepAnT
def _post_deepant(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 45)
    prediction_window_size = args.get("hyper_params", {}).get("prediction_window_size", 1)
    size = window_size + prediction_window_size + 1
    return ReverseWindowing(window_size=size).fit_transform(scores)


_deepant_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 45,
  "description": "Batch size for input data",
  "name": "batch_size",
  "type": "int"
 },
 "early_stopping_delta": {
  "defaultValue": 0.05,
  "description": "If 1 - (loss / last_loss) is less than `delta` for `patience` epochs, stop",
  "name": "early_stopping_delta",
  "type": "float"
 },
 "early_stopping_patience": {
  "defaultValue": 10,
  "description": "If 1 - (loss / last_loss) is less than `delta` for `patience` epochs, stop",
  "name": "early_stopping_patience",
  "type": "int"
 },
 "epochs": {
  "defaultValue": 50,
  "description": "Number of training epochs",
  "name": "epochs",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 1e-05,
  "description": "Learning rate",
  "name": "learning_rate",
  "type": "float"
 },
 "prediction_window_size": {
  "defaultValue": 1,
  "description": "Prediction window: Number of data points that will be predicted from each window",
  "name": "prediction_window_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "Train-validation split for early stopping",
  "name": "split",
  "type": "float"
 },
 "window_size": {
  "defaultValue": 45,
  "description": "History window: Number of time stamps in history, which are taken into account",
  "name": "window_size",
  "type": "int"
 }
}


def deepant(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="DeepAnT",
        main=DockerAdapter(
            image_name="mut:5000/akita/deepant",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=_post_deepant,
        params=_deepant_parameters,
        param_grid=FullParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
