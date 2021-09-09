from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter


_lstm_vae_parameters = {
 "batch_size": {
  "defaultValue": 32,
  "description": "size of batch given for each iteration",
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
 "latent_size": {
  "defaultValue": 5,
  "description": "dimension of latent space",
  "name": "latent_size",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.001,
  "description": "rate at which the gradients are updated",
  "name": "learning_rate",
  "type": "float"
 },
 "lstm_layers": {
  "defaultValue": 10,
  "description": "number of layers in lstm",
  "name": "lstm_layers",
  "type": "int"
 },
 "n_epochs": {
  "defaultValue": 10,
  "description": "number of iterations we train the model",
  "name": "n_epochs",
  "type": "int"
 },
 "rnn_hidden_size": {
  "defaultValue": 5,
  "description": "LTSM cells hidden dimension",
  "name": "rnn_hidden_size",
  "type": "int"
 },
 "window_size": {
  "defaultValue": 10,
  "description": "number of datapoints that the model takes once",
  "name": "window_size",
  "type": "int"
 }
}


def lstm_vae(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="LSTM-VAE",
        main=DockerAdapter(
            image_name="mut:5000/akita/lstm_vae",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_lstm_vae_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
