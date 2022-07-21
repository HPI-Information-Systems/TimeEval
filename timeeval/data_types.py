from enum import Enum
from pathlib import Path
from typing import Union, Callable

import numpy as np


class TrainingType(Enum):
    """Training type of algorithm or dataset.

    TimeEval distinguishes between unsupervised, semi-supervised, and supervised algorithms.
    """
    UNSUPERVISED = "unsupervised"
    """An unsupervised algorithm does not require any training data.

    An unsupervised dataset consists only of a single test time series.
    """
    SEMI_SUPERVISED = "semi-supervised"
    """A semi-supervised algorithm requires normal data for training.

    A semi-supervised dataset consists of a training time series with normal data (no anomalies; all labels are 0) and
    a test time series.
    """
    SUPERVISED = "supervised"
    """A supervised algorithm requires training data with anomalies.

    A supervised dataset consists of a training time series with anomalies and a test time series.
    """

    @staticmethod
    def from_text(name: str) -> 'TrainingType':
        """Converts the string-representation to an Enum-object."""
        if name.lower() == TrainingType.SUPERVISED.value:
            return TrainingType.SUPERVISED
        elif name.lower() == TrainingType.SEMI_SUPERVISED.value:
            return TrainingType.SEMI_SUPERVISED
        else:  # if name.lower() == TrainingType.UNSUPERVISED.value:
            return TrainingType.UNSUPERVISED


class InputDimensionality(Enum):
    """Input dimensionality supported by an algorithm or of a dataset.

    TimeEval distinguishes between univariate and multivariate datasets / time series."""
    UNIVARIATE = "univariate"
    """Univariate datasets consist of a single feature/dimension/channel.

    An univariate algorithm can process only a dataset with a single feature/dimension/channel.
    """
    MULTIVARIATE = "multivariate"
    """Multivariate datasets have 2 or more features/dimensions/channels.

    A multivariate algorithm can process univariate or multivariate datasets.
    """

    @staticmethod
    def from_dimensions(n: int) -> 'InputDimensionality':
        """Converts the feature/dimension/channel count to an Enum-object."""
        if n < 1:
            raise ValueError(f"Zero dimensional dataset is not supported!")
        elif n == 1:
            return InputDimensionality.UNIVARIATE
        else:
            return InputDimensionality.MULTIVARIATE


class ExecutionType(Enum):
    """TimeEval calls each algorithm up to two times with different execution types.

    Depending on the algorithm's :class:`timeeval.data_types.TrainingType`, it requires a training step.
    TimeEval will call these algorithms first with the execution type set to ``TRAIN``.
    Then, for all algorithms, the algorithm is called with execution type ``EXECUTE``.
    """
    TRAIN = "train"
    EXECUTE = "execute"


AlgorithmParameter = Union[np.ndarray, Path]
TSFunction = Callable[[AlgorithmParameter, dict], AlgorithmParameter]
TSFunctionPost = Union[
    Callable[[AlgorithmParameter, dict], np.ndarray],
    Callable[[np.ndarray, dict], np.ndarray],
    Callable[[Path, dict], np.ndarray]
]
