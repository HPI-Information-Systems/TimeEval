"""
An Evaluation Tool for Anomaly Detection Algorithms on Time Series Data.

Provides:
  1. tbd

How to use the documentation:

Documentation is available in two forms: docstrings provided with the code, and a loose standing reference guide,
available on `timeeval.readthedocs.io <https://timeeval.readthedocs.io>`_.

Code snippets in the docstring examples are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's or class' docstring::

  >>> from timeeval import TimeEval
  >>> help(TimeEval)
  ... # doctest: +SKIP
"""

from ._version import __version__
from .algorithm import Algorithm
from .data_types import AlgorithmParameter, TrainingType, InputDimensionality
from .datasets import Datasets, DatasetManager, MultiDatasetManager
from .remote_configuration import RemoteConfiguration
from .resource_constraints import ResourceConstraints
from .timeeval import TimeEval, Status
from .utils.metrics import Metric, DefaultMetrics
