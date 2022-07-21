from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any

from timeeval.adapters.base import Adapter
from timeeval.data_types import (
    TSFunction, TSFunctionPost, ExecutionType, AlgorithmParameter, TrainingType, InputDimensionality
)
from timeeval.params import ParameterConfig


@dataclass
class Algorithm:
    """
    This class is a wrapper for any `Adapter` and an instruction plan for the TimeEval tool.
    It tells TimeEval what algorithm to execute, what pre- and post-steps to perform
    and how the parameters and data are provided to the algorithm.
    Moreover, it defines attributes that are necessary to help TimeEval know
    what kind of time series can be put into the algorithm.

    Parameters
    ----------
    name : str
        The name of the Algorithm shown in the results.
    main : timeeval.adapters.base.Adapter
        The adapter implementation that contains the algorithm to evaluate.
    preprocess : Optional[TSFunction]
        Optional function to perform before ``main`` to modify input data.
    postprocess : Optional[TSFunctionPost]
        Optional function to perform after ``main`` to modify output data.
    data_as_file : bool
        Whether the data input is a ``Path`` or a ``numpy.ndarray``.
    param_schema : Dict[str, Dict[str, Any]]
        Optional schema of the algorithm's input parameters needed by :class:`timeeval_experiments.algorithm_configurator.AlgorithmConfigurator`.
        Schema definition::

            [
                "param_name": {
                    "name": str
                    "defaultValue": Any
                    "description": str
                    "type": str
                },
            ]

    param_config : timeeval.params.search.ParameterConfig
        Optional object of type ParameterConfig to define a search grid or fixed parameters.
    training_type : timeeval.data_types.TrainingType
        Definition of training type to receive the correct dataset formats (needed if TimeEval is run with
        ``force_training_type_match`` config).
    input_dimensionality : timeeval.data_types.InputDimensionality
        Definition of training type to receive the correct dataset formats (needed if TimeEval is run with
        ``force_dimensionality_match`` config option).

    Examples
    --------
    Create a baseline algorithm that always assigns a normal anomaly score:

    >>> import numpy as np
    >>> from timeeval import Algorithm
    >>> from timeeval.adapters import FunctionAdapter
    >>> my_fn = lambda X, args: np.zeros(len(X))
    >>> Algorithm(name="Test Algorithm", main=FunctionAdapter(my_fn), data_as_file=False)
    """

    name: str
    main: Adapter
    preprocess: Optional[TSFunction] = None
    postprocess: Optional[TSFunctionPost] = None
    data_as_file: bool = False
    param_schema: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})
    param_config: ParameterConfig = ParameterConfig.defaults()
    training_type: TrainingType = TrainingType.UNSUPERVISED
    input_dimensionality: InputDimensionality = InputDimensionality.UNIVARIATE

    def train(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        """Execute this algorithm's training procedure.

        .. warning::
            Internal API!

        This method sets the algorithms :class:`~timeeval.data_types.ExecutionType` to ``TRAIN`` and then calls the
        adapter implementation.
        Calling ``train()`` on an unsupervised algorithm is not supported and will raise a :class:`ValueError`.

        Parameters
        ----------
        dataset : timeeval.data_types.AlgorithmParameter
            Either a numpy-array containing the training time series data or a path to the training time series file.
        args : dict
            TimeEval arguments that get passed to the :class:`~timeeval.adapters.base.Adapter`-implementation. Example::

                {
                    "hyper_params": {},
                    "results_path": Path("results"),
                    "resource_constraints": ResourceConstraints(),
                    "dataset_details": ...
                }

        Returns
        -------
        results : timeeval.data_types.AlgorithmParameter
            Optionally returns the training results or a path to the model-file.

            .. note::
                Currently not used by TimeEval!

        :meta private:
        """
        if self.training_type == TrainingType.UNSUPERVISED:
            raise ValueError("Calling 'train()' on an unsupervised algorithm is not supported! "
                             f"Algorithm '{self.name}' has training type '{self.training_type.value}', but you tried "
                             f"to execute the training step. Please check the algorithm configuration for {self.name}!")
        args = args or {}
        args["executionType"] = ExecutionType.TRAIN
        return self.main(dataset, args)

    def execute(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        """Execute this algorithm's test/execute procedure.

        .. warning::
            Internal API!

        This method sets the algorithms :class:`~timeeval.data_types.ExecutionType` to ``EXECUTE`` and then calls the
        adapter implementation.

        Parameters
        ----------
        dataset : timeeval.data_types.AlgorithmParameter
            Either a numpy-array containing the test time series data or a path to the test time series file.
        args : dict
            TimeEval arguments that get passed to the :class:`~timeeval.adapters.base.Adapter`-implementation. Example::

                {
                    "hyper_params": {},
                    "results_path": Path("results"),
                    "resource_constraints": ResourceConstraints(),
                    "dataset_details": ...
                }

        Returns
        -------
        scores : timeeval.data_types.AlgorithmParameter
            Should return the anomaly scores for the test time series, but can also return something else.
            The implementation of the `postprocess`-function must take care of transforming the result to valid
            anomaly scores.

        :meta private:
        """
        args = args or {}
        args["executionType"] = ExecutionType.EXECUTE
        return self.main(dataset, args)

    def prepare_fn(self) -> Optional[Callable[[], None]]:
        """Returns the prepare-step-function of this algorithm.

        .. warning::
            Internal API!

        :meta private:
        """
        return self.main.get_prepare_fn()

    def prepare(self) -> None:
        """Executes the prepare-step of this algorithm.

        .. warning::
            Internal API!

        :meta private:
        """
        fn = self.prepare_fn()
        if fn:
            fn()

    def finalize_fn(self) -> Optional[Callable[[], None]]:
        """Returns the finalize-step-function of this algorithm.

        .. warning::
            Internal API!

        :meta private:
        """
        return self.main.get_finalize_fn()

    def finalize(self) -> None:
        """Executes the finalize-step of this algorithm.

        .. warning::
            Internal API!

        :meta private:
        """
        fn = self.finalize_fn()
        if fn:
            fn()
