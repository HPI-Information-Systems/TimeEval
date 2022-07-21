import datetime as dt
import logging
import signal
import socket
import subprocess
from enum import Enum
from pathlib import Path
from time import time
from types import FrameType
from typing import Callable, List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import tqdm
from distributed.client import Future
from joblib import Parallel, delayed

from .adapters.docker import DockerTimeoutError
from .algorithm import Algorithm
from .constants import RESULTS_CSV
from .core.experiments import Experiments, Experiment
from .core.remote import Remote, RemoteConfiguration
from .core.times import Times
from .data_types import TrainingType, InputDimensionality
from .datasets import Datasets
from .resource_constraints import ResourceConstraints
from .utils.encode_params import dumps_params
from .utils.metrics import Metric, DefaultMetrics
from .utils.tqdm_joblib import tqdm_joblib


class Status(Enum):
    """Status of an experiment.

    The status of each evaluation experiment can have one of three states: ok, error, or timeout.
    """
    OK = 0
    ERROR = 1
    TIMEOUT = 2


class TimeEval:
    """

    TimeEval evaluates all algorithms on all datasets per default.
    You can use the parameters ``skip_invalid_combinations``, ``force_training_type_match``,
    ``force_dimensionality_match``, and ``experiment_combinations_file`` to control which algorithm runs on which dataset.

    Parameters
    ----------
    dataset_mgr : ~timeeval.datasets.datasets.Datasets
        The dataset manager provides the metadata about the datasets. You can either use a
        :class:`~timeeval.datasets.dataset_manager.DatasetManager` or a
        :class:`~timeeval.datasets.multi_dataset_manager.MultiDatasetManager`.
    datasets : List[Tuple[str, str]]
        List of dataset IDs consisting of collection name and dataset name to uniquely identify each dataset. The
        datasets must be known by the ``dataset_mgr``. You can call :func:`~timeeval.datasets.datasets.Datasets.select`
        on the ``dataset_mgr`` to get a list of dataset IDs.
    algorithms : List[Algorithm]
        List of algorithms to evaluate on the datasets.
        The algorithm specification also contains the hyperparameter configurations that TimeEval will test.
    results_path : Path
        Use this parameter to change the path where all evaluation results are stored.
        If TimeEval is used in distributed mode, this path will be created on **all** nodes!
    repetitions : int
        Execute each unique combination of dataset, algorithm, and hyperparameter-setting multiple times.
        This allows you to use TimeEval to measure runtimes more precisely by aggregating the runtime measurements over
        multiple repetitions.
    distributed : bool
        Run TimeEval in distributed mode.
        In this case, you **should** also supply a ``remote_config``.
    remote_config : Optional[RemoteConfiguration]
    resource_constraints : Optional[ResourceConstraints]
    disable_progress_bar : bool
        Don't show the `tqdm <https://tqdm.github.io>`_ progress bars.
    metrics : Optional[List[Metric]]
    skip_invalid_combinations : bool
    force_training_type_match : bool
    force_dimensionality_match : bool
    n_jobs : int
    experiment_combinations_file : Optional[Path]
    """

    RESULT_KEYS = ["algorithm",
                   "collection",
                   "dataset",
                   "algo_training_type",
                   "algo_input_dimensionality",
                   "dataset_training_type",
                   "dataset_input_dimensionality",
                   "train_preprocess_time",
                   "train_main_time",
                   "execute_preprocess_time",
                   "execute_main_time",
                   "execute_postprocess_time",
                   "status",
                   "error_message",
                   "repetition",
                   "hyper_params",
                   "hyper_params_id"]
    """This list contains all the _fixed_ result data frame's column headers.
    TimeEval dynamically adds the metrics and execution times depending on its configuration.
    
    For metrics, their :func:`~timeeval.utils.metrics.Metric.name` will be used as column header, and TimeEval will add
    the following runtime measurements depending on whether they are applicable to the algorithms in the run or not:
    
    - train_preprocess_time: if :func:`~timeeval.algorithm.Algorithm.preprocess` is defined
    - train_main_time: if the algorithm is semi-supervised or supervised
    - execute_preprocess_time: if :func:`~timeeval.algorithm.Algorithm.preprocess` is defined
    - execute_main_time: always
    - execute_postprocess_time: if :func:`~timeeval.algorithm.Algorithm.postprocess` is defined
    """

    DEFAULT_RESULT_PATH = Path("./results")
    """Default path for the results.
    
    If you don't specify the ``results_path``, TimeEval will store the evaluation results in the folder ``results``
    within the current working directory.
    """

    def __init__(self,
                 dataset_mgr: Datasets,
                 datasets: List[Tuple[str, str]],
                 algorithms: List[Algorithm],
                 results_path: Path = DEFAULT_RESULT_PATH,
                 repetitions: int = 1,
                 distributed: bool = False,
                 remote_config: Optional[RemoteConfiguration] = None,
                 resource_constraints: Optional[ResourceConstraints] = None,
                 disable_progress_bar: bool = False,
                 metrics: Optional[List[Metric]] = None,
                 skip_invalid_combinations: bool = True,
                 force_training_type_match: bool = False,
                 force_dimensionality_match: bool = False,
                 n_jobs: int = -1,
                 experiment_combinations_file: Optional[Path] = None) -> None:
        assert len(datasets) > 0, "No datasets given for evaluation!"
        assert len(algorithms) > 0, "No algorithms given for evaluation!"
        assert repetitions > 0, "Negative or 0 repetitions are not supported!"
        assert n_jobs >= -1, f"n_jobs={n_jobs} not supported (must be >= -1)!"
        if experiment_combinations_file is not None:
            assert experiment_combinations_file.exists(), "Experiment combination file not found!"

        dataset_details = []
        not_found_datasets = []
        for d in datasets:
            try:
                dataset_details.append(dataset_mgr.get(d))
            except KeyError:
                not_found_datasets.append(repr(d))
        assert len(
            not_found_datasets) == 0, "Some datasets could not be found in DatasetManager!\n  " \
                                      f"{', '.join(not_found_datasets)}"

        self.log = logging.getLogger(self.__class__.__name__)
        start_date: str = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.results_path = results_path.resolve() / start_date
        self.disable_progress_bar = disable_progress_bar
        self.metrics = metrics or DefaultMetrics.default_list()
        self.metric_names = [m.name for m in self.metrics]
        self.results = pd.DataFrame(columns=TimeEval.RESULT_KEYS + self.metric_names)
        self.distributed = distributed
        self.n_jobs = n_jobs

        self.log.info(f"Results are recorded in the directory {self.results_path}")
        self.results_path.mkdir(parents=True, exist_ok=True)

        resource_constraints = resource_constraints or ResourceConstraints()
        if not distributed and resource_constraints.tasks_per_host > 1:
            self.log.warning(
                f"`tasks_per_host` was set to {resource_constraints.tasks_per_host}. However, non-distributed "
                "execution of TimeEval does currently not support parallelism! Reducing `tasks_per_host` to 1. "
                "The automatic resource limitation will reflect this by increasing the limits for the single task. "
                "Explicitly set constraints to limit the resources for local executions of TimeEval."
            )
            resource_constraints.tasks_per_host = 1

        self.exps = Experiments(dataset_mgr, dataset_details, algorithms, self.results_path,
                                resource_constraints=resource_constraints,
                                repetitions=repetitions,
                                skip_invalid_combinations=skip_invalid_combinations,
                                force_training_type_match=force_training_type_match,
                                force_dimensionality_match=force_dimensionality_match,
                                metrics=self.metrics,
                                experiment_combinations_file=experiment_combinations_file)

        self.remote_config = remote_config or RemoteConfiguration()
        self.remote_config.update_logging_path(self.results_path)
        self.log.debug(f"Updated dask logging filepath to {self.remote_config.dask_logging_filename}")

        if self.distributed:
            self.log.info("TimeEval is running in distributed environment, setting up remoting ...")
            self.remote = Remote(disable_progress_bar=self.disable_progress_bar, remote_config=self.remote_config,
                                 resource_constraints=resource_constraints)
            self.results["future_result"] = np.nan

            self.log.info("... registering signal handlers ...")
            orig_handler: Callable[[int, Optional[FrameType]], Any] = signal.getsignal(signal.SIGINT)  # type: ignore

            def sigint_handler(sig: int, frame: Optional[FrameType] = None) -> Any:
                self.log.warning(f"SIGINT ({sig}) received, shutting down cluster. Please look for dangling Docker "
                                 "containers on all worker nodes (we do not remove them when terminating "
                                 "ungracefully).")
                self.remote.close()
                return orig_handler(sig, frame)
            signal.signal(signal.SIGINT, sigint_handler)

            self.log.info("... remoting setup done.")

    def _run(self) -> None:
        desc = "Submitting evaluation tasks" if self.distributed else "Evaluating"
        for exp in tqdm.tqdm(self.exps, desc=desc, disable=self.disable_progress_bar):
            try:
                future_result: Optional[Future] = None
                result: Optional[Dict] = None

                if exp.algorithm.training_type in [TrainingType.SUPERVISED, TrainingType.SEMI_SUPERVISED]:
                    if exp.resolved_train_dataset_path is None:
                        # Intentionally raise KeyError here if no training dataset is specified.
                        # The Error will be caught by the except clause below.
                        raise KeyError("Path to training dataset not found!")

                    # This check is not necessary for unsupervised algorithms, because they can be executed on all
                    # datasets.
                    if exp.algorithm.training_type != exp.dataset.training_type:
                        raise ValueError(f"Dataset training type ({exp.dataset.training_type}) incompatible to "
                                         f"algorithm training type ({exp.algorithm.training_type})!")

                if (exp.algorithm.input_dimensionality == InputDimensionality.UNIVARIATE and
                        exp.dataset.input_dimensionality == InputDimensionality.MULTIVARIATE):
                    raise ValueError(f"Dataset input dimensionality ({exp.dataset.input_dimensionality}) incompatible "
                                     f"to algorithm input dimensionality ({exp.algorithm.input_dimensionality})!")

                if self.distributed:
                    future_result = self.remote.add_task(exp.evaluate, key=exp.name)
                else:
                    result = exp.evaluate()
                self._record_results(exp, result=result, future_result=future_result)

            except DockerTimeoutError as e:
                self.log.exception(f"Evaluation of {exp.algorithm.name} on the dataset {exp.dataset} timed out.")
                result = {m: np.nan for m in self.metric_names}
                self._record_results(exp, result=result, status=Status.TIMEOUT, error_message=repr(e))
            except Exception as e:
                self.log.exception(f"Exception occurred during the evaluation of {exp.algorithm.name} on the dataset {exp.dataset}.")
                result = {m: np.nan for m in self.metric_names}
                self._record_results(exp, result=result, status=Status.ERROR, error_message=repr(e))

    def _record_results(self,
                        exp: Experiment,
                        result: Optional[Dict] = None,
                        future_result: Optional[Future] = None,
                        status: Status = Status.OK,
                        error_message: Optional[str] = None) -> None:
        new_row = {
            "algorithm": exp.algorithm.name,
            "collection": exp.dataset_collection,
            "dataset": exp.dataset_name,
            "algo_training_type": exp.algorithm.training_type.name,
            "algo_input_dimensionality": exp.algorithm.input_dimensionality.name,
            "dataset_training_type": exp.dataset.training_type.name,
            "dataset_input_dimensionality": exp.dataset.input_dimensionality.name,
            "status": status,
            "error_message": error_message,
            "repetition": exp.repetition,
            "hyper_params": dumps_params(exp.params),
            "hyper_params_id": exp.params_id
        }
        if result is not None and future_result is None:
            new_row.update(result)
        elif result is None and future_result is not None:
            new_row.update({"future_result": future_result})
        self.results = self.results.append(new_row, ignore_index=True)
        self.results.replace(to_replace=[None], value=np.nan, inplace=True)

    def _resolve_future_results(self) -> None:
        self.remote.fetch_results()

        result_keys = self.metric_names + Times.result_keys()
        status_keys = ["status", "error_message"]
        keys = result_keys + status_keys

        def get_future_result(f: Future) -> Tuple[Any, ...]:
            try:
                r = f.result()
                return tuple(r.get(k, None) for k in result_keys) + (Status.OK, None)
            except DockerTimeoutError as e:
                self.log.exception(f"Exception {repr(e)} occurred remotely.")
                status = Status.TIMEOUT
                error_message = repr(e)
            except Exception as e:
                self.log.exception(f"Exception {repr(e)} occurred remotely.")
                status = Status.ERROR
                error_message = repr(e)

            return tuple(np.nan for _ in result_keys) + (status, error_message)

        self.results[keys] = self.results["future_result"].apply(get_future_result).tolist()
        self.results = self.results.drop(['future_result'], axis=1)

    def get_results(self, aggregated: bool = True, short: bool = True) -> pd.DataFrame:
        if not aggregated:
            return self.results

        df = self.results

        if Status.ERROR in df.status.unique() or Status.TIMEOUT in df.status.unique():
            self.log.warning("The results contain errors which are filtered out for the final aggregation. "
                             "To see all results, call .get_results(aggregated=False)")
            df = df[df.status == Status.OK]

        if short:
            time_names = ["train_main_time", "execute_main_time"]
            group_names = ["algorithm", "collection", "dataset"]
        else:
            time_names = Times.result_keys()
            group_names = ["algorithm", "collection", "dataset", "hyper_params_id"]
        keys = [key for key in self.metric_names + time_names if key in df.columns]
        grouped_results = df.groupby(group_names)
        results: pd.DataFrame = grouped_results[keys].mean()

        if short:
            results = results.rename(columns=dict([(k, f"{k}_mean") for k in keys]))
        else:
            std_results = grouped_results.std()[keys]
            results = results.join(std_results, lsuffix="_mean", rsuffix="_std")
        results["repetitions"] = grouped_results["repetition"].count()
        return results

    def save_results(self, results_path: Optional[Path] = None) -> None:
        path = results_path or (self.results_path / RESULTS_CSV)
        self.results.to_csv(path.resolve(), index=False)

    @staticmethod
    def rsync_results_from(results_path: Path, hosts: List[str], disable_progress_bar: bool = False, n_jobs: int = -1) -> None:
        results_path = results_path.resolve()
        hostname = socket.gethostname()
        excluded_aliases = [
            hostname,
            socket.gethostbyname(hostname),
            "localhost",
            socket.gethostbyname("localhost")
        ]
        jobs = [
            delayed(subprocess.call)(["rsync", "-a", f"{host}:{results_path}/", f"{results_path}"])
            for host in hosts
            if host not in excluded_aliases
        ]
        with tqdm_joblib(tqdm.tqdm(hosts, desc="Collecting results", disable=disable_progress_bar, total=len(jobs))):
            Parallel(n_jobs)(jobs)

    def rsync_results(self) -> None:
        TimeEval.rsync_results_from(
            self.results_path,
            self.remote_config.worker_hosts,
            self.disable_progress_bar,
            self.n_jobs
        )

    def _prepare(self) -> None:
        n = len(self.exps)
        self.log.debug(f"Running {n} algorithm prepare steps")
        for algorithm in self.exps.algorithms:
            algorithm.prepare()
        self.log.debug(f"Creating {n} result directories")
        for exp in self.exps:
            exp.results_path.mkdir(parents=True, exist_ok=True)

    def _finalize(self) -> None:
        self.log.debug(f"Running {len(self.exps)} algorithm finalize steps")
        for algorithm in self.exps.algorithms:
            algorithm.finalize()

    def _distributed_prepare(self) -> None:
        tasks: List[Tuple[Callable, List, Dict]] = []
        for algorithm in self.exps.algorithms:
            prepare_fn = algorithm.prepare_fn()
            if prepare_fn:
                tasks.append((prepare_fn, [], {}))
        self.log.debug(f"Collected {len(tasks)} algorithm prepare steps")

        def mkdirs(dirs: List[Path]) -> None:
            for d in dirs:
                d.mkdir(parents=True, exist_ok=True)

        dir_list = [exp.results_path for exp in self.exps]
        tasks.append((mkdirs, [dir_list], {}))
        self.log.debug(f"Collected {len(dir_list)} directories to create on remote nodes")
        self.remote.run_on_all_hosts(tasks, msg="Preparing")

    def _distributed_finalize(self) -> None:
        tasks: List[Tuple[Callable, List, Dict]] = []
        for algorithm in self.exps.algorithms:
            finalize_fn = algorithm.finalize_fn()
            if finalize_fn:
                tasks.append((finalize_fn, [], {}))
        self.log.debug(f"Collected {len(tasks)} algorithm finalize steps")
        self.log.info("Running finalize steps on remote hosts")
        self.remote.run_on_all_hosts(tasks, msg="Finalizing")
        self.log.info("Closing remote")
        self.remote.close()
        self.log.info("Syncing results")
        self.rsync_results()

    def run(self) -> None:
        print("Running PREPARE phase")
        self.log.info("Running PREPARE phase")
        t0 = time()
        if self.distributed:
            self._distributed_prepare()
        else:
            self._prepare()

        print("Running EVALUATION phase")
        self.log.info("Running EVALUATION phase")
        self._run()
        if self.distributed:
            self._resolve_future_results()
        self.save_results()

        print("Running FINALIZE phase")
        self.log.info("Running FINALIZE phase")
        if self.distributed:
            self._distributed_finalize()
        else:
            self._finalize()
        t1 = time()

        msg = f"""FINALIZE phase done.
          Stored results at {self.results_path / RESULTS_CSV}.
          Overall runtime of this TimeEval run: {t1 - t0} seconds
        """
        print(msg)
        self.log.info(msg)
