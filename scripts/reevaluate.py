import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from timeeval import Algorithm, Status, Metric, DefaultMetrics, MultiDatasetManager
from timeeval.adapters.docker import SCORES_FILE_NAME as DOCKER_SCORES_FILE_NAME
from timeeval.constants import RESULTS_CSV, HYPER_PARAMETERS, METRICS_CSV, ANOMALY_SCORES_TS
from timeeval.data_types import ExecutionType
from timeeval.core.experiments import Experiment as TimeEvalExperiment
from timeeval.utils.datasets import load_labels_only
from timeeval.utils.metrics import PrecisionAtK, FScoreAtK

# required to build a lookup-table for algorithm implementations
import timeeval_experiments.algorithms as algorithms
# noinspection PyUnresolvedReferences
from timeeval_experiments.algorithms import *
from timeeval_experiments.baselines import Baselines

INITIAL_WAITING_SECONDS = 5


def path_is_empty(path: Path) -> bool:
    return not any(path.iterdir())


class Evaluator:

    def __init__(self, results_path: Path, data_paths: List[Path], metrics: List[Metric]):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.results_path = results_path
        self.data_paths = data_paths
        self.metrics = metrics
        self.algos = self._build_algorithm_dict()
        self.dmgr = MultiDatasetManager(data_paths)
        self.df: pd.DataFrame = pd.read_csv(results_path / RESULTS_CSV)

        self._logger.warning(f"The Evaluator changes the results folder ({self.results_path}) in-place! "
                             "If you do not want this, cancel this script using Ctrl-C! "
                             f"Waiting {INITIAL_WAITING_SECONDS} seconds before continuing ...")
        time.sleep(INITIAL_WAITING_SECONDS)

    @staticmethod
    def _build_algorithm_dict() -> Dict[str, Algorithm]:
        algo_names = [a for a in dir(algorithms) if not a.startswith("__")]
        algo_list: List[Algorithm] = [eval(f"{a}()") for a in algo_names]
        algos: Dict[str, Algorithm] = {}
        for a in algo_list:
            algos[a.name] = a
        # add baselines
        increasing_baseline = Baselines.increasing()
        algos[increasing_baseline.name] = increasing_baseline
        random_baseline = Baselines.random()
        algos[random_baseline.name] = random_baseline
        normal_baseline = Baselines.normal()
        algos[normal_baseline.name] = normal_baseline
        # aliases for some renamed algorithms:
        algos["Image-embedding-CAE"] = algos["ImageEmbeddingCAE"]
        algos["LTSM-VAE"] = algos["LSTM-VAE"]
        return algos

    def evaluate(self, select_index: Optional[Path], evaluate_successful: bool = False):
        if select_index is None:
            exp_indices = self.df.index.values
        else:
            exp_indices = pd.read_csv(select_index).iloc[:, 0]
        self._logger.info(f"Re-evaluating {len(exp_indices)} experiments from {len(self.df)} experiments of "
                          f"folder {self.results_path}")
        for i in exp_indices:
            s_exp: pd.Series = self.df.iloc[i]
            if not evaluate_successful and s_exp.status == "Status.OK":
                self._logger.info(f"Exp-{i:06d}: Skipping, because experiment was successful.")
                continue

            self._logger.info(f"Exp-{i:06d}: Starting processing ...")
            exp_path = self._exp_path(s_exp)
            docker_scores_path = exp_path / DOCKER_SCORES_FILE_NAME
            processed_scores_path = exp_path / ANOMALY_SCORES_TS
            params_path = exp_path / HYPER_PARAMETERS
            metrics_path = exp_path / METRICS_CSV
            if not docker_scores_path.exists() or not params_path.exists():
                self._logger.error(f"Exp-{i:06d}: Experiment ({s_exp.algorithm}-{s_exp.collection}-{s_exp.dataset}) "
                                   "does not contain any results to start with (scores or hyper params are missing)!")
                continue

            y_true = load_labels_only(self.dmgr.get_dataset_path((s_exp.collection, s_exp.dataset)))
            if not evaluate_successful and processed_scores_path.exists():
                self._logger.debug(f"Exp-{i:06d}: Skipping reprocessing of anomaly scores, they are present.")
                y_scores = np.genfromtxt(processed_scores_path, delimiter=",")
            else:
                self._logger.debug(f"Exp-{i:06d}: Processing anomaly scores.")
                y_scores = np.genfromtxt(docker_scores_path, delimiter=",")
                post_fn = self.algos[s_exp.algorithm].postprocess
                if post_fn is not None:
                    with params_path.open("r") as fh:
                        hyper_params = json.load(fh)
                    dataset = self.dmgr.get(s_exp.collection, s_exp.dataset)
                    args = {
                        "executionType": ExecutionType.EXECUTE,
                        "results_path": exp_path,
                        "hyper_params": hyper_params,
                        "dataset_details": dataset
                    }
                    y_scores = post_fn(y_scores, args)
                _, y_scores = TimeEvalExperiment.scale_scores(y_true, y_scores)
                self._logger.info(f"Exp-{i:06d}: Writing anomaly scores to {processed_scores_path}.")
                y_scores.tofile(str(processed_scores_path), sep="\n")

            if not metrics_path.exists():
                metric_scores = {}
            else:
                metric_scores = pd.read_csv(metrics_path).iloc[0, :].to_dict()

            if not evaluate_successful and all(m.name in metric_scores for m in self.metrics):
                self._logger.debug(f"Exp-{i:06d}: Skipping re-assessment of metrics, they are all present.")
                errors = 0
            else:
                self._logger.debug(f"Exp-{i:06d}: Re-assessing metrics.")
                results = {}
                errors = 0
                for metric_name in self.metrics:
                    try:
                        score = metric_name(y_true, y_scores)
                        results[metric_name.name] = score
                    except Exception as e:
                        self._logger.warning(f"Exp-{i:06d}: Exception while computing metric {metric_name}!", exc_info=e)
                        errors += 1
                        continue

                # update metrics and write them to disk
                metric_scores.update(results)
                if metric_scores:
                    self._logger.info(f"Exp-{i:06d}: Writing new metrics to {metrics_path}!")
                    pd.DataFrame([metric_scores]).to_csv(metrics_path, index=False)
                else:
                    self._logger.warning(f"Exp-{i:06d}: No metrics computed!")

            if metric_scores and errors == 0:
                self._logger.debug(f"Exp-{i:06d}: Updating status to success (Status.OK).")
                s_update = s_exp.copy()
                s_update["status"] = Status.OK
                s_update["error_message"] = "(fixed)"
                for metric_name in metric_scores:
                    if metric_name in s_update:
                        s_update[metric_name] = metric_scores[metric_name]
                self.df.iloc[i] = s_update
            self._logger.info(f"Exp-{i:06d}: ... finished processing.")

        self._logger.info(f"Overwriting results file at {self.results_path / RESULTS_CSV}")
        self.df.to_csv(self.results_path / RESULTS_CSV, index=False)

    def _exp_path(self, exp: pd.Series) -> Path:
        return (self.results_path
                / exp.algorithm
                / exp.hyper_params_id
                / exp.collection
                / exp.dataset
                / str(exp.repetition))


_metrics = {
    DefaultMetrics.ROC_AUC.name: DefaultMetrics.ROC_AUC,
    DefaultMetrics.PR_AUC.name: DefaultMetrics.PR_AUC,
    DefaultMetrics.FIXED_RANGE_PR_AUC.name: DefaultMetrics.FIXED_RANGE_PR_AUC,
    DefaultMetrics.AVERAGE_PRECISION.name: DefaultMetrics.AVERAGE_PRECISION,
    PrecisionAtK().name: PrecisionAtK(),
    FScoreAtK().name: FScoreAtK()
}


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Takes an experiment result folder and re-evaluates (standardization and metric calculation) "
                    "selected experiments."
    )
    parser.add_argument("result_folder", type=Path,
                        help="Folder of the experiment")
    parser.add_argument("data_folders", type=Path, nargs="*",
                        help="Folders, where the datasets from the experiment are stored.")
    parser.add_argument("--select", type=Path,
                        help="Experiments to reevaluate (indices to results.csv; single column with header 'index').")
    parser.add_argument("--loglevel", default="INFO", choices=("ERROR", "WARNING", "INFO", "DEBUG"),
                        help="Set logging verbosity (default: %(default)s)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Set this flag if successful experiments (Status.OK) should be reevaluated as well")
    parser.add_argument("--metrics", type=str, nargs="*", default=["ROC_AUC", "PR_AUC", "RANGE_PR_AUC"],
                        choices=list(_metrics.keys()),
                        help="Metrics to re-calculate. (default: %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _create_arg_parser()
    logging.basicConfig(level=args.loglevel)
    selected_metrics = args.metrics
    selected_metrics = [_metrics[m] for m in selected_metrics]

    rs = Evaluator(args.result_folder, args.data_folders, selected_metrics)
    rs.evaluate(args.select, args.force)
