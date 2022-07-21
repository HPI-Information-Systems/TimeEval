import tempfile
import unittest
from pathlib import Path
from typing import Iterable

import numpy as np

from tests.fixtures.algorithms import SupervisedDeviatingFromMean
from timeeval import (
    TimeEval,
    Algorithm,
    Datasets,
    TrainingType,
    InputDimensionality,
    Status,
    DefaultMetrics,
    ResourceConstraints,
    DatasetManager
)
from timeeval.datasets import Dataset, DatasetRecord
from timeeval.core.experiments import Experiment, Experiments
from timeeval.utils.hash_dict import hash_dict


class TestDatasetAndAlgorithmMatch(unittest.TestCase):

    def setUp(self) -> None:
        self.dmgr = DatasetManager("./tests/example_data")
        self.algorithms = [
            Algorithm(
                name="supervised_deviating_from_mean",
                main=SupervisedDeviatingFromMean(),
                training_type=TrainingType.SUPERVISED,
                input_dimensionality=InputDimensionality.UNIVARIATE,
                data_as_file=False
            )
        ]

    def _prepare_dmgr(self, path: Path,
                      training_type: Iterable[str] = ("unsupervised",),
                      dimensionality: Iterable[str] = ("univariate",)) -> Datasets:
        dmgr = DatasetManager(path / "data")
        for t, d in zip(training_type, dimensionality):
            dmgr.add_dataset(DatasetRecord(
                collection_name="test",
                dataset_name=f"dataset-{t}-{d}",
                train_path="train.csv",
                test_path="test.csv",
                dataset_type="synthetic",
                datetime_index=False,
                split_at=-1,
                train_type=t,
                train_is_normal=True if t == "semi-supervised" else False,
                input_type=d,
                length=10000,
                dimensions=5 if d == "multivariate" else 1,
                contamination=0.1,
                num_anomalies=1,
                min_anomaly_length=100,
                median_anomaly_length=100,
                max_anomaly_length=100,
                mean=0.0,
                stddev=1.0,
                trend="no-trend",
                stationarity="stationary",
                period_size=50
            ))
        return dmgr

    def test_supervised_algorithm(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.dmgr, [("test", "dataset-datetime")], self.algorithms,
                                repetitions=1,
                                results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results(aggregated=False)

        np.testing.assert_array_almost_equal(results["ROC_AUC"].values, [0.810225])

    def test_mismatched_training_type(self):
        algo = Algorithm(
            name="supervised_deviating_from_mean",
            main=SupervisedDeviatingFromMean(),
            training_type=TrainingType.SEMI_SUPERVISED,
            data_as_file=False
        )
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.dmgr, [("test", "dataset-datetime")], [algo],
                                repetitions=1,
                                results_path=Path(tmp_path),
                                skip_invalid_combinations=False)
            timeeval.run()
        results = timeeval.get_results(aggregated=False)

        self.assertEqual(results.loc[0, "status"], Status.ERROR)
        self.assertIn("training type", results.loc[0, "error_message"])
        self.assertIn("incompatible", results.loc[0, "error_message"])

    def test_mismatched_input_dimensionality(self):
        algo = Algorithm(
            name="supervised_deviating_from_mean",
            main=SupervisedDeviatingFromMean(),
            input_dimensionality=InputDimensionality.UNIVARIATE,
            data_as_file=False
        )
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            dmgr = self._prepare_dmgr(tmp_path, training_type=["supervised"], dimensionality=["multivariate"])
            timeeval = TimeEval(dmgr, [("test", "dataset-supervised-multivariate")], [algo],
                                repetitions=1,
                                results_path=tmp_path,
                                skip_invalid_combinations=False)
            timeeval.run()
        results = timeeval.get_results(aggregated=False)

        self.assertEqual(results.loc[0, "status"], Status.ERROR)
        self.assertIn("input dimensionality", results.loc[0, "error_message"])
        self.assertIn("incompatible", results.loc[0, "error_message"])

    def test_missing_training_dataset_timeeval(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.dmgr, [("test", "dataset-int")], self.algorithms,
                                repetitions=1,
                                results_path=Path(tmp_path),
                                skip_invalid_combinations=False)
            timeeval.run()
        results = timeeval.get_results(aggregated=False)

        self.assertEqual(results.loc[0, "status"], Status.ERROR)
        self.assertIn("training dataset", results.loc[0, "error_message"])
        self.assertIn("not found", results.loc[0, "error_message"])

    def test_missing_training_dataset_experiment(self):
        exp = Experiment(
            dataset=Dataset(
                datasetId=("test", "dataset-datetime"),
                dataset_type="synthetic",
                training_type=TrainingType.SUPERVISED,
                num_anomalies=1,
                dimensions=1,
                length=3000,
                contamination=0.0002777777777777778,
                min_anomaly_length=1,
                median_anomaly_length=1,
                max_anomaly_length=1,
                period_size=None
            ),
            algorithm=self.algorithms[0],
            params={},
            params_id=hash_dict({}),
            repetition=0,
            base_results_dir=Path("tmp_path"),
            resource_constraints=ResourceConstraints(),
            metrics=DefaultMetrics.default_list(),
            resolved_test_dataset_path=self.dmgr.get_dataset_path(("test", "dataset-datetime")),
            resolved_train_dataset_path=None
        )
        with self.assertRaises(ValueError) as e:
            exp._perform_training()
        self.assertIn("No training dataset", str(e.exception))

    def test_dont_skip_invalid_combinations(self):
        datasets = [self.dmgr.get(d) for d in self.dmgr.select()]
        exps = Experiments(
            dmgr=self.dmgr,
            datasets=datasets,
            algorithms=self.algorithms,
            metrics=DefaultMetrics.default_list(),
            base_result_path=Path("tmp_path"),
            skip_invalid_combinations=False
        )
        self.assertEqual(len(exps), len(datasets) * len(self.algorithms))

    def test_skip_invalid_combinations(self):
        datasets = [self.dmgr.get(d) for d in self.dmgr.select()]
        exps = Experiments(
            dmgr=self.dmgr,
            datasets=datasets,
            algorithms=self.algorithms,
            metrics=DefaultMetrics.default_list(),
            base_result_path=Path("tmp_path"),
            skip_invalid_combinations=True
        )
        self.assertEqual(len(exps), 1)
        exp = list(exps)[0]
        self.assertEqual(exp.dataset.training_type, exp.algorithm.training_type)
        self.assertEqual(exp.dataset.input_dimensionality, exp.algorithm.input_dimensionality)

    def test_force_training_type_match(self):
        algo = Algorithm(
            name="supervised_deviating_from_mean2",
            main=SupervisedDeviatingFromMean(),
            training_type=TrainingType.SUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            data_as_file=False
        )
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            dmgr = self._prepare_dmgr(tmp_path,
                                      training_type=["unsupervised", "semi-supervised", "supervised", "supervised"],
                                      dimensionality=["univariate", "univariate", "univariate", "multivariate"])
            datasets = [dmgr.get(d) for d in dmgr.select()]
            exps = Experiments(
                dmgr=dmgr,
                datasets=datasets,
                algorithms=self.algorithms + [algo],
                metrics=DefaultMetrics.default_list(),
                base_result_path=tmp_path,
                force_training_type_match=True
            )
        self.assertEqual(len(exps), 3)
        exps = list(exps)
        # algo1 and dataset 3
        exp = exps[0]
        self.assertEqual(exp.algorithm.training_type, TrainingType.SUPERVISED)
        self.assertEqual(exp.dataset.training_type, TrainingType.SUPERVISED)
        self.assertEqual(exp.algorithm.input_dimensionality, InputDimensionality.UNIVARIATE)
        self.assertEqual(exp.dataset.input_dimensionality, InputDimensionality.UNIVARIATE)
        # algo2 and dataset 4
        exp = exps[1]
        self.assertEqual(exp.algorithm.training_type, TrainingType.SUPERVISED)
        self.assertEqual(exp.dataset.training_type, TrainingType.SUPERVISED)
        self.assertEqual(exp.algorithm.input_dimensionality, InputDimensionality.MULTIVARIATE)
        self.assertEqual(exp.dataset.input_dimensionality, InputDimensionality.MULTIVARIATE)
        # algo1 and dataset 3
        exp = exps[2]
        self.assertEqual(exp.algorithm.training_type, TrainingType.SUPERVISED)
        self.assertEqual(exp.dataset.training_type, TrainingType.SUPERVISED)
        self.assertEqual(exp.algorithm.input_dimensionality, InputDimensionality.MULTIVARIATE)
        self.assertEqual(exp.dataset.input_dimensionality, InputDimensionality.UNIVARIATE)

    def test_force_dimensionality_match(self):
        algo = Algorithm(
            name="supervised_deviating_from_mean2",
            main=SupervisedDeviatingFromMean(),
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            data_as_file=False
        )
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            dmgr = self._prepare_dmgr(tmp_path,
                                      training_type=["unsupervised", "supervised", "supervised", "unsupervised"],
                                      dimensionality=["univariate", "multivariate", "univariate", "multivariate"])
            datasets = [dmgr.get(d) for d in dmgr.select()]
            exps = Experiments(
                dmgr=dmgr,
                datasets=datasets,
                algorithms=self.algorithms + [algo],
                metrics=DefaultMetrics.default_list(),
                base_result_path=tmp_path,
                force_dimensionality_match=True
            )
        self.assertEqual(len(exps), 3)
        exps = list(exps)
        # algo1 and dataset 2
        exp = exps[0]
        self.assertEqual(exp.algorithm.training_type, TrainingType.SUPERVISED)
        self.assertEqual(exp.dataset.training_type, TrainingType.SUPERVISED)
        self.assertEqual(exp.algorithm.input_dimensionality, InputDimensionality.UNIVARIATE)
        self.assertEqual(exp.dataset.input_dimensionality, InputDimensionality.UNIVARIATE)
        # algo2 and dataset 2
        exp = exps[1]
        self.assertEqual(exp.algorithm.training_type, TrainingType.UNSUPERVISED)
        self.assertEqual(exp.dataset.training_type, TrainingType.SUPERVISED)
        self.assertEqual(exp.algorithm.input_dimensionality, InputDimensionality.MULTIVARIATE)
        self.assertEqual(exp.dataset.input_dimensionality, InputDimensionality.MULTIVARIATE)
        # algo2 and dataset 4
        exp = exps[2]
        self.assertEqual(exp.algorithm.training_type, TrainingType.UNSUPERVISED)
        self.assertEqual(exp.dataset.training_type, TrainingType.UNSUPERVISED)
        self.assertEqual(exp.algorithm.input_dimensionality, InputDimensionality.MULTIVARIATE)
        self.assertEqual(exp.dataset.input_dimensionality, InputDimensionality.MULTIVARIATE)
