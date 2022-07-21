import unittest
from copy import deepcopy
from pathlib import Path

from tests.fixtures.algorithms import SupervisedDeviatingFromMean
from timeeval import Algorithm, TrainingType, InputDimensionality, DefaultMetrics, DatasetManager
from timeeval.core.experiments import Experiments
from timeeval.params import FullParameterGrid


class TestExperiments(unittest.TestCase):

    def setUp(self) -> None:
        self.dmgr = DatasetManager("./tests/example_data")
        self.algorithms = [
            Algorithm(
                name="supervised_deviating_from_mean",
                main=SupervisedDeviatingFromMean(),
                training_type=TrainingType.SUPERVISED,
                input_dimensionality=InputDimensionality.UNIVARIATE,
                data_as_file=False,
                param_config=FullParameterGrid({
                    "param1": range(3),
                    "param2": ["a", "b"]
                })
            )
        ]

    def test_unique_experiment_name(self):
        """
        The experiment names are used as keys for the distributed evaluation tasks and must be unique!
        """
        datasets = [self.dmgr.get(d) for d in self.dmgr.select()]
        exps = Experiments(
            dmgr=self.dmgr,
            datasets=datasets,
            algorithms=self.algorithms,
            repetitions=2,
            metrics=DefaultMetrics.default_list(),
            base_result_path=Path("tmp_path"),
            skip_invalid_combinations=True
        )
        self.assertEqual(len(exps), 12)
        exp_names = [exp.name for exp in exps]
        self.assertEqual(len(exp_names), len(set(exp_names)))

    def test_common_params_id_for_heuristics(self):
        datasets = [self.dmgr.get(d) for d in self.dmgr.select()]
        algorithms = deepcopy(self.algorithms)
        algorithms[0].param_config = FullParameterGrid({
            "param1": range(2),
            "param2": ["heuristic:PeriodSizeHeuristic(factor=1.0, fb_value=100)"]
        })
        exps = Experiments(
            dmgr=self.dmgr,
            datasets=datasets,
            algorithms=algorithms,
            repetitions=1,
            metrics=DefaultMetrics.default_list(),
            base_result_path=Path("tmp_path"),
            skip_invalid_combinations=False
        )

        tasks = [(exp.dataset_name, exp.params) for exp in exps]
        self.assertListEqual(tasks, [
            ("dataset-datetime", {"param1": 0, "param2":  12}),
            ("dataset-int",      {"param1": 0, "param2": 100}),
            ("dataset-datetime", {"param1": 1, "param2":  12}),
            ("dataset-int",      {"param1": 1, "param2": 100})
        ])

        # the parameter IDs for both datasets must be the same despite having different values in param2, because
        # param2 is set by a heuristic depending on the dataset's metadata
        params_ids = [exp.params_id for exp in exps]
        self.assertEqual(params_ids[0], params_ids[1])
        self.assertEqual(params_ids[2], params_ids[3])
