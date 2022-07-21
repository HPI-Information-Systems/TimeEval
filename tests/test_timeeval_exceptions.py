import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.fixtures.algorithms import ErroneousAlgorithm
from timeeval import TimeEval, Algorithm, Datasets, DatasetManager, Status
from timeeval.adapters import FunctionAdapter


class TestTimeEvalExceptions(unittest.TestCase):
    def setUp(self) -> None:
        self.datasets_config = Path("./tests/example_data/datasets.json")
        self.datasets: Datasets = DatasetManager("./tests/example_data", custom_datasets_file=self.datasets_config)
        self.identity_algorithm = Algorithm(name="test", main=FunctionAdapter.identity(), data_as_file=False)

    @patch("timeeval.core.experiments.load_dataset")
    def test_wrong_df_shape(self, mock_load):
        df = pd.DataFrame(np.random.rand(10, 2))
        mock_load.side_effect = [df]
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.datasets, [("custom", "dataset.1")], [self.identity_algorithm], results_path=Path(tmp_path))
            timeeval.run()
        self.assertTrue("has a shape that was not expected" in timeeval.results.iloc[0].error_message)

    def test_no_algorithms(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            with self.assertRaises(AssertionError):
                timeeval = TimeEval(self.datasets, [("custom", "dataset.1")], [], results_path=Path(tmp_path))
                timeeval.run()

    def test_evaluation_continues_after_exception_in_algorithm(self):
        ERROR_MESSAGE = "error message test"

        algorithms = [
            Algorithm(name="exception", main=ErroneousAlgorithm(error_message=ERROR_MESSAGE), data_as_file=False),
            self.identity_algorithm
        ]
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.datasets, [("custom", "dataset.1")], algorithms, results_path=Path(tmp_path))
            timeeval.run()

        r = timeeval.results

        self.assertEqual(r[r.algorithm == "exception"].iloc[0].status, Status.ERROR)
        self.assertIn(ERROR_MESSAGE, r[r.algorithm == "exception"].iloc[0].error_message)
