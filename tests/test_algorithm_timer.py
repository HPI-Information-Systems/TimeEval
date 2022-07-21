import time
import unittest

import numpy as np

from timeeval import Algorithm, AlgorithmParameter, TrainingType
from timeeval.adapters import FunctionAdapter
from timeeval.data_types import ExecutionType
from timeeval.core.times import Times


def pre(x: AlgorithmParameter, args) -> AlgorithmParameter:
    time.sleep(0.2)
    return x


def main(x: AlgorithmParameter, args) -> AlgorithmParameter:
    time.sleep(0.3)
    return x


def post(x: AlgorithmParameter, args) -> np.ndarray:
    time.sleep(0.1)
    return x  # type: ignore


class TestAlgorithmTimer(unittest.TestCase):
    def test_algorithm_times(self):
        algorithm = Algorithm(main=FunctionAdapter(main), preprocess=pre, postprocess=post,
                              name="test",
                              training_type=TrainingType.SUPERVISED)
        train_times = Times.from_train_algorithm(algorithm, np.random.rand(10), {})
        self.assertAlmostEqual(train_times.preprocess, 0.2, places=1)
        self.assertAlmostEqual(train_times.main, 0.3, places=1)
        score, times = Times.from_execute_algorithm(algorithm, np.random.rand(10), {})
        self.assertAlmostEqual(times.preprocess, 0.2, places=1)
        self.assertAlmostEqual(times.main, 0.3, places=1)
        self.assertAlmostEqual(times.postprocess, 0.1, places=1)

    def test_times_dict(self):
        pre = 0.2
        main = 0.3
        post = 0.1
        times = Times(execution_type=ExecutionType.EXECUTE, main=main, preprocess=pre, postprocess=post)
        self.assertDictEqual(times.to_dict(), {"execute_preprocess_time": pre,
                                               "execute_main_time": main,
                                               "execute_postprocess_time": post})
