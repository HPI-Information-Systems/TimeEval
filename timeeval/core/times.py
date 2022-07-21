import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Callable, Tuple, Any, List
import numpy as np

from ..algorithm import Algorithm
from ..data_types import AlgorithmParameter, ExecutionType


@dataclass
class Times:
    execution_type: ExecutionType
    main: float
    preprocess: Optional[float] = None
    postprocess: Optional[float] = None

    @staticmethod
    def result_keys() -> List[str]:
        names = ["train_preprocess_time", "train_main_time",
                 "execute_preprocess_time", "execute_main_time", "execute_postprocess_time"]
        return names

    def to_dict(self) -> Dict:
        dd = asdict(self)
        del dd["execution_type"]
        return {f"{self.execution_type.value}_{k}_time": v for k, v in dd.items()}

    @staticmethod
    def from_execute_algorithm(algorithm: Algorithm, X: AlgorithmParameter, args: dict) -> Tuple[np.ndarray, 'Times']:
        x, pre_time = timer(algorithm.preprocess, X, args) if algorithm.preprocess else (X, np.nan)
        x, main_time = timer(algorithm.execute, x, args)
        x, post_time = timer(algorithm.postprocess, x, args) if algorithm.postprocess else(x, np.nan)
        return x, Times(ExecutionType.EXECUTE, main_time, preprocess=pre_time, postprocess=post_time)

    @staticmethod
    def from_train_algorithm(algorithm: Algorithm, X: AlgorithmParameter, args: dict) -> 'Times':
        x, pre_time = timer(algorithm.preprocess, X, args) if algorithm.preprocess else (X, np.nan)
        x, main_time = timer(algorithm.train, x, args)
        return Times(ExecutionType.TRAIN, main_time, preprocess=pre_time)


def timer(fn: Callable, X: AlgorithmParameter, args: dict) -> Tuple[Any, float]:
    start = time.time()
    fn_result = fn(X, args)
    end = time.time()
    duration = end - start
    return fn_result, duration
