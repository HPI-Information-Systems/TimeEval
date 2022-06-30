import abc
import warnings
from typing import Iterable, Callable, List, Tuple

import numpy as np
from prts import ts_precision, ts_recall, ts_fscore
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.utils import column_or_1d, assert_all_finite, check_consistent_length


class Metric(abc.ABC):
    def __call__(self, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
        y_true, y_score = self._validate_scores(y_true, y_score, **kwargs)
        return self.score(y_true, y_score)

    def _validate_scores(self, y_true: np.ndarray, y_score: np.ndarray,
                         inf_is_1: bool = True,
                         neginf_is_0: bool = True,
                         nan_is_0: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        y_true = np.array(y_true).copy()
        y_score = np.array(y_score).copy()
        # check labels
        if y_true.dtype == np.float_ and y_score.dtype == np.int_:
            warnings.warn("Assuming that y_true and y_score where permuted, because their dtypes indicate so. "
                          "y_true should be an integer array and y_score a float array!")
            return self._validate_scores(y_score, y_true)

        y_true: np.ndarray = column_or_1d(y_true)  # type: ignore
        assert_all_finite(y_true)

        # check scores
        y_score: np.ndarray = column_or_1d(y_score)  # type: ignore

        check_consistent_length([y_true, y_score])
        if not self.supports_continuous_scorings():
            if y_score.dtype != np.int_:
                raise ValueError("When using Metrics other than AUC-metric that need discrete (0 or 1) scores (like "
                                 "Precision, Recall or F1-Score), the scores must be integers and should only contain"
                                 "the values {0, 1}. Please consider applying a threshold to the scores!")
        else:
            if y_score.dtype != np.float_:
                raise ValueError("When using continuous scoring metrics, the scores must be floats!")

        # substitute NaNs and Infs
        nan_mask = np.isnan(y_score)
        inf_mask = np.isinf(y_score)
        neginf_mask = np.isneginf(y_score)
        penalize_mask = np.full_like(y_score, dtype=bool, fill_value=False)
        if inf_is_1:
            y_score[inf_mask] = 1
        else:
            penalize_mask = penalize_mask | inf_mask
        if neginf_is_0:
            y_score[neginf_mask] = 0
        else:
            penalize_mask = penalize_mask | neginf_mask
        if nan_is_0:
            y_score[nan_mask] = 0.
        else:
            penalize_mask = penalize_mask | nan_mask
        y_score[penalize_mask] = (~np.array(y_true[penalize_mask], dtype=bool)).astype(np.int_)

        assert_all_finite(y_score)
        return y_true, y_score

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        ...

    @abc.abstractmethod
    def supports_continuous_scorings(self) -> bool:
        ...


def _auc_metric(y_true: np.ndarray, y_score: Iterable[float], _curve_function: Callable,
                plot: bool = False,
                plot_store: bool = False) -> float:
    x, y, thresholds = _curve_function(y_true, y_score)
    if "precision_recall" in _curve_function.__name__:
        # swap x and y
        x, y = y, x
    area = auc(x, y)
    if plot:
        import matplotlib.pyplot as plt

        name = _curve_function.__name__
        plt.plot(x, y, label=name, drawstyle="steps-post")
        # plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.title(f"{name} | area = {area:.4f}")
        if plot_store:
            plt.savefig(f"fig-{name}.pdf")
        plt.show()
    return area


class RocAUC(Metric):
    def __init__(self, plot: bool = False, plot_store: bool = False):
        self._plot = plot
        self._plot_store = plot_store

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return _auc_metric(y_true, y_score, roc_curve, plot=self._plot, plot_store=self._plot_store)

    def supports_continuous_scorings(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ROC_AUC"


class PrAUC(Metric):
    """Computes the area under the precision recall curve.
    """

    def __init__(self, plot: bool = False, plot_store: bool = False):
        self._plot = plot
        self._plot_store = plot_store

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return _auc_metric(y_true, y_score, precision_recall_curve, plot=self._plot, plot_store=self._plot_store)

    def supports_continuous_scorings(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "PR_AUC"


class RangePrAUC(Metric):
    """Computes the area under the precision recall curve when using the range-based precision and range-based
    recall metric introduced by Tatbul et al. at NeurIPS 2018 [1]_.

    Parameters
    ----------
    max_samples: int
        TimeEval uses a community implementation of the range-based precision and recall metrics, which is quite slow.
        To prevent long runtimes caused by scorings with high precision (many thresholds), just a specific amount of
        possible thresholds is sampled. This parameter controls the maximum number of thresholds; too low numbers
        degrade the metrics' quality.
    r_alpha : float
        Weight of the existence reward for the range-based recall.
    p_alpha : float
        Weight of the existence reward for the range-based precision. For most - when not all - cases, `p_alpha`
        should be set to 0.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    plot : bool
    plot_store : bool

    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. "Precision and Recall for
       Time Series." In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018. http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """

    def __init__(self,
                 max_samples: int = 100,
                 r_alpha: float = 0.5,
                 p_alpha: float = 0,
                 cardinality: str = "reciprocal",
                 bias: str = "flat",
                 plot: bool = False,
                 plot_store: bool = False):
        self._plot = plot
        self._plot_store = plot_store
        self._max_samples = max_samples
        self._r_alpha = r_alpha
        self._p_alpha = p_alpha
        self._cardinality = cardinality
        self._bias = bias

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return _auc_metric(y_true, y_score, self._range_precision_recall_curve,
                           plot=self._plot,
                           plot_store=self._plot_store)

    def _range_precision_recall_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        thresholds = np.unique(y_score)
        thresholds.sort()
        # The first precision and recall values are precision=class balance and recall=1.0, which corresponds to a
        # classifier that always predicts the positive class, independently of the threshold. This means that we can
        # skip the first threshold!
        p0 = y_true.sum() / len(y_true)
        r0 = 1.0
        thresholds = thresholds[1:]

        # sample thresholds
        n_thresholds = thresholds.shape[0]
        if n_thresholds > self._max_samples:
            every_nth = n_thresholds // (self._max_samples - 1)
            sampled_thresholds = thresholds[::every_nth]
            if thresholds[-1] == sampled_thresholds[-1]:
                thresholds = sampled_thresholds
            else:
                thresholds = np.r_[sampled_thresholds, thresholds[-1]]

        recalls = np.zeros_like(thresholds)
        precisions = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            y_pred = (y_score >= threshold).astype(np.int64)
            recalls[i] = ts_recall(y_true, y_pred,
                                   alpha=self._r_alpha,
                                   cardinality=self._cardinality,
                                   bias=self._bias)
            precisions[i] = ts_precision(y_true, y_pred,
                                         alpha=self._p_alpha,
                                         cardinality=self._cardinality,
                                         bias=self._bias)
        sorted_idx = recalls.argsort()[::-1]
        return np.r_[p0, precisions[sorted_idx], 1], np.r_[r0, recalls[sorted_idx], 0], thresholds

    def supports_continuous_scorings(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "RANGE_PR_AUC"


class AveragePrecision(Metric):
    """Computes the average precision metric aver all possible thresholds.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments that get passed down to :func:`sklearn.metrics._ranking.average_precision_score`

    See Also
    --------
    sklearn.metrics._ranking.average_precision_score : Implementation of the average precision metric.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return average_precision_score(y_true, y_score, pos_label=1, **self._kwargs)

    def supports_continuous_scorings(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "AVERAGE_PRECISION"


class RangePrecision(Metric):
    """Computes the range-based precision metric introduced by Tatbul et al. at NeurIPS 2018 [1]_.

    Parameters
    ----------
    alpha : float
        Weight of the existence reward. For most - when not all - cases, `p_alpha` should be set to 0.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.

    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. "Precision and Recall for
       Time Series." In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018. http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """

    def __init__(self, alpha: float = 0, cardinality: str = "reciprocal", bias: str = "flat"):
        self._alpha = alpha
        self._cardinality = cardinality
        self._bias = bias

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return ts_precision(y_true, y_score, alpha=self._alpha, cardinality=self._cardinality, bias=self._bias)

    def supports_continuous_scorings(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "RANGE_PRECISION"


class RangeRecall(Metric):
    """Computes the range-based recall metric introduced by Tatbul et al. at NeurIPS 2018 [1]_.

    Parameters
    ----------
    alpha : float
        Weight of the existence reward. If 0: no existence reward, if 1: only existence reward.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.

    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. "Precision and Recall for
       Time Series." In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018. http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """

    def __init__(self, alpha: float = 0, cardinality: str = "reciprocal", bias: str = "flat"):
        self._alpha = alpha
        self._cardinality = cardinality
        self._bias = bias

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return ts_recall(y_true, y_score, alpha=self._alpha, cardinality=self._cardinality, bias=self._bias)

    def supports_continuous_scorings(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "RANGE_RECALL"


class RangeFScore(Metric):
    """Computes the range-based F-score using the recall and precision metrics by Tatbul et al. at NeurIPS 2018 [1]_.

    The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its
    worst value at 0. This implementation uses the range-based precision and range-based recall as basis.

    Parameters
    ----------
    beta : float
        F-score beta determines the weight of recall in the combined score.
        beta < 1 lends more weight to precision, while beta > 1 favors recall.
    p_alpha : float
        Weight of the existence reward for the range-based precision. For most - when not all - cases, `p_alpha`
        should be set to 0.
    r_alpha : float
        Weight of the existence reward. If 0: no existence reward, if 1: only existence reward.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    p_bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    r_bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.

    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. "Precision and Recall for
       Time Series." In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018. http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """

    def __init__(self,
                 beta: float = 1,
                 p_alpha: float = 0,
                 r_alpha: float = 0.5,
                 cardinality: str = "reciprocal",
                 p_bias: str = "flat",
                 r_bias: str = "flat"):
        self._beta = beta
        self._p_alpha = p_alpha
        self._r_alpha = r_alpha
        self._cardinality = cardinality
        self._p_bias = p_bias
        self._r_bias = r_bias

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return ts_fscore(y_true, y_score,
                         beta=self._beta,
                         p_alpha=self._p_alpha, r_alpha=self._r_alpha,
                         cardinality=self._cardinality,
                         p_bias=self._p_bias, r_bias=self._p_bias)

    def supports_continuous_scorings(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return f"RANGE_F{self._beta:.1f}_SCORE"


class DefaultMetrics:
    ROC_AUC = RocAUC()
    PR_AUC = PrAUC()
    RANGE_PR_AUC = RangePrAUC(max_samples=50, r_alpha=0, cardinality="one", bias="flat")
    AVERAGE_PRECISION = AveragePrecision()
    RANGE_PRECISION = RangePrecision()
    RANGE_RECALL = RangeRecall()
    RANGE_F1 = RangeFScore(beta=1)
    FIXED_RANGE_PR_AUC = RangePrAUC()

    @staticmethod
    def default() -> Metric:
        return DefaultMetrics.ROC_AUC

    @staticmethod
    def default_list() -> List[Metric]:
        return [DefaultMetrics.ROC_AUC]
