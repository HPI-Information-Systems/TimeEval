from typing import List, Dict
from pathlib import Path


METRICS_CSV = "metrics.csv"
EXECUTION_LOG = "execution.log"
ANOMALY_SCORES_TS = "anomaly_scores.ts"
HYPER_PARAMETERS = "hyper_params.json"
RESULTS_CSV = "results.csv"


class HPI_CLUSTER:
    """Cluster constant for the HPI cluster.

    These constants are applicable only for the HPI infrastructure and might not be useful for you.
    """
    odin01: str = "odin01"
    odin02: str = "odin02"
    odin03: str = "odin03"
    odin04: str = "odin04"
    odin05: str = "odin05"
    odin06: str = "odin06"
    odin07: str = "odin07"
    odin08: str = "odin08"
    odin09: str = "odin09"
    odin10: str = "odin10"
    odin11: str = "odin11"
    odin12: str = "odin12"
    odin13: str = "odin13"
    odin14: str = "odin14"

    odin01_ip: str = "172.20.11.101"
    odin02_ip: str = "172.20.11.102"
    odin03_ip: str = "172.20.11.103"
    odin04_ip: str = "172.20.11.104"
    odin05_ip: str = "172.20.11.105"
    odin06_ip: str = "172.20.11.106"
    odin07_ip: str = "172.20.11.107"
    odin08_ip: str = "172.20.11.108"
    odin09_ip: str = "172.20.11.109"
    odin10_ip: str = "172.20.11.110"
    odin11_ip: str = "172.20.11.111"
    odin12_ip: str = "172.20.11.112"
    odin13_ip: str = "172.20.11.113"
    odin14_ip: str = "172.20.11.114"

    nodes: List[str] = [
        odin01, odin02, odin03, odin04, odin05, odin06, odin07, odin08, odin09, odin10, odin11, odin12, odin13, odin14
    ]
    """All nodes of the homogenous HPI cluster."""

    nodes_ip: List[str] = [
        odin01_ip, odin02_ip, odin03_ip, odin04_ip, odin05_ip, odin06_ip, odin07_ip, odin08_ip, odin09_ip, odin10_ip,
        odin11_ip, odin12_ip, odin13_ip, odin14_ip
    ]
    """All IP addresses of the nodes in the homogenous HPI cluster."""

    BENCHMARK = "benchmark"
    CORRELATION_ANOMALIES = "correlation-anomalies"
    UNIVARIATE_ANOMALY_TEST_CASES = "univariate-anomaly-test-cases"
    MULTIVARIATE_ANOMALY_TEST_CASES = "multivariate-anomaly-test-cases"
    MULTIVARIATE_TEST_CASES = "multivariate-test-cases"
    VARIABLE_LENGTH_TEST_CASES = "variable-length"

    akita_dataset_paths: Dict[str, Path] = {
        BENCHMARK: Path("/home/projects/akita/data/benchmark-data/data-processed"),
        CORRELATION_ANOMALIES: Path("/home/projects/akita/data/correlation-anomalies"),
        VARIABLE_LENGTH_TEST_CASES: Path("/home/projects/akita/data/variable-length"),
        UNIVARIATE_ANOMALY_TEST_CASES: Path("/home/projects/akita/data/univariate-anomaly-test-cases"),
        MULTIVARIATE_ANOMALY_TEST_CASES: Path("/home/projects/akita/data/multivariate-anomaly-test-cases"),
        MULTIVARIATE_TEST_CASES: Path("/home/projects/akita/data/multivariate-test-cases")
    }
    """This dictionary contains the paths to the dataset collection folders."""
