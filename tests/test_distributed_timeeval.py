import os
import socket
import tempfile
import time
import unittest
from itertools import cycle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import pytest
from durations import Duration

from tests.fixtures.algorithms import DeviatingFromMean, DeviatingFromMedian
from tests.fixtures.call_mocks import MockProcess, MockRsync
from tests.fixtures.dask_mocks import MockDaskClient, MockDaskSSHCluster, MockDaskExceptionClient, \
    MockDaskDockerTimeoutExceptionClient
from tests.fixtures.docker_mocks import MockDockerClient, TEST_DOCKER_IMAGE
from timeeval import TimeEval, Algorithm, DatasetManager, RemoteConfiguration, Status
from timeeval.adapters import DockerAdapter
from timeeval.params import FullParameterGrid
from timeeval.utils.hash_dict import hash_dict


class TestDistributedTimeEval(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.read_csv("tests/example_data/results.csv")
        self.algorithms = [
            Algorithm(name="deviating_from_mean", main=DeviatingFromMean()),
            Algorithm(name="deviating_from_median", main=DeviatingFromMedian(),
                      param_config=FullParameterGrid({"test": [np.int64(2), np.int32(4)]}))
        ]
        self.test_config = RemoteConfiguration(
            scheduler_host="localhost",
            worker_hosts=["localhost"],
            kwargs_overwrites={
                # add the project source files to the python paths of the Dask worker processes:
                "worker_options": {"preload": f"\"import sys; sys.path.insert(0, '{os.getcwd()}')\""}
            }
        )

    @pytest.mark.dask
    def test_distributed_results_and_shutdown_cluster(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = DatasetManager("./tests/example_data", custom_datasets_file=datasets_config)

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                distributed=True, results_path=Path(tmp_path),
                                remote_config=self.test_config)
            timeeval.run()

            compare_columns = ["algorithm", "collection", "dataset", "ROC_AUC"]
            results: pd.DataFrame = timeeval.results[compare_columns]
            results = results.groupby(by=["algorithm", "collection", "dataset"])["ROC_AUC"].mean()
            results = pd.DataFrame(results).reset_index()
            pd.testing.assert_frame_equal(
                results,
                self.results.loc[:, compare_columns]
            )

            # wait a bit before testing if all processes have died to allow for slight delay
            time.sleep(0.5)
            processes = [p for p in psutil.process_iter()]
            self.assertFalse(
                any(
                    ("distributed.cli.dask_worker" in c) or ("distributed.cli.dask_scheduler" in c)
                    for p in processes for c in p.cmdline()
                ),
                msg="Not all dask processes were correctly shut down. Still running processes: "
                    f"{[p.cmdline() for p in processes]}"
            )

    @patch("timeeval.core.remote.Popen")
    @patch("timeeval.timeeval.subprocess.call")
    @patch("timeeval.adapters.docker.docker.from_env")
    @patch("timeeval.core.remote.Client")
    @patch("timeeval.core.remote.SSHCluster")
    def test_distributed_phases(self, mock_cluster, mock_client, mock_docker, mock_call, mock_popen):
        mock_client.return_value = MockDaskClient()
        mock_cluster.return_value = MockDaskSSHCluster(workers=2)
        mock_docker.return_value = MockDockerClient(write_scores_file=True)
        rsync = MockRsync()
        mock_call.side_effect = rsync
        mock_popen.return_value = MockProcess()

        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = DatasetManager("./tests/example_data", custom_datasets_file=datasets_config)
        adapter = DockerAdapter("test-image:latest")

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())),
                                [Algorithm(name="docker", main=adapter, data_as_file=True)],
                                distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="test-host", worker_hosts=["test-host2"]),
                                results_path=Path(tmp_path),
                                n_jobs=1)
            timeeval.run()

            self.assertTrue(
                (timeeval.results_path / "docker" / hash_dict({}) / "custom" / "dataset.1" / "1").exists()
            )
            self.assertTrue(timeeval.remote.client.closed)
            self.assertTrue(timeeval.remote.client.did_shutdown)
            target_path = timeeval.results_path  # == "/results/YYYY_mm_dd_hh_mm"
            self.assertListEqual(rsync.params[0], ["rsync", "-a", "test-host2:" + str(target_path) + "/", str(target_path)])

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_phases(self, mock_docker):
        mock_docker.return_value = MockDockerClient(write_scores_file=True)

        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = DatasetManager("./tests/example_data", custom_datasets_file=datasets_config)
        adapter = DockerAdapter("test-image:latest")

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())),
                                [Algorithm(name="docker", main=adapter, data_as_file=True)], results_path=Path(tmp_path))
            timeeval.run()

            self.assertTrue(
                (timeeval.results_path / "docker" / hash_dict({}) / "custom" / "dataset.1" / "1").exists()
            )

    @patch("timeeval.core.remote.Popen")
    @patch("timeeval.timeeval.subprocess.call")
    @patch("timeeval.adapters.docker.docker.from_env")
    @patch("timeeval.core.remote.Client")
    @patch("timeeval.core.remote.SSHCluster")
    def test_aliases_excluded(self, mock_cluster, mock_client, mock_docker, mock_call, mock_popen):
        mock_client.return_value = MockDaskClient()
        mock_cluster.return_value = MockDaskSSHCluster(workers=2)
        mock_docker.return_value = MockDockerClient()
        rsync = MockRsync()
        mock_call.side_effect = rsync
        mock_popen.return_value = MockProcess()

        datasets = DatasetManager("./tests/example_data")

        with tempfile.TemporaryDirectory() as tmp_path:
            hosts = [
                socket.gethostname(), "127.0.0.1", socket.gethostbyname(socket.gethostname()), "test-host"
            ]
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [self.algorithms[0]], distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="localhost", worker_hosts=hosts),
                                results_path=Path(tmp_path),
                                n_jobs=1)
            timeeval.rsync_results()
            self.assertEqual(len(rsync.params), 1)
            self.assertTrue(rsync.params[0], ["rsync", "-a", f"test-host:{tmp_path}/", tmp_path])

    @patch("timeeval.core.remote.Popen")
    @patch("timeeval.timeeval.subprocess.call")
    @patch("timeeval.adapters.docker.docker.from_env")
    @patch("timeeval.core.remote.Client")
    @patch("timeeval.core.remote.SSHCluster")
    def test_catches_future_exception(self, mock_cluster, mock_client, mock_docker, mock_call, mock_popen):
        mock_client.return_value = MockDaskExceptionClient()
        mock_cluster.return_value = MockDaskSSHCluster(workers=2)
        mock_docker.return_value = MockDockerClient()
        mock_call.side_effect = MockRsync()
        mock_popen.return_value = MockProcess()

        datasets = DatasetManager("./tests/example_data")

        with tempfile.TemporaryDirectory() as tmp_path:
            hosts = [
                socket.gethostname(), "127.0.0.1", socket.gethostbyname(socket.gethostname()), "test-host"
            ]
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [self.algorithms[0]], distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="localhost", worker_hosts=hosts),
                                results_path=Path(tmp_path))
            timeeval.run()
            self.assertListEqual(timeeval.results[["status", "error_message"]].values[0].tolist(), [Status.ERROR, "ExceptionForTest('test-exception')"])

    @patch("timeeval.core.remote.Popen")
    @patch("timeeval.timeeval.subprocess.call")
    @patch("timeeval.adapters.docker.docker.from_env")
    @patch("timeeval.core.remote.Client")
    @patch("timeeval.core.remote.SSHCluster")
    def test_catches_future_timeout_exception(self, mock_cluster, mock_client, mock_docker, mock_call, mock_popen):
        mock_client.return_value = MockDaskDockerTimeoutExceptionClient()
        mock_cluster.return_value = MockDaskSSHCluster(workers=2)
        mock_docker.return_value = MockDockerClient()
        mock_call.side_effect = MockRsync()
        mock_popen.return_value = MockProcess()

        datasets = DatasetManager("./tests/example_data")

        with tempfile.TemporaryDirectory() as tmp_path:
            hosts = [
                socket.gethostname(), "127.0.0.1", socket.gethostbyname(socket.gethostname()), "test-host"
            ]
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [self.algorithms[0]], distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="localhost", worker_hosts=hosts),
                                results_path=Path(tmp_path))
            timeeval.run()
            self.assertListEqual(timeeval.results[["status", "error_message"]].values[0].tolist(),
                                 [Status.TIMEOUT, "DockerTimeoutError('test-exception-timeout')"])

    @pytest.mark.dask
    @pytest.mark.docker
    def test_catches_future_exception_dask(self):
        datasets = DatasetManager("./tests/example_data")
        algo = Algorithm(name="docker-test-timeout",
                         main=DockerAdapter(TEST_DOCKER_IMAGE, skip_pull=True, timeout=Duration("5 seconds")),
                         data_as_file=True,
                         param_config=FullParameterGrid({"raise": [True], "sleep": [1]}))

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [algo],
                                distributed=True,
                                remote_config=self.test_config,
                                results_path=Path(tmp_path))
            timeeval.run()
        status = timeeval.results.loc[0, "status"]
        error_message = timeeval.results.loc[0, "error_message"]

        self.assertEqual(status, Status.ERROR)
        self.assertTrue("Please consider log files" in error_message)

    @pytest.mark.dask
    @pytest.mark.docker
    def test_catches_future_timeout_exception_dask(self):
        datasets = DatasetManager("./tests/example_data")
        algo = Algorithm(name="docker-test-timeout",
                         main=DockerAdapter(TEST_DOCKER_IMAGE, skip_pull=True, timeout=Duration("1 seconds")),
                         data_as_file=True)

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [algo],
                                distributed=True,
                                remote_config=self.test_config,
                                results_path=Path(tmp_path))
            timeeval.run()

        status = timeeval.results.loc[0, "status"]
        error_message = timeeval.results.loc[0, "error_message"]
        self.assertEqual(status, Status.TIMEOUT)
        self.assertIn("could not create results after", error_message)

    @pytest.mark.dask
    @pytest.mark.docker
    def test_runs_docker_in_dask(self):
        datasets = DatasetManager("./tests/example_data")
        algo = Algorithm(name="docker-test-timeout",
                         main=DockerAdapter(TEST_DOCKER_IMAGE, skip_pull=True, timeout=Duration("5 seconds")),
                         data_as_file=True,
                         param_config=FullParameterGrid({"sleep": [1]}))

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [algo],
                                distributed=True,
                                remote_config=self.test_config,
                                results_path=Path(tmp_path))
            timeeval.run()

        status = timeeval.results.loc[0, "status"]
        self.assertEqual(status, Status.OK)
