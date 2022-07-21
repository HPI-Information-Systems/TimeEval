import os
import socket
import tempfile
import time
import unittest
from pathlib import Path

import pytest
from dask.distributed import SSHCluster

from timeeval.core.remote import Remote, RemoteConfiguration


class TestRemote(unittest.TestCase):
    @pytest.mark.dask
    def test_restart_already_running(self):
        already_running_cluster = SSHCluster(hosts=["localhost"])
        with self.assertLogs("Remote", level="WARNING") as logger:
            new_remote = Remote(remote_config=RemoteConfiguration(scheduler_host="localhost"))
            self.assertTrue("restart cluster" in logger.output[0])
            already_running_cluster.close()
            new_remote.close()

    @pytest.mark.dask
    def test_other_than_already_running_exception(self):
        with self.assertRaises(Exception):
            Remote(remote_config=RemoteConfiguration(scheduler_host=None))

    @pytest.mark.dask
    def test_run_on_all_hosts(self):
        def _test_func(*args, **kwargs):
            a = time.time_ns()
            os.mkdir(args[0] / str(a))

        with tempfile.TemporaryDirectory() as tmp_path:
            remote = Remote(
                remote_config=RemoteConfiguration(
                    scheduler_host="localhost",
                    worker_hosts=["localhost", "localhost"]
                ))
            remote.run_on_all_hosts([(_test_func, [Path(tmp_path)], {})])
            remote.close()
            self.assertEqual(len(os.listdir(tmp_path)), 2)

    @pytest.mark.dask
    def test_run_on_all_hosts_ssh(self):
        current_host = socket.gethostname()

        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path).resolve()
            remote = Remote(remote_config=RemoteConfiguration(
                scheduler_host=current_host,
                worker_hosts=[current_host]
            ))
            remote.run_on_all_hosts_ssh(f"touch {tmp_path}/$(hostname)")
            remote.close()
            self.assertListEqual([d.name for d in tmp_path.iterdir()], [current_host])
