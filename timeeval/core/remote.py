import logging
import time
from asyncio import Future, run_coroutine_threadsafe, get_event_loop
from pathlib import Path
from subprocess import Popen
from typing import List, Callable, Optional, Tuple, Dict

import tqdm
from dask import config as dask_config
from dask.distributed import Client, SSHCluster

from ..remote_configuration import RemoteConfiguration
from ..resource_constraints import ResourceConstraints


class Remote:
    def __init__(self, disable_progress_bar: bool = False,
                 remote_config: RemoteConfiguration = RemoteConfiguration(),
                 resource_constraints: ResourceConstraints = ResourceConstraints()):
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = remote_config
        self.limits = resource_constraints
        self.log.debug(f"Remoting configuration: {self.config}\n"
                       f"with {self.limits.tasks_per_host} tasks per host")
        self.futures: List[Future] = []

        # setup logging of Dask:
        self.log.info("Configuring dask logging")
        dask_config.config["distributed"]["logging"] = self.config.get_remote_logging_config()

        self.log.debug("Creating folder structure for dask logfile")
        logfile_folder = Path(self.config.dask_logging_filename).parent.resolve()
        failed_on = self.run_on_all_hosts_ssh(f"mkdir -p {logfile_folder}")
        if len(failed_on) != 0:
            raise ValueError(f"Could not create target folder for Dask logfile on hosts {', '.join(failed_on)}!")

        self.log.info("Starting Dask SSH cluster ...")
        self.cluster = self.start_or_restart_cluster()
        self.client = Client(self.cluster.scheduler_address)
        self.log.info("... Dask SSH cluster successfully started!")
        self.disable_progress_bar = disable_progress_bar

    def start_or_restart_cluster(self, n: int = 0) -> SSHCluster:
        if n >= 5:
            raise RuntimeError("Could not start an SSHCluster because there is already one running, "
                               "that cannot be stopped!")
        try:
            return SSHCluster(**self.config.to_ssh_cluster_kwargs(self.limits))
        except Exception as e:
            if "Worker failed to start" in str(e):
                self.log.debug(f"Received SSHCluster error message: {e}")
                scheduler_host = self.config.scheduler_host
                port = self.config.scheduler_port
                scheduler_address = f"{scheduler_host}:{port}"

                self.log.warning(f"Failed to start cluster, because address already in use! "
                                 f"Trying to restart cluster at {scheduler_address}!")

                with Client(scheduler_address) as client:
                    client.shutdown()

                return self.start_or_restart_cluster(n + 1)
            raise e

    def add_task(self, task: Callable, *args, config: Optional[dict] = None, **kwargs) -> Future:  # type: ignore[no-untyped-def]
        config = config or {}
        self.log.debug(f"Submitting task {task} to cluster")
        future = self.client.submit(task, *args, **config, **kwargs)
        self.futures.append(future)
        return future  # type: ignore

    def run_on_all_hosts(self, tasks: List[Tuple[Callable, List, Dict]],
                         msg: str = "Executing remote tasks",
                         progress: bool = True) -> None:
        self.log.debug(f"Running {len(tasks)} tasks on all cluster nodes and waiting for results")
        for task, args, kwargs in tqdm.tqdm(tasks, desc=msg, disable=self.disable_progress_bar or not progress):
            self.log.debug(f"({msg}) Running task '{task}' with args {args} and kwargs {kwargs}")
            self.client.run(task, *args, **kwargs)

    def run_on_all_hosts_ssh(self, command: str) -> List[str]:
        failed_on: List[str] = []
        for host in self.config.worker_hosts:
            process = Popen(f"ssh {host} '{command}'", shell=True)
            status = process.wait()
            if status == 0:
                self.log.debug(f"Command '{command}' on host {host} successful!")
            else:
                self.log.error(f"Command '{command}' on host {host} failed!")
                failed_on.append(host)
        return failed_on

    def fetch_results(self) -> None:
        n_experiments = len(self.futures)
        self.log.debug(f"Waiting for the results of {n_experiments} tasks submitted previously to the cluster")
        coroutine_future = run_coroutine_threadsafe(self.client.gather(self.futures, asynchronous=True), get_event_loop())
        progress_bar = tqdm.trange(n_experiments, desc="Evaluating distributedly", position=0, disable=self.disable_progress_bar)

        while not coroutine_future.done():
            n_done = sum([f.done() for f in self.futures])
            progress_bar.update(n_done - progress_bar.n)
            if progress_bar.n == n_experiments:
                break
            time.sleep(0.5)

        progress_bar.update(n_experiments - progress_bar.n)
        progress_bar.close()

    def close(self) -> None:
        self.log.debug("Shutting down Dask SSH cluster and Dask client")
        self.cluster.close()
        self.client.shutdown()
        self.client.close()

        self.log.debug("Renaming dask logfile based on the hostname to allow syncing it!")
        logfile = Path(self.config.dask_logging_filename).resolve()
        command = f"mv {logfile} {logfile.parent}/{logfile.stem}-$(hostname){logfile.suffix}"
        failed_on = self.run_on_all_hosts_ssh(command)
        if len(failed_on) == 0:
            self.log.debug(f"... logfile renaming successful!")
        else:
            self.log.error(f"... could not rename logfile on hosts {', '.join(failed_on)}!")
