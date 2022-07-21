import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any, Dict

from .resource_constraints import ResourceConstraints


DEFAULT_SCHEDULER_HOST = "localhost"
DEFAULT_DASK_PORT = 8786
DEFAULT_DASK_LOG_FILENAME = "dask.log"


@dataclass
class RemoteConfiguration:
    """This class holds the configuration for distributed TimeEval.

    TimeEval uses a :obj:`dask.distributed.SSHCluster` to distribute the evaluation tasks to multiple compute nodes.
    Please read the Dask documentation carefully and then use the constructor arguments to setup a TimeEval cluster.

    Parameters
    ----------
    scheduler_host : str
        IP address or hostname for the :obj:`distributed.Scheduler`.
        This node will be responsible to coordinate the cluster.
        The scheduler does not perform any evaluations.
    scheduler_port : int
        Port for the scheduler.
    worker_hosts  : List[str]
        List of IP address or hostnames for the :obj:`distributed.Worker`.
        These nodes will execute the evaluation tasks.
    remote_python : str
        Path to the Python-executable.
        If you set up all your nodes in the same way, the default is fine.
    kwargs_overwrites : dict
        Use this option to overwrite any configuration options of the :obj:`~dask.distributed.SSHCluster`.

        .. warning::
            Only use if you know what you are doing!

    dask_logging_file_level : str
        Logging level for the file-based Dask logger.
    dask_logging_console_level : str
        Logging level for the console-based Dask logger.
    dask_logging_filename : str
        Name of the Dask logging file without any parent paths.
        Each node will write its own logging file and TimeEval will automatically postfix the filenames with the hostname and
        place the Dask logging files into the ``results_path``.

    Examples
    --------
    Two-node cluster where the first node hosts the scheduler but also takes part in the evaluation:

    >>> from timeeval import TimeEval, RemoteConfiguration
    >>> config = RemoteConfiguration(scheduler_host="192.168.1.1", worker_hosts=["192.168.1.1", "192.168.1.2"])
    >>> TimeEval(dm=..., datasets=[], algorithms=[], distributed=True, remote_config=config)
    """
    scheduler_host: str = DEFAULT_SCHEDULER_HOST
    scheduler_port: int = DEFAULT_DASK_PORT
    worker_hosts: List[str] = field(default_factory=lambda: [])
    remote_python: str = field(default_factory=lambda: sys.executable)
    kwargs_overwrites: dict = field(default_factory=lambda: {})
    dask_logging_file_level: str = "INFO"
    dask_logging_console_level: str = "INFO"
    dask_logging_filename: str = DEFAULT_DASK_LOG_FILENAME

    def update_logging_path(self, results_path: Path) -> None:
        """Updates the path to the log filename.

        .. warning::
            Internal API!

        .. attention::
            Always call this function before calling
            :func:`~timeeval.remote_configuration.RemoteConfiguration.get_remote_logging_config`!

        :meta private:
        """
        name = self.dask_logging_filename
        path = results_path / name
        self.dask_logging_filename = str(path.resolve())

    def to_ssh_cluster_kwargs(self, limits: ResourceConstraints) -> Dict[str, Any]:
        """Creates the kwargs for the Dask SSHCluster-constructor.

        .. warning::
            Internal API!

        See Also
        --------
        dask.distributed.SSHCluster : Object that receives these options.

        :meta private:
        """
        mem_limit = limits.task_memory_limit or "auto"
        config = {
            "hosts": [self.scheduler_host] + self.worker_hosts,
            # "connect_options": {
            #     "port": self.scheduler_port
            # },
            # https://distributed.dask.org/en/latest/worker.html?highlight=worker_options#distributed.worker.Worker
            "worker_options": {
                "nprocs": limits.tasks_per_host,
                "nthreads": 1,
                "memory_limit": mem_limit,
            },
            # defaults are fine: https://distributed.dask.org/en/latest/scheduling-state.html?highlight=dask.distributed.Scheduler#distributed.scheduler.Scheduler
            # "scheduler_options": {},
            # "worker_module": "distributed.cli.dask_worker",  # default
            "remote_python": self.remote_python
        }
        config.update(self.kwargs_overwrites)
        return config

    def get_remote_logging_config(self) -> Dict[str, Any]:
        """Creates the logging configuration for Dask based on the remote configuration.

        .. warning::
            Internal API!

        :meta private:
        """
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "incremental": False,
            "formatters": {
                "brief": {
                    "format": "%(name)s - %(levelname)s - %(message)s"
                },
                "verbose-file": {
                    "format": "%(asctime)s - %(levelname)s - %(process)d %(name)s - %(message)s"
                }
            },
            "handlers": {
                "stdout": {
                    "level": self.dask_logging_console_level.upper(),
                    "formatter": "brief",
                    "class": "logging.StreamHandler"
                },
                "log_file": {
                    "level": self.dask_logging_file_level.upper(),
                    "formatter": "verbose-file",
                    "filename": self.dask_logging_filename,
                    "class": "logging.FileHandler",
                    "mode": "a"
                }
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["stdout", "log_file"]
            }
        }
