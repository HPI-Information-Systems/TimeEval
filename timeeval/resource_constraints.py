from dataclasses import dataclass
from typing import Optional, Tuple

import psutil
from durations import Duration

MB = 1024 ** 2
""":math:`1 MB = 2^{20} \\text{Bytes}`

Can be used to set the memory limit.

Examples
--------

>>> from timeeval.resource_constraints import ResourceConstraints, MB
>>> ResourceConstraints(task_memory_limit=500 * MB)
"""
GB = 1024 ** 3
""":math:`1 GB = 2^{30} \\text{Bytes}`

Can be used to set the memory limit.

Examples
--------

>>> from timeeval.resource_constraints import ResourceConstraints, GB
>>> ResourceConstraints(task_memory_limit=1 * GB)
"""
DEFAULT_TASKS_PER_HOST = 1
DEFAULT_TIMEOUT = Duration("8 hours")


@dataclass
class ResourceConstraints:
    """Use this class to configure resource constraints and how TimeEval deals with preliminary results.

    .. warning::
        Resource constraints are just supported by the :class:`~timeeval.adapters.docker.DockerAdapter`!

    For docker: Swap is always disabled. Resource constraints are enforced using explicit resource limits on the Docker
    container.

    Parameters
    ----------
    tasks_per_host : int
        Specify, how many evaluation tasks are executed on each host. This setting influences the default memory and
        CPU limits if :attr:`~timeeval.resource_constraints.ResourceConstraints.task_memory_limit` and
        :attr:`~timeeval.resource_constraints.ResourceConstraints.task_cpu_limit` are ``None``: the available
        resources of the node are shared equally between the tasks.

        Because each tasks, in effect, trains or executes a time series anomaly detection algorithm, the tasks are
        resource-intensive, which means that over-provisioning is not useful and could decrease overall performance. If
        runtime measurements are taken, **make sure that no resources are shared between the tasks**!
    task_memory_limit : Optional[int]
        Specify the maximum allowed memory in Bytes. You can use :const:`~timeeval.resource_constraints.MB` and
        :const:`~timeeval.resource_constraints.GB` for better readability. This setting limits the available main
        memory per task to a fixed value.
    task_cpu_limit : Optional[float]
        Specify the maximum allowed CPU usage in fractions of CPUs (e.g. 0.25 means: only use 1/4 of a single
        CPU core). Usually, it is advisable to use whole CPU cores (e.g. 1.0 for 1 CPU core, 2.0 for 2 CPU cores, etc.).
    train_timeout : Duration
        Default timeout for training an algorithm. This value can be overridden for each algorithm in its
        :class:`~timeeval.adapters.docker.DockerAdapter`.
    execute_timeout : Duration
        Default timeout for executing an algorithm. This value can be overridden for each algorithm in its
        :class:`~timeeval.adapters.docker.DockerAdapter`.
    use_preliminary_model_on_train_timeout : bool
        If this option is enabled (default), then algorithms can save preliminary models (model checkpoints) to disk and
        TimeEval will use the last preliminary model if the training step runs into the training timeout. This is
        especially useful for machine learning algorithms that use an iterative training process (e.g. using SGD). As
        long as the algorithm implementation stores the best-so-far model after each training epoch, the training must
        not be limited by the number of epochs but just by the training time.
    use_preliminary_scores_on_execute_timeout : bool
        If this option is enabled (default) and an algorithm exceeds the execution timeout, TimeEval will look for any
        preliminary result. This allows the evaluation of progressive algorithms that output a rough result, refine it
        over time, and would otherwise run into the execution timeout.
    """

    tasks_per_host: int = DEFAULT_TASKS_PER_HOST
    task_memory_limit: Optional[int] = None
    task_cpu_limit: Optional[float] = None
    train_timeout: Duration = DEFAULT_TIMEOUT
    execute_timeout: Duration = DEFAULT_TIMEOUT
    use_preliminary_model_on_train_timeout: bool = True
    use_preliminary_scores_on_execute_timeout: bool = True

    def get_compute_resource_limits(self,
                                    memory_overwrite: Optional[int] = None,
                                    cpu_overwrite: Optional[float] = None) -> Tuple[int, float]:
        """Calculates the resource constraints for a single task.

        There are three sources for resource limits (in decreasing priority):

        1. Overwrites (passed to this function as arguments)
        2. Explicitly set resource limits (on this object using `task_memory_limit` and `task_cpu_limit`)
        3. Default resource constraints

        **Overall default**:

        1 task per node using all available cores and RAM (except small margin for OS).

        When multiple tasks are specified, the resources are equally shared between all concurrent tasks. This means
        that CPU limit is set based on node CPU count divided by the number of tasks and memory limit is set based on
        total memory of node minus 1 GB (for OS) divided by the number of tasks.

        .. attention::
            Must be called on the node that will execute the task!

        Parameters
        ----------
        memory_overwrite: int
            If this is set, it will overwrite the memory limit.
        cpu_overwrite : float
            If this is set, it will overwrite the CPU limit.

        Returns
        -------
        memory_limit, cpu_limit : Tuple[int,float]
            Tuple of memory and CPU limit.
            Memory limit is expressed in Bytes and CPU limit is expressed in fractions of CPUs (e.g. 0.25 means: only
            use 1/4 of a single CPU core).
        """
        if memory_overwrite:
            memory_limit = memory_overwrite
        elif self.task_memory_limit:
            memory_limit = self.task_memory_limit
        else:
            usable_memory = psutil.virtual_memory().total - 1 * GB
            memory_limit = usable_memory // self.tasks_per_host

        if cpu_overwrite:
            cpu_limit = cpu_overwrite
        elif self.task_cpu_limit:
            cpu_limit = self.task_cpu_limit
        else:
            cpus = psutil.cpu_count()
            cpu_limit = cpus / self.tasks_per_host

        return memory_limit, cpu_limit

    def get_train_timeout(self, timeout_overwrite: Optional[Duration] = None) -> Duration:
        """Returns the maximum runtime of a training task in seconds.

        Parameters
        ----------
        timeout_overwrite : Duration
            If this is set, it will overwrite the global timeout.

        Returns
        -------
        train_timeout : Duration
            The training timeout with the highest precedence (method overwrite then global configuration).
        """
        return self._get_timeout_with_overwrite(self.train_timeout, timeout_overwrite)

    def get_execute_timeout(self, timeout_overwrite: Optional[Duration] = None) -> Duration:
        """Returns the maximum runtime of an execution task in seconds.

        Parameters
        ----------
        timeout_overwrite : Duration
            If this is set, it will overwrite the global timeout.

        Returns
        -------
        execute_timeout : Duration
            The execution timeout with the highest precedence (method overwrite then global configuration).
        """
        return self._get_timeout_with_overwrite(self.execute_timeout, timeout_overwrite)

    @staticmethod
    def _get_timeout_with_overwrite(timeout: Duration, overwrite: Optional[Duration]) -> Duration:
        if overwrite is not None:
            timeout = overwrite
        return timeout

    @staticmethod
    def default_constraints() -> 'ResourceConstraints':
        """Creates a configuration object with the default resource constraints."""
        return ResourceConstraints()
