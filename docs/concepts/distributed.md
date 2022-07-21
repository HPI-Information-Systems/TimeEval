# Distributed TimeEval

TimeEval is able to run multiple tests in parallel on a cluster.
It uses {obj}`Dask's SSHCluster <dask.distributed.SSHCluster>` to distribute tasks.
In order to use this feature, the `TimeEval` class accepts a `distributed: bool` flag
and  additional configuration options with `ssh_cluster_kwargs: dict` to setup the {obj}`~dask.distributed.SSHCluster`.
