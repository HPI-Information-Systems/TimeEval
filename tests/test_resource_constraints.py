import unittest

import psutil

from timeeval import TimeEval, Datasets
from timeeval.resource_constraints import ResourceConstraints, GB


class TestResourceConstraints(unittest.TestCase):
    def setUp(self) -> None:
        # must reserve 1 GB for OS and other software
        self.usable_memory = psutil.virtual_memory().total - GB
        self.usable_cpus = float(psutil.cpu_count())

    def test_default(self):
        limits = ResourceConstraints()
        mem, cpu = limits.get_resource_limits()

        self.assertEqual(mem, self.usable_memory)
        self.assertEqual(cpu, self.usable_cpus)

    def test_default_from_tasks_per_host(self):
        tasks = 2
        limits = ResourceConstraints(tasks_per_host=tasks)
        mem, cpu = limits.get_resource_limits()

        self.assertEqual(mem, self.usable_memory / tasks)
        self.assertEqual(cpu, self.usable_cpus / tasks)

    def test_explicit_limits(self):
        mem_limit = 1325
        cpu_limit = 1.256
        mem, cpu = ResourceConstraints(
            task_memory_limit=mem_limit
        ).get_resource_limits()
        self.assertEqual(mem, mem_limit)
        self.assertEqual(cpu, self.usable_cpus)

        mem, cpu = ResourceConstraints(
            task_cpu_limit=cpu_limit
        ).get_resource_limits()
        self.assertEqual(mem, self.usable_memory)
        self.assertEqual(cpu, cpu_limit)

    def test_overwrites(self):
        tasks = 2
        mem_overwrite = 1325
        cpu_overwrite = 1.256
        limits = ResourceConstraints(tasks_per_host=tasks, task_memory_limit=12)
        mem, cpu = limits.get_resource_limits(
            memory_overwrite=mem_overwrite,
            cpu_overwrite=cpu_overwrite,
        )

        self.assertEqual(mem, mem_overwrite)
        self.assertEqual(cpu, cpu_overwrite)

    def test_tasks_per_node_overwrite_when_non_distributed(self):
        limits = ResourceConstraints(tasks_per_host=4)

        timeeval = TimeEval(Datasets("./tests/example_data"), [], [], distributed=False, resource_constraints=limits)
        self.assertEqual(1, timeeval.exps.resource_constraints.tasks_per_host)
