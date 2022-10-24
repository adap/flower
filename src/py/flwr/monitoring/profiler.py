import os
import platform
import socket
import time
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock, Thread
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import nvsmi
import psutil

from flwr.common import NDArrays, Scalar


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

FloatSample = Tuple[int, float]

ProfFunDec = TypeVar(
    "ProfFunDec",
    bound=Callable[
        [NDArrays, Dict[str, Scalar]],  # parameters, config
        Union[
            Tuple[NDArrays, int, Dict[str, Scalar]],  # parameters, num_samples, metrics
            Tuple[float, int, Dict[str, Scalar]],  # loss, num_samples, metrics
        ],
    ],
)


@dataclass
class SimpleCPU:
    name: str = "cpu"
    total_mem_mb: float = 0.0
    all_procs_mem_used_mb: List[FloatSample] = field(default_factory=list)
    utilization: List[FloatSample] = field(default_factory=list)


@dataclass
class SimpleGPU:
    uuid: str
    gpu_id: int
    name: str
    total_mem_mb: float = 0.0
    all_procs_mem_used_mb: List[FloatSample] = field(default_factory=list)
    utilization: List[FloatSample] = field(default_factory=list)


@dataclass
class SimpleCPUProcess:
    this_proc_mem_used_mb: List[FloatSample] = field(default_factory=list)


@dataclass
class SimpleGPUProcess:
    this_proc_mem_used_mb: List[FloatSample] = field(default_factory=list)


@dataclass(eq=True, frozen=True)
class Task:
    id: str
    pid: int
    task_name: str
    start_ns: int = 0
    stop_ns: int = 0
    processes: Dict[str, Union[SimpleCPUProcess, SimpleGPUProcess]] = field(
        default_factory=dict
    )


class SystemMonitor(Thread):
    """Class used for monitoring system usage utilization.

    It keeps track of CPU and GPU utilization, and both RAM and VRAM
    usage (each gpu separately) by pinging for information every
    `interval` seconds in a separate thread.
    """

    def __init__(self, *, node_id: Optional[str] = None, interval: float = 1.0):
        super(SystemMonitor, self).__init__()
        self.node_id = node_id if node_id else socket.getfqdn()
        self.tasks: Dict[str, Task] = dict()
        # self.cpu_samples: Dict[int, List[FloatSample]] = dict()
        self.resources: Dict[str, Union[SimpleCPU, SimpleGPU]] = dict()
        self.start_time_ns: int = 0
        self.stop_time_ns: int = 0
        self.stopped: bool = False
        self.interval = interval
        self._lock = Lock()
        self._collect_resources()

    def get_resources(self):
        return self.resources

    def get_tasks(self):
        return self.tasks

    def _collect_resources(self) -> None:
        # Retrieve GPU info
        all_gpus = nvsmi.get_gpus()
        for gpu in all_gpus:
            self.resources[gpu.uuid] = SimpleGPU(
                uuid=gpu.uuid, gpu_id=gpu.id, name=gpu.name
            )
            self.resources[gpu.uuid].total_mem_mb = gpu.mem_total

        # Retrieve CPU info
        cpu_name = platform.processor()
        self.resources["CPU"] = SimpleCPU(
            name=cpu_name, total_mem_mb=psutil.virtual_memory().total
        )

    def is_running(self) -> bool:
        return not self.stopped

    def register_tasks(self, tasks: List[Tuple[str, int, str]]) -> None:
        """Include list of tasks in set of tasks that are being monitored.

        Args:
            tasks (List[Tuple[str, int, str]]): List of (task_id,pid,task_name)s to be included
        """
        with self._lock:
            for task in tasks:
                task_id, pid, task_name = task
                self.tasks[task_id] = Task(id=task_id, pid=pid, task_name=task_name)

    def unregister_tasks(self, tasks: List[Tuple[str, int, str]]) -> None:
        """Removes list of tasks in set of tasks that are being monitored.

        Args:
            tasks (List[Tuple[str, int, str]]): List of (task_id,pid,task_name)s to be removed
        """
        with self._lock:
            for task in tasks:
                task_id, _, _ = task
                self.tasks.pop(task_id, None)

    def _safe_copy_task_ids_and_pids(self) -> List[Tuple[str, int]]:
        """Returns temporary copy of tasks to be monitored.

        Returns:
            List[Task]: List of tasks to be tracked
        """
        with self._lock:
            return [(task.id, task.pid) for task in self.tasks.values()]

    def run(self) -> None:
        """Runs thread and sleeps for self.interval seconds."""
        self.start_time_ns = time.time_ns()
        self.stopped = False
        while not self.stopped:
            if self.tasks:
                self._read_utilization()
            time.sleep(self.interval)  # Needed or duplicated?
        self.stop_time_ns = time.time_ns()

    def stop(self) -> None:
        """Stops thread."""
        self.stopped = True

    def _get_gpu_stats(self) -> None:
        tasks_ids = [task.id for task in self.tasks.values()]
        workers_pids = [task.pid for task in self.tasks.values()]

        # Retrieve GPU process specific info
        pros = nvsmi.get_gpu_processes()
        timestamp = time.time_ns()
        for pro in pros:
            try:
                idx = workers_pids.index(pro.pid)
                self.tasks[tasks_ids[idx]].processes[
                    pro.gpu_uuid
                ].this_proc_mem_used_mb.append((timestamp, pro.used_memory))
            except:
                pass
        # Retrieve GPU total and per process memory utilization
        gpus_all = nvsmi.get_gpus()
        timestamp = time.time_ns()
        for gpu in gpus_all:
            if gpu.uuid in self.resources.keys():
                uuid = gpu.uuid
                self.resources[uuid].all_procs_mem_used_mb.append(
                    (timestamp, gpu.mem_used)
                )
                self.resources[uuid].utilization.append((timestamp, gpu.gpu_util))

    def _get_cpu_stats(self) -> None:
        if psutil is not None:
            timestamp = time.time_ns()
            workers_pids = [task.pid for task in self.tasks.values()]
            for pid in workers_pids:
                cpu_percent = psutil.Process(int(pid)).cpu_percent(
                    interval=self.interval
                )
                self.resources["cpu"].utilization.append((timestamp, cpu_percent))

    def _read_utilization(self) -> None:
        # with self._lock:
        #    self._get_cpu_stats()
        #    self._get_gpu_stats()
        pass

    def aggregate_statistics(self) -> Dict[str, Scalar]:
        return {}

    """
    @staticmethod
    def _get_basic_stats_from_list(
        prefix: str, values: List[float]
    ) -> Dict[str, float]:
        stats = dict()
        stats[f"{prefix}.mean"] = np.mean(values)
        stats[f"{prefix}.median"] = np.median(values)
        stats[f"{prefix}.min"] = np.min(values)
        stats[f"{prefix}.max"] = np.max(values)
        return stats

    def aggregate_statistics(self) -> Dict[str, Scalar]:
        stats: Dict[str, Scalar] = {}
        basename = f"_flwr.sys_monitor.{self.node_id}"
        stats[f"{basename}.duration_ns"] = self.stop_time_ns - self.start_time_ns
        # GPUs
        for gpu_uuid, gpu in self.gpus.items():
            for att_name in gpu.__dict__.keys():
                base_gpu_att = f"{basename}.gpu_info.{gpu_uuid}.{att_name}"
                att = getattr(gpu, att_name)
                if isinstance(att, list) and all(isinstance(v, float) for _, v in att):
                    values = [v for _, v in att]
                    stats = {
                        **stats,
                        **self._get_basic_stats_from_list(base_gpu_att, values),
                    }
                if isinstance(att, dict):
                    for pid, l in att.items():
                        base_gpu_att_pid = (
                            f"{basename}.gpu_info.{gpu_uuid}.{att_name}.{pid}"
                        )
                        values = [v for _, v in l]
                        stats = {
                            **stats,
                            **self._get_basic_stats_from_list(base_gpu_att_pid, values),
                        }

                elif isinstance(att, float):
                    stats[f"{basename}.{att_name}."] = att
        # CPU
        for pid, samples in self.cpu_samples.items():
            base_cpu_util = f"{basename}.cpu_info.{self.cpu_name}.utilization.{pid}"
            values = [v for _, v in samples]
            stats = {**stats, **self._get_basic_stats_from_list(base_cpu_util, values)}

        return stats
        """


def basic_profiler(interval: float = 0.1):
    def numpy_profiler(_func: ProfFunDec) -> ProfFunDec:
        @wraps(_func)
        def wrapper(
            *args, **kwargs
        ) -> Tuple[Union[NDArrays, float], int, Dict[str, Scalar]]:
            list_tasks = [(_func.__name__, os.getpid(), _func.__name__)]
            system_monitor = SystemMonitor(interval=interval)
            system_monitor.register_tasks(tasks=list_tasks)
            system_monitor.start()
            output, num_examples, metrics = _func(*args, **kwargs)
            system_monitor.unregister_tasks(tasks=list_tasks)
            system_monitor.stop()
            stats_dict = system_monitor.aggregate_statistics()
            metrics = {**metrics, **stats_dict}
            return output, num_examples, metrics

        return cast(ProfFunDec, wrapper)

    return numpy_profiler
