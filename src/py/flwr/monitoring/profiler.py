import datetime
import os
import pickle
import platform
import socket
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path, PurePath
from subprocess import check_output
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
    all_proc_mem_used_mb: List[FloatSample] = field(default_factory=list)


@dataclass
class SimpleCPUProcess:
    this_proc_mem_used_mb: List[FloatSample] = field(default_factory=list)
    utilization: List[FloatSample] = field(default_factory=list)


@dataclass
class SimpleGPU:
    uuid: str
    gpu_id: int
    name: str
    total_mem_mb: float = 0.0
    utilization: List[FloatSample] = field(default_factory=list)
    all_proc_mem_used_mb: List[FloatSample] = field(default_factory=list)


@dataclass
class SimpleGPUProcess:
    this_proc_mem_used_mb: List[FloatSample] = field(default_factory=list)


@dataclass
class Node:
    id: str
    list_gpus: Dict[str, SimpleGPU] = field(default_factory=dict)


@dataclass(eq=True, frozen=False)
class Task:
    id: str
    pid: int
    task_name: str
    start_time_ns: int = 0
    stop_time_ns: int = 0
    cpu_process: SimpleCPUProcess = SimpleCPUProcess()
    gpu_processes: Dict[str, SimpleGPUProcess] = field(default_factory=dict)


class SystemMonitor(Thread):
    """Class used for monitoring system usage utilization.

    It keeps track of CPU and GPU utilization, and both RAM and VRAM
    usage (each gpu separately) by pinging for information every
    `interval_s` seconds in a separate thread.
    """

    def __init__(
        self,
        *,
        node_id: Optional[str] = None,
        interval_s: float = 1.0,
        save_path_root: Optional[Path] = None,
    ):
        super(SystemMonitor, self).__init__()
        self.node_id = node_id if node_id else socket.getfqdn()
        self.tasks: Dict[str, Task] = dict()
        self.cpu: SimpleCPU = SimpleCPU()
        self.gpus: Dict[str, SimpleGPU] = dict()
        self.start_time_ns: int = 0
        self.stop_time_ns: int = 0
        self.stopped: bool = True
        self.interval_s: float = interval_s
        self._lock: Lock = Lock()
        self.save_path_root: Path = (
            save_path_root if save_path_root else Path(Path.home() / "flwr_monitor")
        )
        self.active_task_ids: List["str"] = []
        self._collect_resources()

    def get_resources(self) -> Dict[str, Union[SimpleCPU, SimpleGPU]]:
        return {**self.gpus, "cpu": self.cpu}

    def get_tasks(self) -> Dict[str, Task]:
        return self.tasks

    def get_active_tasks(self) -> Dict[str, Task]:
        return {k: v for k, v in self.tasks.items() if k in self.active_task_ids}

    def get_active_tasks_ids(self) -> List[str]:
        return self.active_task_ids

    def is_running(self) -> bool:
        return not self.stopped

    def register_tasks(self, tasks: List[Tuple[str, int, str]]) -> None:
        """Include list of tasks in set of tasks that are being monitored.

        Args:
            tasks (List[Tuple[str, int, str, List[int]]]): List of (task_id,pid,task_name, gpu_ids)s to be included
        """
        with self._lock:
            for task in tasks:
                task_id, pid, task_name = task

                self.tasks[task_id] = Task(
                    id=task_id,
                    pid=pid,
                    task_name=task_name,
                    start_time_ns=time.time_ns(),
                )
                self.active_task_ids.append(task_id)

    def unregister_tasks(self, task_ids: List[str]) -> None:
        """Removes list of tasks in set of tasks that are being monitored.

        Args:
            tasks (List[Tuple[str, int, str]]): List of (task_id,pid,task_name)s to be removed
        """
        with self._lock:
            for task_id in task_ids:
                self.tasks[task_id].stop_time_ns = time.time_ns()
                self.active_task_ids.pop(self.active_task_ids.index(task_id))

    def save_and_clear(self, sub_folder: PurePath):

        save_path = self.save_path_root / sub_folder / self.node_id
        resources_folder = save_path / "resources"
        resources_folder.mkdir(parents=True, exist_ok=True)
        task_folder = save_path / "tasks"
        task_folder.mkdir(parents=True, exist_ok=True)

        # Save Resources
        for hw in ["cpu", "gpus"]:
            filename = resources_folder / f"{hw}.pickle"
            with open(filename, "wb") as handle:
                pickle.dump(getattr(self, hw), handle)

        # Clear CPU and GPU
        del self.cpu.all_proc_mem_used_mb[:]
        for k in self.gpus.keys():
            del self.gpus[k].all_proc_mem_used_mb[:]
            del self.gpus[k].utilization[:]

        # Save Tasks
        for task_id, task in self.tasks.items():
            filename = task_folder / f"{task_id}.pickle"
            with open(filename, "wb") as handle:
                pickle.dump(task, handle)

        # Release memory from Tasks
        self.tasks.clear()

    def _collect_resources(self) -> None:
        # Retrieve GPU info
        all_gpus = nvsmi.get_gpus()
        for gpu in all_gpus:
            self.gpus[gpu.uuid] = SimpleGPU(uuid=gpu.uuid, gpu_id=gpu.id, name=gpu.name)
            self.gpus[gpu.uuid].total_mem_mb = gpu.mem_total

        # Retrieve CPU and system RAM info
        cpu_name = platform.processor()
        self.cpu = SimpleCPU(name=cpu_name, total_mem_mb=psutil.virtual_memory().total)

    def _safe_copy_task_ids_and_pids(self) -> List[Tuple[str, int]]:
        """Returns temporary copy of tasks to be monitored.

        Returns:
            List[Task]: List of tasks to be tracked
        """
        with self._lock:
            return [(task.id, task.pid) for task in self.tasks.values()]

    def run(self) -> None:
        """Runs thread and sleeps for self.interval_s seconds."""
        self.start_time_ns = time.time_ns()
        self.stopped = False
        while not self.stopped:
            if self.tasks:
                self._collect_system_usage()
            time.sleep(self.interval_s)  # Sleep is managed by collect_cpu
        self.stop_time_ns = time.time_ns()

    def stop(self) -> None:
        """Stops thread."""
        self.stopped = True
        self.stop_time_ns = time.time_ns()

    def _collect_gpu_usage(self) -> None:
        # Need to get PID of a task, same order guaranteed in Python 3.7
        task_id_pid_map = {task.pid: task.id for task in self.tasks.values()}

        # Retrieve single process GPU memory usage
        timestamp = time.time_ns()

        pros = nvsmi.get_gpu_processes()
        for pro in pros:
            if pro.pid in task_id_pid_map.keys():
                uuid = pro.gpu_uuid
                task_id = task_id_pid_map[pro.pid]
                if uuid not in self.tasks[task_id].gpu_processes.keys():
                    self.tasks[task_id].gpu_processes[uuid] = SimpleGPUProcess()

                self.tasks[task_id].gpu_processes[uuid].this_proc_mem_used_mb.append(
                    (timestamp, pro.used_memory)
                )

        # Retrieve GPU total memory utilization
        gpus_all = nvsmi.get_gpus()
        timestamp = time.time_ns()
        for gpu in gpus_all:
            uuid: str = gpu.uuid
            if uuid in self.gpus.keys():
                self.gpus[uuid].all_proc_mem_used_mb.append((timestamp, gpu.mem_used))
                self.gpus[uuid].utilization.append((timestamp, gpu.gpu_util))

    def _get_cpu_process_utilization(self) -> None:
        timestamp = time.time_ns()
        # Tracked processed
        task_map = {task.pid: task.id for task in self.tasks.values()}
        pid_list = ",".join([str(x) for x in task_map.keys()])
        try:
            output = check_output(
                ["ps", "-p", pid_list, "--no-headers", "-o", "pid,%mem,%cpu"]
            )
            for line in output.splitlines():
                pid, mem, cpu_percent = line.split()
                self.tasks[task_map[int(pid)]].cpu_process.utilization.append(
                    (timestamp, float(cpu_percent))
                )
                self.tasks[task_map[int(pid)]].cpu_process.this_proc_mem_used_mb.append(
                    (timestamp, float(mem))
                )
        except:
            pass

    def _collect_cpu_usage(self) -> None:
        if psutil is not None:
            # System Memory Utilization
            timestamp = time.time_ns()
            cpu_mem_used = psutil.virtual_memory().used
            self.cpu.all_proc_mem_used_mb.append((timestamp, cpu_mem_used))
        self._get_cpu_process_utilization()

    def _collect_system_usage(self) -> None:
        with self._lock:
            self._collect_cpu_usage()
            self._collect_gpu_usage()

    def aggregate_statistics(self, task_ids: Optional[List[str]]) -> Dict[str, Scalar]:
        # System-wise
        metrics = {}
        stop_time_ns = self.stop_time_ns if self.stop_time_ns > 0 else time.time_ns()
        metrics["round_duration"] = stop_time_ns - self.start_time_ns

        # Max GPU memory across all clients for all GPUs
        max_this_proc_mem_used_mb: Dict[
            str, Dict[str, float]
        ] = {}  # task_id: {uuid:mem_mb}
        selected_task_ids = task_ids if task_ids else [k for k in self.tasks.keys()]
        for task_id in selected_task_ids:
            task = self.tasks[task_id]
            print(task)
            max_this_proc_mem_used_mb[task_id] = {}
            for uuid, gpu_process in task.gpu_processes.items():
                this_task_uuid_mem_usage_mb = [
                    x[1] for x in gpu_process.this_proc_mem_used_mb
                ]
                this_max: float = max(this_task_uuid_mem_usage_mb)
                max_this_proc_mem_used_mb[task_id][uuid] = this_max

        metrics["max_this_proc_mem_used_mb"] = max_this_proc_mem_used_mb

        # Max GPU Memory Used for all process
        max_all_proc_mem_used_mb: Dict[str, float] = {}
        for gpu_uuid, gpu in self.gpus.items():
            mem_values = [x[1] for x in gpu.all_proc_mem_used_mb]
            max_all_proc_mem_used_mb[gpu_uuid] = max(mem_values)
        metrics["max_all_proc_mem_used_mb"] = max_all_proc_mem_used_mb

        # Training Times per task
        training_times_ns: Dict[str, int] = {}  # task_id:
        for task_id in selected_task_ids:
            task = self.tasks[task_id]
            training_times_ns[task_id] = task.stop_time_ns - task.start_time_ns

        metrics["training_times_ns"] = training_times_ns

        # CPU
        """metrics["cpu_all_procs_mem_used_mb"] = max(
            [x[1] for x in self.cpu]
        )
        for resource in self.gpus.values():
            gpu_id = resource.gpu_id
            metrics[f"gpu{gpu_id}_max_utilization"] = max(
                [x[1] for x in resource.utilization]
            )
            metrics[f"gpu{gpu_id}_all_process_mem_used_mb"] = max(
                [x[1] for x in resource.all_procs_mem_used_mb]
            )
            metrics[f"gpu{gpu_id}_total_mem_mb"] = resource.total_mem_mb
            """

        return metrics


def basic_profiler(interval_s: float = 0.1, save_path=None):
    def numpy_profiler(_func: ProfFunDec) -> ProfFunDec:
        @wraps(_func)
        def wrapper(
            *args, **kwargs
        ) -> Tuple[Union[NDArrays, float], int, Dict[str, Scalar]]:
            this_task_id = _func.__name__
            list_tasks = [(this_task_id, os.getpid(), _func.__name__)]
            system_monitor = SystemMonitor(interval_s=interval_s, save_path=None)
            system_monitor.register_tasks(tasks=list_tasks)
            system_monitor.start()
            output, num_examples, metrics = _func(*args, **kwargs)
            system_monitor.unregister_tasks(task_ids=[this_task_id])
            system_monitor.stop()
            return output, num_examples, metrics

        return cast(ProfFunDec, wrapper)

    return numpy_profiler
