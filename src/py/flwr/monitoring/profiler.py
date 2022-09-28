import logging
import os
import platform
import socket
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Thread
from typing import Callable, Dict, List, Tuple, TypeVar, cast

import numpy as np
import nvsmi
import psutil

from flwr.common import NDArrays, Scalar

logger = logging.getLogger(__name__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

FloatSample = Tuple[int, float]

ProfFitFunDec = TypeVar(
    "ProfFitFunDec",
    bound=Callable[
        [NDArrays, Dict[str, Scalar]], Tuple[NDArrays, int, Dict[str, Scalar]]
    ],
)


@dataclass
class SimpleGPUProcess:
    uuid: str
    mem_total: float = 0.0
    mem_total_used: List[FloatSample] = field(default_factory=list)
    mem_proc_used: List[FloatSample] = field(default_factory=list)
    utilization: List[FloatSample] = field(default_factory=list)


class SystemMonitor(Thread):
    """Class for system usage utilization monitoring.

    It keeps track of CPU and GPU utilization, and both RAM and VRAM
    usage (each gpu separately) by pinging for information every
    `interval` seconds in a separate thread.
    """

    def __init__(self, interval: float = 0.7):
        super(SystemMonitor, self).__init__()
        self.fqdn = socket.getfqdn().replace(".", "|")
        self.pid: int = os.getpid()
        self.gpus: Dict[str, SimpleGPUProcess] = dict()
        self.cpu_name: str = platform.processor()
        self.cpu_samples: List[FloatSample] = []
        self.start_time_ns: int = 0
        self.stop_time_ns: int = 0
        self.stopped: bool = False
        self.interval = interval
        self.values = defaultdict(list)

    def run(self):
        self.start_time_ns = time.time_ns()
        while not self.stopped:
            self._read_utilization()
            time.sleep(self.interval)

    def stop(self):
        self.stop_time_ns = time.time_ns()
        self.stopped = True

    def _get_gpu_stats(self):
        if nvsmi is not None:
            # Retrieve GPU process specific info
            pros = nvsmi.get_gpu_processes()
            timestamp = time.time_ns()
            for pro in pros:
                if pro.pid == self.pid:
                    uuid = pro.gpu_uuid
                    if uuid not in self.gpus:
                        self.gpus[uuid] = SimpleGPUProcess(uuid)
                    self.gpus[uuid].mem_proc_used.append((timestamp, pro.used_memory))

            # Retrieve GPU general info
            gpus_all = nvsmi.get_gpus()
            timestamp = time.time_ns()
            for gpu in gpus_all:
                if gpu.uuid in self.gpus.keys():
                    uuid = gpu.uuid
                    self.gpus[uuid].mem_total = gpu.mem_total
                    self.gpus[uuid].mem_total_used.append((timestamp, gpu.mem_used))
                    self.gpus[uuid].utilization.append((timestamp, gpu.gpu_util))

    def _get_cpu_stats(self):
        if psutil is not None:
            timestamp = time.time_ns()
            cpu_percent = psutil.Process(self.pid).cpu_percent(interval=None)
            self.cpu_samples.append((timestamp, cpu_percent))

    def _read_utilization(self):
        self._get_cpu_stats()
        self._get_gpu_stats()

    @staticmethod
    def _get_basic_stats_from_list(prefix: str, values: List[float]):
        stats = dict()
        stats[f"{prefix}.mean"] = np.mean(values)
        stats[f"{prefix}.median"] = np.median(values)
        stats[f"{prefix}.min"] = np.min(values)
        stats[f"{prefix}.max"] = np.max(values)
        return stats

    def aggregate_statistics(self) -> Dict[str, Scalar]:
        stats: Dict[str, Scalar] = {}
        basename = f"_flwr.sys_monitor.{self.fqdn}"
        stats[f"{basename}.duration"] = self.stop_time_ns - self.start_time_ns
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
                elif isinstance(att, float):
                    stats[f"{basename}.{att_name}."] = att
        # CPU
        base_cpu_util = f"{basename}.cpu_info.{self.cpu_name}.utilization"
        values = [v for _, v in self.cpu_samples]
        stats = {**stats, **self._get_basic_stats_from_list(base_cpu_util, values)}

        return stats


def basic_profiler(interval: float = 0.1):
    def basic_profiler(_fit: ProfFitFunDec) -> ProfFitFunDec:
        def wrapper(*args, **kwargs):
            system_monitor = SystemMonitor(interval=interval)
            system_monitor.start()
            parameters, num_examples, metrics = _fit(*args, **kwargs)
            system_monitor.stop()
            stats_dict = system_monitor.aggregate_statistics()
            metrics = {**metrics, **stats_dict}
            return parameters, num_examples, metrics

        return cast(ProfFitFunDec, wrapper)

    return basic_profiler
