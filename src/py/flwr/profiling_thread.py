import copy
import glob
import inspect
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from threading import Thread
from typing import Dict, List, Union, Type, Callable, Any
from typing import Optional
import numpy as np
import nvsmi
import psutil
import GPUtil

logger = logging.getLogger(__name__)


class SystemMonitor(Thread):
    """Class for system usage utilization monitoring.
    It keeps track of CPU, RAM, GPU, VRAM usage (each gpu separately) by
    pinging for information every x seconds in a separate thread.
    """

    def __init__(self, pid, start=True, interval=0.7):
        self.stopped = True
        self.pid = pid

        super(SystemMonitor, self).__init__()
        self.interval = interval
        self.values = defaultdict(list) # Create a dictionary with an empty list as the default value when accessing non-existing keys
        self.lock = threading.Lock()
        self.daemon = True
        if start:
            self.start()

    def _read_utilization(self):
        with self.lock:
            if psutil is not None:
                self.values["cpu_util_percent"].append(
                    float(psutil.Process(self.pid).cpu_percent(interval=None))
                )
            if nvsmi is not None:
                pros = nvsmi.get_gpu_processes()
                gpus = GPUtil.getGPUs()
                for g in gpus:
                    total_memory = g.memoryTotal
                for pro in pros:
                    if pro.pid == self.pid:
                        self.values["vram_util_percent" + str(pro.gpu_id)].append(
                            float(float(pro.used_memory)/(total_memory-1024))
                        )

    def get_data(self):
        if self.stopped:
            return {}

        with self.lock:
            ret_values = copy.deepcopy(self.values)
            for key, val in self.values.items():
                del val[:]

        res = {k: np.max(v) for k, v in ret_values.items() if len(v) > 0}
        return res

    def run(self):
        self.stopped = False
        while not self.stopped:
            self._read_utilization()
            time.sleep(self.interval)

    def stop(self):
        self.stopped = True

