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
# import GPUtil
from subprocess import Popen, PIPE


logger = logging.getLogger(__name__)

class GPU:
    def __init__(self, ID, uuid, load, memoryTotal, memoryUsed, memoryFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu):
        self.id = ID
        self.uuid = uuid
        self.load = load
        self.memoryUtil = float(memoryUsed)/float(memoryTotal)
        self.memoryTotal = memoryTotal
        self.memoryUsed = memoryUsed
        self.memoryFree = memoryFree
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu

def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number

def get_gpus():
    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen(["nvidia-smi",
                   "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu",
                   "--format=csv,noheader,nounits"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except:
        return []
    output = stdout.decode('UTF-8')
    # output = output[2:-1] # Remove b' and ' from string added by python
    # print(output)
    ## Parse output
    # Split on line break
    lines = output.split(os.linesep)
    # print(lines)
    numDevices = len(lines) - 1
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        # print(line)
        vals = line.split(', ')
        # print(vals)
        for i in range(12):
            # print(vals[i])
            if (i == 0):
                deviceIds = int(vals[i])
            elif (i == 1):
                uuid = vals[i]
            elif (i == 2):
                gpuUtil = safeFloatCast(vals[i]) / 100
            elif (i == 3):
                memTotal = safeFloatCast(vals[i])
            elif (i == 4):
                memUsed = safeFloatCast(vals[i])
            elif (i == 5):
                memFree = safeFloatCast(vals[i])
            elif (i == 6):
                driver = vals[i]
            elif (i == 7):
                gpu_name = vals[i]
            elif (i == 8):
                serial = vals[i]
            elif (i == 9):
                display_active = vals[i]
            elif (i == 10):
                display_mode = vals[i]
            elif (i == 11):
                temp_gpu = safeFloatCast(vals[i]);
        GPUs.append(
            GPU(deviceIds, uuid, gpuUtil, memTotal, memUsed, memFree, driver, gpu_name, serial, display_mode,
                display_active, temp_gpu))
    return GPUs  # (deviceIds, gpuUtil, memUtil)

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
                gpus = get_gpus()
                # gpus = GPUtil.getGPUs()
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

