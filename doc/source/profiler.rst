Profiler
==========

The profiler obtains estimates of the GPU and CPU memory utilisation for each client performing local training. The measurements serve as an input to the resource scheduling strategy that allocates hardware resources across active clients in the next round of training.

It introduces SystemMonitor, a new class derived from threading.Thread that keeps track of CPU, RAM, GPU and VRAM usage (each GPU separately) by pinging for information every 0.7 seconds in a separate thread. We use the locking mechanism to guarantee the synchronisation of threads. Each client profiling thread is bound to the ID of a particular client and runs in parallel to the local training routine.

To obtain statistics about CPUs on running processes, we make use of psutil (process and system utilities), which is a platform-agnostic package for Python implementing many functions provided by traditional UNIX command line tools.

To fetch statistics about GPUs for a specific process, we use gputil, a Python package that provides an interface to the NVIDIA command line tool for their System Management Interface (nvidia-smi).


Once the information from Profiler is obtained, we can calculate using the following formula:

$$ num_gpus = vram_measured_for_single_worker / total_vram_in_system $$

Then, it can be passed to simulation and set as the GPU resource for each worker along with other client-specific resources.

The original setup results in sequentially executing each client in the round while our profiler help system to significantly reduce the wall-clock time by accurately assigning resources when simulating FL workloads.
