import torch


def parse_ray_resources(cpus: int, vram: int):
    """ Given the amount of VRAM specified for a given experiment,
        figure out what's the corresponding ration in the GPU assigned
        for experiment. Return % of GPU to use. Here we take into account
        that the CUDA runtime allocates ~1GB upon initialization. We therefore
        substract first that amount from the total detected VRAM. CPU resources
        as returned without modification."""

    gpu_ratio = 0.0
    if torch.cuda.is_available():
        # use that figure to get a good estimate of the VRAM needed per experiment
        # (taking into account ~600MB of just CUDA init stuff)

        # Get VRAM of first GPU
        total_vram = torch.cuda.get_device_properties(0).total_memory

        # convert to MB (since the user inputs VRAM in MB)
        total_vram = float(total_vram)/(1024**2)

        # discard 1GB VRAM (which is roughtly what takes for CUDA runtime)
        # You can verify this yourself by just running:
        # `t = torch.randn(10).cuda()` (will use ~1GB VRAM)
        total_vram -= 1024

        gpu_ratio = float(vram)/total_vram
        print(f"GPU percentage per client: {100*gpu_ratio:.2f} % ({vram}/{total_vram})")

        # ! Limitation: this won't work well if multiple GPUs with different VRAMs are detected by Ray
        # The code above asumes therefore all GPUs have the same amount of VRAM. The same `gpu_ratio` will
        # be used in GPUs #1, #2, etc (even though there won't be 1GB taken by CUDA runtime)
        # TODO: probably we can do something smarter: run a single training batch and monitor the real memory usage. This remove user's input an no longer requiring the user to specify VRAM (which often takes a few rounds of trial-error)
    else:
        print("No CUDA device found. Disabling GPU usage for Flower clients.")

    # these keys are the ones expected by ray to specify CPU and GPU resources for each
    # Ray Task, representing a client workload.
    return {'num_cpus': cpus, 'num_gpus': gpu_ratio}
