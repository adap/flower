import argparse
from socket import getfqdn

import ray
import nvsmi

from flwr.simulation.ray_monitoring import RaySystemMonitor



def launch_ray_system_monitor(
    *, namespace="flwr_experiment", interval_s: float = 0.1
) -> None:
    with ray.init(address="auto", namespace=namespace):
        this_node_id = ray.get_runtime_context().get_node_id()
        ray_monitor = RaySystemMonitor.options(
            name=f"{this_node_id}",
            lifetime="detached",
            num_cpus=4,
            num_gpus=0,
            max_concurrency=4,
        ).remote(
            interval_s=interval_s, node_id=this_node_id,
        )  # type: ignore

        c = ray.get(ray_monitor._collect_resources.remote())
        print(c)
        print(f"Launched RaySystemMonitor on node {getfqdn()} with ID={this_node_id}, {c}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Launch Ray System Monitor.")
    args = parser.parse_args()
    parser.add_argument(
        "--namespace",
        type=str,
        default="flwr_experiment",
        help="Experiment namespace. Used when tracking multiple experiments",
    )
    parser.add_argument(
        "--interval", type=float, default=0.2, help="Sampling interval in seconds."
    )
    args = parser.parse_args()
    launch_ray_system_monitor(namespace=args.namespace, interval_s=args.interval)
