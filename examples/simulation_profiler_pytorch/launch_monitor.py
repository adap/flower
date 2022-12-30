import argparse
from socket import getfqdn

import ray

from flwr.simulation.ray_monitoring import RaySystemMonitor


def launch_ray_system_monitor(
    *, ip: str, port: int, namespace="flwr_experiment", interval_s: float = 0.1
) -> None:
    address = f"ray://{ip}:{port}" if ip and port else "auto"
    address = "auto"
    with ray.init(address=address, namespace=namespace):
        this_node_id = ray.get_runtime_context().get_node_id()
        ray_monitor = RaySystemMonitor.options(
            name=f"{this_node_id}",
            lifetime="detached",
            num_cpus=2,
            num_gpus=0,
            max_concurrency=2,
        ).remote(
            interval_s=interval_s, node_id=this_node_id
        )  # type: ignore
        print(f"Launched RaySystemMonitor on node {getfqdn()} with ID={this_node_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Launch Ray System Monitor.")
    parser.add_argument(
        "--ip", type=str, default="localhost", help="Ray cluster head IP."
    )
    parser.add_argument(
        "--port", type=int, default=10001, help="Ray cluster head port."
    )
    parser.add_argument(
        "--namespace", type=str, default="flwr_experiment", help="Monitor namespace."
    )
    parser.add_argument(
        "--interval", type=float, default=0.2, help="Sampling interval in seconds."
    )
    args = parser.parse_args()
    launch_ray_system_monitor(ip=args.ip, port=args.port, interval_s=args.interval)
