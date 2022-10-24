import argparse
from socket import getfqdn

import ray

from flwr.simulation.ray_monitoring import RaySystemMonitor


def launch_ray_system_monitor(
    *, ip: str, port: int, namespace="raysysmon", interval: float = 0.1
) -> None:
    address = f"ray://{ip}:{port}" if ip and port else "auto"
    with ray.init(address=address, namespace=namespace):
        this_node_id = ray.get_runtime_context().get_node_id()
        ray_monitor = RaySystemMonitor.options(name=f"{this_node_id}", lifetime="detached").remote(interval=interval, node_id=this_node_id)  # type: ignore
        obj_ref = ray_monitor.start.remote()
        ray.get(obj_ref)
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
        "--namespace", type=str, default="raysysmon", help="Monitor namespace."
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Sampling interval in seconds."
    )
    args = parser.parse_args()
    launch_ray_system_monitor(ip=args.ip, port=args.port, interval=args.interval)
