import random
import argparse

parser = argparse.ArgumentParser(description="Generated Docker Compose")
parser.add_argument(
    "--total_clients", type=int, default=2, help="Total clients to spawn (default: 2)"
)
parser.add_argument(
    "--num_rounds", type=int, default=100, help="Number of FL rounds (default: 100)"
)
parser.add_argument(
    "--data_percentage",
    type=float,
    default=0.6,
    help="Portion of client data to use (default: 0.6)",
)
parser.add_argument(
    "--random", action="store_true", help="Randomize client configurations"
)


def create_docker_compose(args):
    # cpus is used to set the number of CPUs available to the container as a fraction of the total number of CPUs on the host machine.
    # mem_limit is used to set the memory limit for the container.
    client_configs = [
        {"mem_limit": "3g", "batch_size": 32, "cpus": 4, "learning_rate": 0.001},
        {"mem_limit": "6g", "batch_size": 256, "cpus": 1, "learning_rate": 0.05},
        {"mem_limit": "4g", "batch_size": 64, "cpus": 3, "learning_rate": 0.02},
        {"mem_limit": "5g", "batch_size": 128, "cpus": 2.5, "learning_rate": 0.09},
        # Add or modify the configurations depending on your host machine
    ]

    docker_compose_content = f"""
version: '3'
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - 9090:9090
    deploy:
      restart_policy:
        condition: on-failure
    command:
      - --config.file=/etc/prometheus/prometheus.yml
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - cadvisor

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.0
    container_name: cadvisor
    privileged: true
    deploy:
      restart_policy:
        condition: on-failure
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
      - /var/run/docker.sock:/var/run/docker.sock  

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - 3000:3000
    deploy:
      restart_policy:
        condition: on-failure
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./config/grafana.ini:/etc/grafana/grafana.ini
      - ./config/provisioning/datasources:/etc/grafana/provisioning/datasources
      - ./config/provisioning/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
      - cadvisor
    command:
      - --config=/etc/grafana/grafana.ini


  server:
    container_name: server
    build:
      context: .
      dockerfile: Dockerfile
    command: python server.py --number_of_rounds={args.num_rounds}
    environment:
      FLASK_RUN_PORT: 6000
      DOCKER_HOST_IP: host.docker.internal
    volumes:
      - .:/app
      - /var/run/docker.sock:/var/run/docker.sock      
    ports:
      - "6000:6000"
      - "8265:8265"
      - "8000:8000"
    depends_on:
      - prometheus
      - grafana
"""
    # Add client services
    for i in range(1, args.total_clients + 1):
        if args.random:
            config = random.choice(client_configs)
        else:
            config = client_configs[(i - 1) % len(client_configs)]
        docker_compose_content += f"""
  client{i}:
    container_name: client{i}
    build:
      context: .
      dockerfile: Dockerfile
    command: python client.py --server_address=server:8080 --data_percentage={args.data_percentage}  --client_id={i} --total_clients={args.total_clients} --batch_size={config["batch_size"]} --learning_rate={config["learning_rate"]}
    deploy:
      resources:
        limits:
          cpus: "{(config['cpus'])}"
          memory: "{config['mem_limit']}"
    volumes:
      - .:/app
      - /var/run/docker.sock:/var/run/docker.sock
    ports:
      - "{6000 + i}:{6000 + i}"
    depends_on:
      - server
    environment:
      FLASK_RUN_PORT: {6000 + i}
      container_name: client{i}
      DOCKER_HOST_IP: host.docker.internal
"""

    docker_compose_content += "volumes:\n  grafana-storage:\n"

    with open("docker-compose.yml", "w") as file:
        file.write(docker_compose_content)


if __name__ == "__main__":
    args = parser.parse_args()
    create_docker_compose(args)
