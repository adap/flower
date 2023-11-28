import random


def create_docker_compose(total_clients, number_of_rounds):
    # cpus is used to set the number of CPUs available to the container as a fraction of the total number of CPUs on the host machine.
    # mem_limit is used to set the memory limit for the container.
    client_configs = [
        {'mem_limit': '3g', 'batch_size': 32,  "cpus": 3.5, 'learning_rate': 0.001},
        {'mem_limit': '4g', 'batch_size': 64,  "cpus": 3, 'learning_rate': 0.02},
        {'mem_limit': '5g', 'batch_size': 128, "cpus": 2.5, 'learning_rate': 0.09},
        {'mem_limit': '6g', 'batch_size': 256, "cpus": 1, 'learning_rate': 0.15},
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
    mem_limit: 500m
    command:
      - --config.file=/etc/prometheus/prometheus.yml
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - cadvisor
    restart: on-failure

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.0
    container_name: cadvisor
    restart: on-failure
    privileged: true
    mem_limit: 500m
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
    mem_limit: 400m
    restart: on-failure
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./config/grafana.ini:/etc/grafana/grafana.ini
      - ./config/provisioning/datasources:/etc/grafana/provisioning/datasources
      - ./config/provisioning/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
      - cadvisor
    command:
      - --config=/etc/grafana/grafana.ini


  server:
    container_name: server
    shm_size: '6g'
    build:
      context: .
      dockerfile: Dockerfile
    command: python server.py --number_of_rounds={number_of_rounds} --total_clients={total_clients}
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
    for i in range(1, total_clients + 1):
        config = random.choice(client_configs)
        docker_compose_content += f"""
  client{i}:
    container_name: client{i}
    build:
      context: .
      dockerfile: Dockerfile
    command: python client.py --server_address=server:8080 --batch_size={config["batch_size"]} --learning_rate={config["learning_rate"]}
    mem_limit: {config['mem_limit']}
    deploy:
      resources:
        limits:
          cpus: "{(config['cpus'])}"
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

    with open('docker-compose.yml', 'w') as file:
        file.write(docker_compose_content)

if __name__ == "__main__":
    total_clients = 2  # Number of clients that will be created
    number_of_rounds = 5  # Number of rounds that the server will run for
    create_docker_compose(total_clients,number_of_rounds)
