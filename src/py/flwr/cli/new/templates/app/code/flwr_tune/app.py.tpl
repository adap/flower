"""$project_name: A Flower / FlowerTune app."""

import os
from datetime import datetime

from flwr.client import ClientApp
from flwr.server import ServerApp

from $import_name.client import gen_client_fn
from $import_name.server import gen_server_fn

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"

# Create output directory given current timestamp
current_time = datetime.now()
folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
os.makedirs(save_path, exist_ok=True)

# ClientApp
client = ClientApp(
    client_fn=gen_client_fn(
        save_path,
    ),
)

# ServerApp
server = ServerApp(
    server_fn=gen_server_fn(
        save_path,
    ),
)
