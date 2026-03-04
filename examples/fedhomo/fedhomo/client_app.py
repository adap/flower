"""fedhomo: A Flower Baseline."""

import logging

import torch
from flwr.client import ClientApp
from flwr.common import Context

from fedhomo.dataset import load_data
from fedhomo.encrypted_client import EncryptedFlowerClient
from fedhomo.model import get_model

log = logging.getLogger(__name__)


def client_fn(context: Context):
    num_partitions = int(context.node_config["num-partitions"])
    partition_id = int(context.node_config["partition-id"])
    dataset = context.run_config["dataset"]
    local_epochs = context.run_config["local-epochs"]

    log.info(
        "Client %s: loading partition %d of %d (%s)",
        context.node_id, partition_id, num_partitions, dataset,
    )

    net = get_model(dataset)
    print(f"DEBUG node_config={context.node_config}", flush=True)
    print(f"DEBUG partition_id={partition_id} num_partitions={num_partitions}", flush=True)
    trainloader, valloader = load_data(partition_id, num_partitions, dataset)

    return EncryptedFlowerClient(
        cid=context.node_id,
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        epochs=local_epochs,
    )



app = ClientApp(client_fn=client_fn)
