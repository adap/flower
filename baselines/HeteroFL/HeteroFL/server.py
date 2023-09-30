import time
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
from flwr.common.typing import NDArrays, Scalar
from models import test
from torch.utils.data import DataLoader
from utils import save_model


def gen_evaluate_fn(
    trainloader: DataLoader,
    testloader: DataLoader,
    clients_testloaders,
    label_split,
    device: torch.device,
    model,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generates the function for centralized evaluation.

    Parameters
    ----------
    trainloader: DataLoader
        The Dataloader that was used for training the main model
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
            Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire test set for evaluation."""
        if server_round % 50 != 0 and server_round < 395:
            return 1, {}

        net = model
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        if server_round % 100 == 0:
            save_model(net, "model_after_round_{}.pth".format(server_round))

        print("in stat")
        start_time = time.time()
        with torch.no_grad():
            net.train(True)
            for input in trainloader:
                input_dict = {}
                input_dict["img"] = input[0].to(device)
                input_dict["label"] = input[1].to(device)
                net(input_dict)
        end_time = time.time()
        print(f"end of stat, time taken = {start_time - end_time}")

        local_loss = 0
        local_accuracy = 0
        for i in range(len(clients_testloaders)):
            client_loss, client_accuracy = test(
                net,
                clients_testloaders[i],
                label_split[i].type(torch.int),
                device=device,
            )
            local_loss += client_loss
            local_accuracy += client_accuracy

        global_loss, global_accuracy = test(net, testloader, device=device)

        # return statistics
        print(f"schezwan computed accuracy = {global_accuracy}")
        print(f"local_loss = {local_loss}, local_accuracy = {local_accuracy}")
        return global_loss, {
            "global_accuracy": global_accuracy,
            "local_loss": local_loss,
            "local_accuracy": local_accuracy,
        }

    return evaluate
