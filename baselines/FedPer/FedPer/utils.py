"""
Utility functions
"""

""" FIRST PART INCLUDES VISUALIZATION FUNCTIONS """

import numpy as np
import pickle

from pathlib import Path
from secrets import token_hex
from typing import Dict, Optional, Union
from matplotlib import pyplot as plt
from flwr.server.history import History


def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "centralized"
    metric_dict = (hist.metrics_centralized if metric_type == "centralized" else hist.metrics_distributed)
    rounds, values = zip(*metric_dict["accuracy"])

    # let's extract centralised loss (main metric reported in FedProx paper)
    rounds_loss, values_loss = zip(*hist.losses_centralized)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()

def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    extra_results: Optional[Dict] = {},
    default_filename: Optional[str] = "results.pkl",
) -> None:
    """Saves results from simulation to pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store both history and extra_results.
        If path is a directory, the default_filename will be used.
        path doesn't exist, it will be created. If file exists, a
        randomly generated suffix will be added to the file name. This
        is done to avoid overwritting results.
    extra_results : Optional[Dict]
        A dictionary containing additional results you would like
        to be saved to disk. Default: {} (an empty dictionary)
    default_filename: Optional[str]
        File used by default if file_path points to a directory instead
        to a file. Default: "results.pkl"
    """

    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Adds a randomly generated suffix to the file name (so it doesn't
        overwrite the file)."""
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Appends the default file name to the path."""
        print("Using default filename")
        return path_ / default_filename

    if path.is_dir(): path = _complete_path_with_default_name(path)

    if path.is_file(): path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")
    data = {"history": history, **extra_results}
    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn
from torch import Tensor









class ModelSplit(ABC, nn.Module):
    """Abstract class for splitting a model into body and head. Optionally, a fixed head can also be created."""

    def __init__(
            self,
            model: nn.Module,
            has_fixed_head: bool = False
    ):
        """
        Initialize ModelSplit attributes. A call is made to the _setup_model_parts method.

        Args:
            model: dict containing the vocab sizes of the input attributes.
            has_fixed_head: whether the model should contain a fixed_head.
        """
        super().__init__()

        self._body, self._head = self._get_model_parts(model)

        self._fixed_head = copy.deepcopy(self.head) if has_fixed_head else None
        self._use_fixed_head = False

    @abstractmethod
    def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """
        Return the body and head of the model.

        Args:
            model: model to be split into head and body

        Returns:
            Tuple where the first element is the body of the model and the second is the head.
        """
        pass

    @property
    def body(self) -> nn.Module:
        """Return model body."""
        return self._body

    @body.setter
    def body(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """
        Set model body.

        Args:
            state_dict: dictionary of the state to set the model body to.
        """
        self.body.load_state_dict(state_dict, strict=True)

    @property
    def head(self) -> nn.Module:
        """Return model head."""
        return self._head

    @head.setter
    def head(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """
        Set model head.

        Args:
            state_dict: dictionary of the state to set the model head to.
        """
        self.head.load_state_dict(state_dict, strict=True)

    @property
    def fixed_head(self) -> Optional[nn.Module]:
        """Return model fixed_head."""
        return self._fixed_head

    @fixed_head.setter
    def fixed_head(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """
        Set model fixed_head.

        Args:
            state_dict: dictionary of the state to set the model fixed head to.
        """
        if self._fixed_head is None:
            # When the fixed_head was not initialized
            return
        self._fixed_head.load_state_dict(state_dict, strict=True)
        self.disable_fixed_head()

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get model parameters (without fixed head).

        Returns:
            Body and head parameters
        """
        return [val.cpu().numpy() for val in [*self.body.state_dict().values(), *self.head.state_dict().values()]]

    def set_parameters(self, state_dict: Dict[str, Tensor]) -> None:
        """
        Set model parameters.

        Args:
            state_dict: dictionary of the state to set the model to.
        """
        # Copy to maintain the order of the parameters and add the missing parameters to the state_dict
        ordered_state_dict = OrderedDict(self.state_dict().copy())
        # Update with the values of the state_dict
        ordered_state_dict.update({k: v for k, v in state_dict.items()})
        self.load_state_dict(ordered_state_dict, strict=True)

    def enable_head(self) -> None:
        """Enable gradient tracking for the head parameters."""
        for param in self.head.parameters():
            param.requires_grad = True

    def enable_body(self) -> None:
        """Enable gradient tracking for the body parameters."""
        for param in self.body.parameters():
            param.requires_grad = True

    def disable_head(self) -> None:
        """Disable gradient tracking for the head parameters."""
        for param in self.head.parameters():
            param.requires_grad = False

    def disable_body(self) -> None:
        """Disable gradient tracking for the body parameters."""
        for param in self.body.parameters():
            param.requires_grad = False

    def disable_fixed_head(self) -> None:
        """Disable gradient tracking for the fixed head parameters."""
        if self._fixed_head is None:
            return
        for param in self._fixed_head.parameters():
            param.requires_grad = False

    def use_fixed_head(self, use_fixed_head: bool) -> None:
        """
        Set whether the fixed head should be used for forward.

        Args:
            use_fixed_head: boolean indicating whether to use the fixed head or not.
        """
        self._use_fixed_head = use_fixed_head

    def forward(self, inputs: Any) -> Any:
        """Forward inputs through the body and the head (or fixed head)."""
        x = self.body(inputs)
        if self._use_fixed_head and self.fixed_head is not None:
            return self.fixed_head(x)
        return self.head(x)

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import torch.nn as nn


class ModelManager(ABC):
    """Manager for models with Body/Head split."""

    def __init__(
            self,
            client_id: int,
            model_split_class: Type[ModelSplit],
            has_fixed_head: bool = False
    ):
        """
        Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            model_split_class: Class to be used to split the model into body and head\
                (concrete implementation of ModelSplit).
            has_fixed_head: Whether a fixed head should be created.
        """
        super().__init__()

        self.client_id = client_id
        # self.config = config
        self._model = model_split_class(self._create_model(), has_fixed_head)

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Return model to be splitted into head and body."""
        pass

    @abstractmethod
    def train(
        self,
        train_id: int,
        epochs: int = 1,
        tag: Optional[str] = None,
        fine_tuning: bool = False
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """
        Train the model maintained in self.model.

        Args:
            train_id: id of the train round.
            epochs: number of training epochs.
            tag: str of the form <Algorithm>_<model_train_part>.
                <Algorithm> - indicates the federated algorithm that is being performed\
                              (FedAvg, FedPer, FedRep, FedBABU or FedHybridAvgLGDual).
                              In the case of FedHybridAvgLGDual the tag also includes which part of the algorithm\
                                is being performed, either FedHybridAvgLGDual_FedAvg or FedHybridAvgLGDual_LG-FedAvg.
                <model_train_part> - indicates the part of the model that is being trained (full, body, head).
                This tag can be ignored if no difference in train behaviour is desired between federated algortihms.
            fine_tuning: whether the training performed is for model fine-tuning or not.

        Returns:
            Dict containing the train metrics.
        """
        pass

    @abstractmethod
    def test(self, test_id: int) -> Dict[str, float]:
        """
        Test the model maintained in self.model.

        Args:
            test_id: id of the test round.

        Returns:
            Dict containing the test metrics.
        """
        pass

    @abstractmethod
    def train_dataset_size(self) -> int:
        """Return train data set size."""
        pass

    @abstractmethod
    def test_dataset_size(self) -> int:
        """Return test data set size."""
        pass

    @abstractmethod
    def total_dataset_size(self) -> int:
        """Return total data set size."""
        pass

    @property
    def model(self) -> nn.Module:
        """Return model."""
        return self._model
    
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    #parameters_to_weights,
    #weights_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    #parameters_to_weights,
    #weights_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


    
from pathlib import Path
from enum import Enum

# FL Algorithms
class Algorithms(Enum):
    FEDAVG = "FedAvg"
    FEDPER = "FedPer"
