# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils for handling working with numpy."""
from typing import Any, Dict, List

import numpy as np
import yaml


class YamlHandler:
    """A Singleton class for handling YAML serialization and deserialization.

    It supports custom representations for numpy arrays and lists.
    """

    _instance: "YamlHandler" = None

    def __new__(cls, *args, **kwargs) -> "YamlHandler":
        """Creates or returns the singleton instance of the class.

        Returns
        -------
        yaml_handler: YamlHandler
            The singleton instance of the YamlHandler class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
            cls._instance._setup_yaml()
        return cls._instance

    def _setup_yaml(self) -> None:
        """Sets up the custom YAML representers for numpy arrays and lists."""
        if not self._initialized:
            yaml.add_representer(np.ndarray, self._numpy_array_representer)
            yaml.add_representer(list, self._list_representer)
            self._initialized = True

    @staticmethod
    def _numpy_array_representer(dumper: yaml.Dumper, data: np.ndarray) -> yaml.Node:
        """Custom numpy array representer.

        Needed to ensure that numpy.arrays are serialized as lists style
        (e.g., [1, 2, 3]).

        Parameters
        ----------
        dumper : yaml.Dumper
            The YAML dumper.
        data : np.ndarray
            The numpy array to represent.

        Returns
        -------
        yaml_node: yaml.Node
            The YAML node representing the numpy array.
        """
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data.tolist(), flow_style=True
        )

    @staticmethod
    def _list_representer(dumper: yaml.Dumper, data: List[Any]) -> yaml.Node:
        """Custom list representer.

        Needed to ensure that list are serialized in a flow
        style (e.g., [1, 2, 3]), not hyphened.

        Parameters
        ----------
        dumper : yaml.Dumper
            The YAML dumper.
        data : List[Any]
            The list to represent.

        Returns
        -------
        yaml_node: yaml.Node
            The YAML node representing the list.
        """
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    def dump(self, data, filename) -> None:
        """Dumps a dictionary to a YAML file.

        Parameters
        ----------
        data : dict
            The dictionary to dump.
        filename : str
            The name of the YAML file to save the data in.
        """
        with open(filename, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

    def load(self, filename: str) -> Dict[str, Any]:
        """Load a dictionary from a YAML file.

        Parameters
        ----------
        filename : str
            The name of the YAML file to load the data from.

        Returns
        -------
        data: Dict[str, Any]
            The dictionary loaded from the YAML file.
        """
        with open(filename) as file:
            return yaml.safe_load(file)

    def load_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load a dictionary from an in-memory dictionary, ensuring it matches the
        deserialization process.

        Parameters
        ----------
        data : Dict
            The in-memory dictionary to load.

        Returns
        -------
        Dict[str, Any]
            The dictionary loaded from the in-memory dictionary.
        """
        yaml_str = yaml.dump(data, default_flow_style=False)
        return yaml.safe_load(yaml_str)

    def dump_to_str(self, data: Dict[str, Any]) -> str:
        """Dumps a dictionary to a YAML string.

        Parameters
        ----------
        data : dict
            The dictionary to dump.

        Returns
        -------
        str
            The YAML string representation of the dictionary.
        """
        return yaml.dump(data, default_flow_style=False)
