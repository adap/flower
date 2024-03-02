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
"""Test for Flower command line interface `run` command."""

from .run import load_flower_toml


def test_load_flower_toml() -> None:
    """Test if load_template returns a string."""
    # Prepare
    template_file = "flower.toml.tpl"
    expected_config = {
        "flower": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "Apache-2.0",
            "authors": ["The Flower Authors <hello@flower.ai>"],
            "components": {
                "serverapp": "fedgpt.server:app",
                "clientapp": "fedgpt.client:app",
            },
            "engine": {
                "name": "simulation",
                "simulation": {"super-node": {"count": 10}},
            },
        }
    }

    # Execute
    config = load_flower_toml()

    # Assert
    assert isinstance(config, expected_config)


# def test_new(tmp_path: str) -> None:
#     """Test if project is created for framework."""
#     # Prepare
#     project_name = "FedGPT"
#     framework = MlFramework.PYTORCH
#     expected_files_top_level = {
#         "requirements.txt",
#         "fedgpt",
#         "README.md",
#         "flower.toml",
#     }
#     expected_files_module = {
#         "__init__.py",
#         "server.py",
#         "client.py",
#     }

#     # Current directory
#     origin = os.getcwd()

#     try:
#         # Change into the temprorary directory
#         os.chdir(tmp_path)

#         # Execute
#         new(project_name=project_name, framework=framework)

#         # Assert
#         file_list = os.listdir(os.path.join(tmp_path, project_name.lower()))
#         assert set(file_list) == expected_files_top_level

#         file_list = os.listdir(
#             os.path.join(tmp_path, project_name.lower(), project_name.lower())
#         )
#         assert set(file_list) == expected_files_module
#     finally:
#         os.chdir(origin)
