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
"""Validates the project's name property."""


import re


def validate_project_name(name: str) -> bool:
    """Validate the project name against PEP 621 and PEP 503 specifications.

    Conventions at a glance:
    - Must be lowercase
    - Must not contain special characters
    - Must use hyphens(recommended) or underscores. No spaces.
    - Recommended to be no more than 40 characters long (But it can be)

    Parameters
    ----------
    name : str
        The project name to validate.

    Returns
    -------
    bool
        True if the name is valid, False otherwise.
    """
    if not name or len(name) > 40 or not re.match(r"^[a-z0-9-_]+$", name):
        return False
    return True
