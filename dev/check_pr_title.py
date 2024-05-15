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
"""This module is used to check a given PR title format."""

import pathlib
import re
import sys

import yaml


if __name__ == "__main__":

    pr_title = sys.argv[1]

    # Load the YAML configuration
    with (pathlib.Path(__file__).parent.resolve() / "changelog.yml").open(
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    # Extract types, project, and scope from the config
    types = "|".join(config["types"])
    project = "|".join(config["project"]) + "|\\*"
    scope = config["scope"]

    # Construct the pattern
    pattern_template = config["pattern_template"]
    pattern = pattern_template.format(types=types, project=project, scope=scope)

    # Check for the pattern in the first argument given to the script
    if re.search(pattern, pr_title):
        print("PR title is valid")
        sys.exit(0)
    else:
        print(
            f"PR title `{pr_title}` is invalid, it should be of the form: <PR_TYPE>(<PR_SCOPE>) "
            f"<PR_SUBJECT> with <PR_TYPE> in {types}, and "
            f"<PR_SCOPE> in {project} where '*' is used when modifying multiple projects),"
            "and <PR_SUBJECT> starting with "
            "a capitalized verb in the imperative mood and without a dot at the end.\n"
            "A valid example is: `feat(framework) Add flwr build CLI command`"
        )
        sys.exit(1)
