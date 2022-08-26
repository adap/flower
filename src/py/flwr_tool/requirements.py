# Copyright 2022 Adap GmbH. All Rights Reserved.
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
"""This module will when called will extract requirements.txt file from
pyproject.toml files for all examples."""

import glob
from os import path

import tomli

DIR_PATH = path.dirname(path.realpath(__file__))
EXAMPLES_PATH = path.normpath(f"{DIR_PATH}/../../../examples")


def clean_version(version):
    return version.replace("^", "")


def regenerate() -> None:
    pyprojects = glob.glob(f"{EXAMPLES_PATH}/**/pyproject.toml")

    for pp in pyprojects:
        # Directory into which the requirements.txt will be written
        basedir = path.dirname(pp)

        # Final output
        txt = ""

        # Open pyproject.toml and extracts dependencies
        with open(pp, "rb") as f:
            toml_dict = tomli.load(f)
            dependencies = toml_dict["tool"]["poetry"]["dependencies"]
            for lib, version in dependencies.items():
                if lib == "python":
                    continue

                if type(version) is dict:
                    if "extras" in version:
                        txt += f"{lib}[{','.join(version['extras'])}]=={clean_version(version['version'])}\n"
                    elif "git" in version:
                        # Example:
                        # tensorflow-privacy = {'git': 'https://github.com/path/to/package-two', 'rev': '41b95ec'}
                        # transformed to:
                        # tensorflow-privacy @ git+https://github.com/tensorflow/privacy@aaf4c25
                        txt += f"{lib} @ git+{version['git']}@{version['rev']}\n"
                    else:
                        raise Exception("Unhandled version dict")

                if type(version) is str:
                    txt += f"{lib}=={clean_version(version)}\n"

        with open(f"{basedir}/requirements.txt", "w", encoding="utf-8") as f:
            f.write(txt)


if __name__ == "__main__":
    regenerate()
