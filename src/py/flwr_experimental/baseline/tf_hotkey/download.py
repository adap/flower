# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Helper script to download Spoken Keyword dataset."""


from logging import INFO

from flwr.common.logger import log
from flwr_experimental.baseline.dataset import tf_hotkey_partitioned


def main() -> None:
    """Download data."""
    log(INFO, "Download Keyword Detection")
    tf_hotkey_partitioned.hotkey_load()


if __name__ == "__main__":
    main()
