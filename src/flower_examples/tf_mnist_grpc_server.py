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
"""Minimal example on how to start a simple Flower server."""

import flower as flwr


def main() -> None:
    """Start server and train four rounds."""
    server = flwr.Server(client_manager=flwr.SimpleClientManager())
    flwr.app.start_server(server, config={"num_rounds": 4})


if __name__ == "__main__":
    main()
