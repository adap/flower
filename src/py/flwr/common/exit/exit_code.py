# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Exit codes."""


from __future__ import annotations


class ExitCode:
    """Exit codes for Flower components."""

    # Success exit codes (0-99)
    SUCCESS = 0  # Successful exit without any errors or signals
    GRACEFUL_EXIT_SIGINT = 1  # Graceful exit triggered by SIGINT
    GRACEFUL_EXIT_SIGQUIT = 2  # Graceful exit triggered by SIGQUIT
    GRACEFUL_EXIT_SIGTERM = 3  # Graceful exit triggered by SIGTERM

    # SuperLink-specific exit codes (100-199)
    SUPERLINK_THREAD_CRASH = 100

    # ServerApp-specific exit codes (200-299)

    # SuperNode-specific exit codes (300-399)
    SUPERNODE_REST_ADDRESS_INVALID = 300
    SUPERNODE_NODE_AUTH_KEYS_REQUIRED = 301
    SUPERNODE_NODE_AUTH_KEYS_INVALID = 302

    # ClientApp-specific exit codes (400-499)

    # Simulation-specific exit codes (500-599)

    # Common exit codes (600-)
    COMMON_ADDRESS_INVALID = 600
    COMMON_MISSING_EXTRA_REST = 601
    COMMON_TLS_NOT_SUPPORTED = 602

    def __new__(cls) -> ExitCode:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


# All short help messages for exit codes
EXIT_CODE_HELP = {
    # Success exit codes (0-99)
    ExitCode.SUCCESS: "",
    ExitCode.GRACEFUL_EXIT_SIGINT: "",
    ExitCode.GRACEFUL_EXIT_SIGQUIT: "",
    ExitCode.GRACEFUL_EXIT_SIGTERM: "",
    # SuperLink-specific exit codes (100-199)
    ExitCode.SUPERLINK_THREAD_CRASH: "An important background thread has crashed.",
    # ServerApp-specific exit codes (200-299)
    # SuperNode-specific exit codes (300-399)
    ExitCode.SUPERNODE_REST_ADDRESS_INVALID: (
        "When using the REST API, please provide `https://` or "
        "`http://` before the server address (e.g. `http://127.0.0.1:8080`)"
    ),
    ExitCode.SUPERNODE_NODE_AUTH_KEYS_REQUIRED: (
        "Node authentication requires file paths to both "
        "'--auth-supernode-private-key' and '--auth-supernode-public-key' "
        "to be provided (providing only one of them is not sufficient)."
    ),
    ExitCode.SUPERNODE_NODE_AUTH_KEYS_INVALID: (
        "Node uthentication requires elliptic curve private and public key pair. "
        "Please ensure that the file path points to a valid private/public key "
        "file and try again."
    ),
    # ClientApp-specific exit codes (400-499)
    # Simulation-specific exit codes (500-599)
    # Common exit codes (600-)
    ExitCode.COMMON_ADDRESS_INVALID: (
        "Please provide a valid URL, IPv4 or IPv6 address."
    ),
    ExitCode.COMMON_MISSING_EXTRA_REST: """
Extra dependencies required for using the REST-based Fleet API are missing.

To use the REST API, install `flwr` with the `rest` extra:

    `pip install "flwr[rest]"`.
""",
    ExitCode.COMMON_TLS_NOT_SUPPORTED: "Please use the '--insecure' flag.",
}
