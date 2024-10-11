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
"""Abstract classes for Flower User Auth Plugin."""

from flwr.common.auth_plugin import SuperExecAuthPlugin, UserAuthPlugin, Metadata
from flwr.proto.exec_pb2 import LoginRequest, LoginResponse
from pathlib import Path
from typing import Sequence, Tuple, Union, Any, List, Dict, Optional, Set
import base64
import sys
import csv
import typer
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    load_ssh_private_key,
    load_ssh_public_key,
)
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_private_key,
    bytes_to_public_key,
    public_key_to_bytes,
    generate_shared_key,
    compute_hmac,
)
from flwr.common.logger import log
from logging import WARNING, INFO

Metadata = List[Any]

_PUBLIC_KEY_HEADER = "public-key"
_AUTH_TOKEN_HEADER = "auth-token"


def _get_value_from_tuples(
    key_string: str, tuples: Sequence[Tuple[str, Union[str, bytes]]]
) -> bytes:
    value = next((value for key, value in tuples if key == key_string), "")
    if isinstance(value, str):
        return value.encode()

    return value


class PublicKeySuperExecPlugin(SuperExecAuthPlugin):
    """Abstract Flower SuperExec Auth Plugin class."""

    def __init__(self, config: Dict[str, Any]):
        private_key_path = config.get("private-key-path")
        public_key_path = config.get("public-key-path")
        list_public_keys_path = config.get("list-public-keys-path")
        self.user_public_keys, self.superexec_private_key, public_key = _try_setup_superexec_user_authentication(private_key_path, public_key_path, list_public_keys_path)
        if len(self.user_public_keys) == 0:
            log(WARNING, "Authentication enabled, but no known public keys configured")
        else:
            log(
                INFO,
                "User authentication enabled with %d known public keys",
                len(self.user_public_keys),
            )
        
        self.encoded_server_public_key = base64.urlsafe_b64encode(public_key_to_bytes(public_key))
            
    def send_auth_endpoint(self) -> LoginResponse:
        """
        Send relevant auth url as a LoginResponse.
        """
        return LoginResponse(auth_type="public-key", auth_url=self.encoded_server_public_key.decode('utf-8'))

    def authenticate(self, metadata: Sequence[Tuple[str, Union[str, bytes]]]):
        """
        Authenticate auth tokens in the provided metadata.
        """
        user_public_key_bytes = base64.urlsafe_b64decode(
            _get_value_from_tuples(
                _PUBLIC_KEY_HEADER, metadata
            )
        )
        if user_public_key_bytes not in self.user_public_keys:
            return False
        
        shared_secret = base64.urlsafe_b64decode(
            _get_value_from_tuples(
                _AUTH_TOKEN_HEADER, metadata
            )
        )
        public_key = bytes_to_public_key(user_public_key_bytes)

        expected_shared_secret = generate_shared_key(self.superexec_private_key, public_key)
        return shared_secret == expected_shared_secret


class PublicKeyUserPlugin(UserAuthPlugin):
    """Abstract Flower User Auth Plugin class."""

    def __init__(self, config: Dict[str, Any], federation: str):
        federation_config = config["tool"]["flwr"]["federations"][federation]["authentication"]
        superexec_public_key_encoded_str = federation_config.get("superexec-public-key")
        if superexec_public_key_encoded_str is None:
            typer.secho(
                "❌ SuperExec that you connect to has user authentication enabled. "
                "Please run the command `flwr login` to get the SuperExec's public "
                "key and try again.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)
        superexec_public_key_bytes = base64.urlsafe_b64decode(superexec_public_key_encoded_str.encode('utf-8'))
        self.superexec_public_key = bytes_to_public_key(superexec_public_key_bytes)
        private_key_path = federation_config.get("private-key-path")
        public_key_path = federation_config.get("public-key-path")
        self.private_key, self.public_key = _try_setup_user_authentication(private_key_path, public_key_path)
        self.shared_secret = generate_shared_key(
            self.private_key, self.superexec_public_key
        )
        self.encoded_public_key = base64.urlsafe_b64encode(
            public_key_to_bytes(self.public_key)
        )

    @staticmethod
    def login(auth_url: str, config: Dict[str, Any], federation: str):
        """
        Read relevant auth details from federation config.
        """
        config["tool"]["flwr"]["federations"][federation]["authentication"] = {}
        config["tool"]["flwr"]["federations"][federation]["authentication"]["auth-type"] = "public-key"
        config["tool"]["flwr"]["federations"][federation]["authentication"]["superexec-public-key"] = auth_url
        return config

    def provide_auth_details(self, metadata: Metadata) -> Metadata:
        """
        Provide relevant auth tokens in the metadata.
        """
        metadata.append(
            (
                _PUBLIC_KEY_HEADER,
                self.encoded_public_key,
            )
        )

        metadata.append(
            (
                _AUTH_TOKEN_HEADER,
                base64.urlsafe_b64encode(
                    self.shared_secret
                ),
            )
        )
        return metadata

def _try_setup_user_authentication(
    private_key_path: Optional[str],
    public_key_path: Optional[str],
) -> Tuple[
    ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey
]:
    if not private_key_path and not public_key_path:
        typer.secho(
            "❌ SuperExec that you connect to has user authentication enabled. "
            "User authentication requires file paths to 'private_key_path', "
            "'public_key_path' to be provided.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    if not private_key_path or not public_key_path:
        typer.secho(
            "❌ User authentication requires file paths to 'private_key_path', "
            "and 'public_key_path' to be provided (providing only one of them "
            "is not sufficient).",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    try:
        ssh_private_key = load_ssh_private_key(
            Path(private_key_path).read_bytes(),
            None,
        )
        if not isinstance(ssh_private_key, ec.EllipticCurvePrivateKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm) as err:
        typer.secho(
            "❌ Error: Unable to parse the private key file in "
            "'private_key_path'. User authentication requires elliptic curve "
            "private and public key pair. Please ensure that the file "
            "path points to a valid private key file and try again.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err

    try:
        ssh_public_key = load_ssh_public_key(Path(public_key_path).read_bytes())
        if not isinstance(ssh_public_key, ec.EllipticCurvePublicKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm) as err:
        typer.secho(
            "❌ Error: Unable to parse the public key file in "
            "'public_key_path'. User authentication requires elliptic curve "
            "private and public key pair. Please ensure that the file "
            "path points to a valid public key file and try again.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err

    return (ssh_private_key, ssh_public_key)


def _try_setup_superexec_user_authentication(
    private_key_path: Optional[str],
    public_key_path: Optional[str],
    list_public_keys_path: Optional[str],
) -> Tuple[Set[bytes], ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
    if (
        not list_public_keys_path
        and not private_key_path
        and not public_key_path
    ):
        sys.exit(
            "Authentication requires providing file paths for "
            "'list-public-keys-path', 'private-key-path' and "
            "'public-key-path' in config.yaml file. "
            "Provide all three to enable authentication."
        )

    if (
        not list_public_keys_path
        or not private_key_path
        or not public_key_path
    ):
        sys.exit(
            "Authentication requires providing file paths for "
            "'list-public-keys-path', 'private-key-path' and "
            "'public-key-path' in config.yaml file. "
            "Provide all three to enable authentication."
        )

    client_keys_file_path = Path(list_public_keys_path)
    if not client_keys_file_path.exists():
        sys.exit(
            "The provided path to the known public keys CSV file does not exist: "
            f"{client_keys_file_path}. "
            "Please provide the CSV file path containing known public keys "
            "to 'list-public-keys-path' in config.yaml file."
        )

    client_public_keys: Set[bytes] = set()

    try:
        ssh_private_key = load_ssh_private_key(
            Path(private_key_path).read_bytes(),
            None,
        )
        if not isinstance(ssh_private_key, ec.EllipticCurvePrivateKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        sys.exit(
            "Error: Unable to parse the private key file in "
            "'--auth-superexec-private-key'. Authentication requires elliptic "
            "curve private and public key pair. Please ensure that the file "
            "path points to a valid private key file and try again."
        )

    try:
        ssh_public_key = load_ssh_public_key(
            Path(public_key_path).read_bytes()
        )
        if not isinstance(ssh_public_key, ec.EllipticCurvePublicKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm):
        sys.exit(
            "Error: Unable to parse the public key file in "
            "'--auth-superexec-public-key'. Authentication requires elliptic "
            "curve private and public key pair. Please ensure that the file "
            "path points to a valid public key file and try again."
        )

    with open(client_keys_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for element in row:
                public_key = load_ssh_public_key(element.encode())
                if isinstance(public_key, ec.EllipticCurvePublicKey):
                    client_public_keys.add(public_key_to_bytes(public_key))
                else:
                    sys.exit(
                        "Error: Unable to parse the public keys in the CSV "
                        "file. Please ensure that the CSV file path points to a valid "
                        "known SSH public keys files and try again."
                    )
        return (
            client_public_keys,
            ssh_private_key,
            ssh_public_key,
        )
