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
"""Helper class for temporarily identifying anonymous clients."""


from datetime import datetime
from logging import ERROR
from typing import Callable, Optional

from flwr.common.logger import log

STAGE_BOUND = 0
STAGE_UNBOUND = 1
STAGE_TO_BE_BOUND = 2
STAGE_TO_BE_UNBOUND = 3


class EphemeralIDManager:
    """Static class managing the ephemeral ID."""

    _stage = STAGE_UNBOUND
    _create_node: Optional[Callable[[], None]] = None
    _delete_node: Optional[Callable[[], None]] = None
    _max_rounds = 0
    _ttl = 0.0
    _round_count = 0
    _created_at = datetime.now()

    def __new__(cls) -> None:  # type: ignore
        """Prevent instantiation of the EphemeralIDManager class."""
        raise TypeError("Binding is a static class and cannot be instantiated.")

    @staticmethod
    def set_create_node_delete_node(
        create_node: Optional[Callable[[], None]],
        delete_node: Optional[Callable[[], None]],
    ) -> None:
        """Set the `create_node()` / `delete_node()` functions."""
        EphemeralIDManager._create_node = create_node
        EphemeralIDManager._delete_node = delete_node

    @staticmethod
    def on_bind(max_rounds: int, ttl: float) -> None:
        """On ClientBinder.bind() method being called."""
        # The valid stage
        if EphemeralIDManager._stage == STAGE_UNBOUND:
            if (
                EphemeralIDManager._create_node is None
                or EphemeralIDManager._create_node is None
            ):
                raise TypeError(
                    "Temporary binding is only supported for grpc-rere clients."
                )
            EphemeralIDManager._stage = STAGE_TO_BE_BOUND
            EphemeralIDManager._max_rounds = max_rounds
            EphemeralIDManager._ttl = ttl
            EphemeralIDManager._round_count = 0
            EphemeralIDManager._created_at = datetime.now()
        # Invalid stages
        elif EphemeralIDManager._stage == STAGE_BOUND:
            log(ERROR, "Temporary binding has already been established.")
        elif EphemeralIDManager._stage == STAGE_TO_BE_BOUND:
            log(ERROR, "Binding in progress.")
        elif EphemeralIDManager._stage == STAGE_TO_BE_UNBOUND:
            log(ERROR, "Unbinding in progress.")
        else:
            raise ValueError(f"Unknown binding stage: {EphemeralIDManager._stage}")

    @staticmethod
    def on_unbind() -> None:
        """On ClientBinder.unbind() method being called."""
        # The valid stage
        if EphemeralIDManager._stage == STAGE_BOUND:
            EphemeralIDManager._stage = STAGE_TO_BE_UNBOUND
        # Invalid stages
        elif EphemeralIDManager._stage == STAGE_UNBOUND:
            log(ERROR, "Temporary binding not found.")
        elif EphemeralIDManager._stage == STAGE_TO_BE_BOUND:
            log(ERROR, "Binding in progress.")
        elif EphemeralIDManager._stage == STAGE_TO_BE_UNBOUND:
            log(ERROR, "Unbinding in progress.")
        else:
            raise ValueError(f"Unknown binding stage: {EphemeralIDManager._stage}")

    @staticmethod
    def before_send() -> None:
        """Before `send()` being called in the client loop."""
        # Check if the current binding has timed out.
        if EphemeralIDManager._stage == STAGE_BOUND:
            elapsed = (datetime.now() - EphemeralIDManager._created_at).total_seconds()
            if elapsed >= EphemeralIDManager._ttl:
                EphemeralIDManager._stage = STAGE_TO_BE_UNBOUND

        if (
            EphemeralIDManager._create_node is None
            or EphemeralIDManager._delete_node is None
        ):
            raise TypeError(
                "Temporary binding is only supported for grpc-rere clients."
            )
        if EphemeralIDManager._stage == STAGE_TO_BE_BOUND:
            EphemeralIDManager._create_node()  # pylint: disable=not-callable
            EphemeralIDManager._stage = STAGE_BOUND
        elif EphemeralIDManager._stage == STAGE_TO_BE_UNBOUND:
            EphemeralIDManager._delete_node()  # pylint: disable=not-callable
            EphemeralIDManager._stage = STAGE_UNBOUND

    @staticmethod
    def after_send() -> None:
        """After `send()` being called in the client loop."""
        if EphemeralIDManager._stage == STAGE_BOUND:
            EphemeralIDManager._round_count += 1
            # Check if the current binding is expired.
            elapsed = (datetime.now() - EphemeralIDManager._created_at).total_seconds()
            if (
                elapsed >= EphemeralIDManager._ttl
                or EphemeralIDManager._round_count >= EphemeralIDManager._max_rounds
            ):
                EphemeralIDManager._stage = STAGE_TO_BE_UNBOUND

        if (
            EphemeralIDManager._create_node is None
            or EphemeralIDManager._delete_node is None
        ):
            raise TypeError(
                "Temporary binding is only supported for grpc-rere clients."
            )
        if EphemeralIDManager._stage == STAGE_TO_BE_UNBOUND:
            EphemeralIDManager._delete_node()  # pylint: disable=not-callable
            EphemeralIDManager._stage = STAGE_UNBOUND


class ClientBinder:
    """Helper class allowing the anonymous client to be temporarily identifiable."""

    def __new__(cls) -> None:  # type: ignore
        """Prevent instantiation of the ClientBinder class."""
        raise TypeError("ClientBinder is a static class and cannot be instantiated.")

    @staticmethod
    def bind(max_rounds: int, max_time_seconds: float) -> None:
        """Establish a temporary binding for the client.

        This function will have no effect if the client is non-anonymous.

        Parameters
        ----------
        max_rounds : int
            The maximum rounds of communication that the binding can last.
        max_time_seconds: float
            The maximum time in seconds that the binding can last.
        """
        EphemeralIDManager.on_bind(max_rounds, max_time_seconds)

    @staticmethod
    def unbind() -> None:
        """Remove the temporary binding established for the client.

        This function will have no effect if the client is non-anonymous.
        """
        EphemeralIDManager.on_unbind()
