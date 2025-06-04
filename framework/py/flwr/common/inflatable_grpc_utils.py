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
"""InflatableObject utils."""


from time import sleep
from typing import Callable, Optional

from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611

from .inflatable import (
    InflatableObject,
    get_object_head_values_from_object_content,
    get_object_id,
)
from .message import Message
from .record import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict

# Helper registry that maps names of classes to their type
inflatable_class_registry: dict[str, type[InflatableObject]] = {
    Array.__qualname__: Array,
    ArrayRecord.__qualname__: ArrayRecord,
    ConfigRecord.__qualname__: ConfigRecord,
    Message.__qualname__: Message,
    MetricRecord.__qualname__: MetricRecord,
    RecordDict.__qualname__: RecordDict,
}


class ObjectUnavailableError(Exception):
    """Exception raised when an object has been pre-registered but is not yet
    available."""

    def __init__(self, object_id: str):
        super().__init__(f"Object with ID '{object_id}' is not yet available.")


class ObjectIdNotPreregisteredError(Exception):
    """Exception raised when an object ID is not pre-registered."""

    def __init__(self, object_id: str):
        super().__init__(f"Object with ID '{object_id}' could not be found.")


def push_object_to_servicer(
    obj: InflatableObject,
    push_object_fn: Callable[[str, bytes], None],
    object_ids_to_push: Optional[set[str]] = None,
) -> set[str]:
    """Recursively deflate an object and push it to the servicer.

    Objects with the same ID are not pushed twice. If `object_ids_to_push` is set,
    only objects with those IDs are pushed. It returns the set of pushed object
    IDs.

    Parameters
    ----------
    obj : InflatableObject
        The object to push.
    push_object_fn : Callable[[str, bytes], None]
        A function that takes an object ID and its content as bytes, and pushes
        it to the servicer. This function should raise `ObjectIdNotPreregisteredError`
        if the object ID is not pre-registered.
    object_ids_to_push : Optional[set[str]] (default: None)
        A set of object IDs to push. If object ID of the given object is not in this
        set, the object will not be pushed.

    Returns
    -------
    set[str]
        A set of object IDs that were pushed to the servicer.
    """
    pushed_object_ids: set[str] = set()
    # Push children if it has any
    if children := obj.children:
        for child in children.values():
            pushed_object_ids |= push_object_to_servicer(
                child, push_object_fn, object_ids_to_push
            )

    # Deflate object and push
    object_content = obj.deflate()
    object_id = get_object_id(object_content)
    # Push always if no object set is specified, or if the object is in the set
    if object_ids_to_push is None or object_id in object_ids_to_push:
        # The function may raise an error if the object ID is not pre-registered
        push_object_fn(object_id, object_content)
        pushed_object_ids.add(object_id)

    return pushed_object_ids


def pull_object_from_servicer(
    object_id: str,
    pull_object_fn: Callable[[str], bytes],
) -> InflatableObject:
    """Recursively inflate an object by pulling it from the servicer.

    Parameters
    ----------
    object_id : str
        The ID of the object to pull.
    pull_object_fn : Callable[[str], bytes]
        A function that takes an object ID and returns the object content as bytes.
        The function should raise `ObjectUnavailableError` if the object is not yet
        available, or `ObjectIdNotPreregisteredError` if the object ID is not
        pre-registered.

    Returns
    -------
    InflatableObject
        The pulled object.
    """
    # Pull object
    while True:
        try:
            # The function may raise an error if the object ID is not pre-registered
            object_content: bytes = pull_object_fn(object_id)
            break  # Exit loop if object is successfully pulled
        except ObjectUnavailableError:
            sleep(0.1)  # Retry after a short delay

    # Extract object class and object_ids of children
    obj_type, children_obj_ids, _ = get_object_head_values_from_object_content(
        object_content=object_content
    )
    # Resolve object class
    cls_type = inflatable_class_registry[obj_type]

    # Pull all children objects
    children: dict[str, InflatableObject] = {}
    for child_object_id in children_obj_ids:
        children[child_object_id] = pull_object_from_servicer(
            child_object_id, pull_object_fn
        )

    # Inflate object passing its children
    return cls_type.inflate(object_content, children=children)


def make_pull_object_fn_grpc(
    pull_object_grpc: Callable[[PullObjectRequest], PullObjectResponse],
    node: Node,
    run_id: int,
) -> Callable[[str], bytes]:
    """Create a pull object function that uses gRPC to pull objects.

    Parameters
    ----------
    pull_object_grpc : Callable[[PullObjectRequest], PullObjectResponse]
        The gRPC function to pull objects, e.g., `FleetStub.PullObject`.
    node : Node
        The node making the request.
    run_id : int
        The run ID for the current operation.

    Returns
    -------
    Callable[[str], bytes]
        A function that takes an object ID and returns the object content as bytes.
        The function raises `ObjectIdNotPreregisteredError` if the object ID is not
        pre-registered, or `ObjectUnavailableError` if the object is not yet available.
    """

    def pull_object_fn(object_id: str) -> bytes:
        request = PullObjectRequest(node=node, run_id=run_id, object_id=object_id)
        response: PullObjectResponse = pull_object_grpc(request)
        if not response.object_found:
            raise ObjectIdNotPreregisteredError(object_id)
        if not response.object_available:
            raise ObjectUnavailableError(object_id)
        return response.object_content

    return pull_object_fn


def make_push_object_fn_grpc(
    push_object_grpc: Callable[[PushObjectRequest], PushObjectResponse],
    node: Node,
    run_id: int,
) -> Callable[[str, bytes], None]:
    """Create a push object function that uses gRPC to push objects.

    Parameters
    ----------
    push_object_grpc : Callable[[PushObjectRequest], PushObjectResponse]
        The gRPC function to push objects, e.g., `FleetStub.PushObject`.
    node : Node
        The node making the request.
    run_id : int
        The run ID for the current operation.

    Returns
    -------
    Callable[[str, bytes], None]
        A function that takes an object ID and its content as bytes, and pushes it
        to the servicer. The function raises `ObjectIdNotPreregisteredError` if
        the object ID is not pre-registered.
    """

    def push_object_fn(object_id: str, object_content: bytes) -> None:
        request = PushObjectRequest(
            node=node, run_id=run_id, object_id=object_id, object_content=object_content
        )
        response: PushObjectResponse = push_object_grpc(request)
        if not response.stored:
            raise ObjectIdNotPreregisteredError(object_id)

    return push_object_fn
