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
"""InflatableObject utilities."""

import concurrent.futures
import random
import threading
import time
from typing import Callable, Optional

from .constant import (
    HEAD_BODY_DIVIDER,
    HEAD_VALUE_DIVIDER,
    MAX_CONCURRENT_PULLS,
    MAX_CONCURRENT_PUSHES,
    PULL_BACKOFF_CAP,
    PULL_INITIAL_BACKOFF,
    PULL_MAX_TIME,
    PULL_MAX_TRIES_PER_OBJECT,
)
from .inflatable import (
    InflatableObject,
    UnexpectedObjectContentError,
    _get_object_head,
    get_object_head_values_from_object_content,
    get_object_id,
    is_valid_sha256_hash,
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


def push_objects(
    objects: dict[str, InflatableObject],
    push_object_fn: Callable[[str, bytes], None],
    *,
    object_ids_to_push: Optional[set[str]] = None,
    keep_objects: bool = False,
    max_concurrent_pushes: int = MAX_CONCURRENT_PUSHES,
) -> None:
    """Push multiple objects to the servicer.

    Parameters
    ----------
    objects : dict[str, InflatableObject]
        A dictionary of objects to push, where keys are object IDs and values are
        `InflatableObject` instances.
    push_object_fn : Callable[[str, bytes], None]
        A function that takes an object ID and its content as bytes, and pushes
        it to the servicer. This function should raise `ObjectIdNotPreregisteredError`
        if the object ID is not pre-registered.
    object_ids_to_push : Optional[set[str]] (default: None)
        A set of object IDs to push. If not provided, all objects will be pushed.
    keep_objects : bool (default: False)
        If `True`, the original objects will be kept in the `objects` dictionary
        after pushing. If `False`, they will be removed from the dictionary to avoid
        high memory usage.
    max_concurrent_pushes : int (default: MAX_CONCURRENT_PUSHES)
        The maximum number of concurrent pushes to perform.
    """
    if object_ids_to_push is not None:
        # Filter objects to push only those with IDs in the set
        objects = {k: v for k, v in objects.items() if k in object_ids_to_push}

    lock = threading.Lock()

    def push(obj_id: str) -> None:
        """Push a single object."""
        object_content = objects[obj_id].deflate()
        if not keep_objects:
            with lock:
                del objects[obj_id]
        push_object_fn(obj_id, object_content)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_concurrent_pushes
    ) as executor:
        list(executor.map(push, list(objects.keys())))


def pull_objects(  # pylint: disable=too-many-arguments
    object_ids: list[str],
    pull_object_fn: Callable[[str], bytes],
    *,
    max_concurrent_pulls: int = MAX_CONCURRENT_PULLS,
    max_time: Optional[float] = PULL_MAX_TIME,
    max_tries_per_object: Optional[int] = PULL_MAX_TRIES_PER_OBJECT,
    initial_backoff: float = PULL_INITIAL_BACKOFF,
    backoff_cap: float = PULL_BACKOFF_CAP,
) -> dict[str, bytes]:
    """Pull multiple objects from the servicer.

    Parameters
    ----------
    object_ids : list[str]
        A list of object IDs to pull.
    pull_object_fn : Callable[[str], bytes]
        A function that takes an object ID and returns the object content as bytes.
        The function should raise `ObjectUnavailableError` if the object is not yet
        available, or `ObjectIdNotPreregisteredError` if the object ID is not
        pre-registered.
    max_concurrent_pulls : int (default: MAX_CONCURRENT_PULLS)
        The maximum number of concurrent pulls to perform.
    max_time : Optional[float] (default: PULL_MAX_TIME)
        The maximum time to wait for all pulls to complete. If `None`, waits
        indefinitely.
    max_tries_per_object : Optional[int] (default: PULL_MAX_TRIES_PER_OBJECT)
        The maximum number of attempts to pull each object. If `None`, pulls
        indefinitely until the object is available.
    initial_backoff : float (default: PULL_INITIAL_BACKOFF)
        The initial backoff time in seconds for retrying pulls after an
        `ObjectUnavailableError`.
    backoff_cap : float (default: PULL_BACKOFF_CAP)
        The maximum backoff time in seconds. Backoff times will not exceed this value.

    Returns
    -------
    dict[str, bytes]
        A dictionary where keys are object IDs and values are the pulled
        object contents.
    """
    if max_tries_per_object is None:
        max_tries_per_object = int(1e9)
    if max_time is None:
        max_time = float("inf")

    results: dict[str, bytes] = {}
    results_lock = threading.Lock()
    err_to_raise: Optional[Exception] = None
    early_stop = threading.Event()
    start = time.monotonic()

    def pull_with_retries(object_id: str) -> None:
        """Attempt to pull a single object with retry and backoff."""
        nonlocal err_to_raise
        tries = 0
        delay = initial_backoff

        while not early_stop.is_set():
            try:
                object_content = pull_object_fn(object_id)
                with results_lock:
                    results[object_id] = object_content
                return

            except ObjectUnavailableError as err:
                tries += 1
                if (
                    tries >= max_tries_per_object
                    or time.monotonic() - start >= max_time
                ):
                    # Stop all work if one object exhausts retries
                    early_stop.set()
                    with results_lock:
                        if err_to_raise is None:
                            err_to_raise = err
                    return

                # Apply exponential backoff with ±20% jitter
                sleep_time = delay * (1 + random.uniform(-0.2, 0.2))
                early_stop.wait(sleep_time)
                delay = min(delay * 2, backoff_cap)

            except ObjectIdNotPreregisteredError as err:
                # Permanent failure: object ID is invalid
                early_stop.set()
                with results_lock:
                    if err_to_raise is None:
                        err_to_raise = err
                return

    # Submit all pull tasks concurrently
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_concurrent_pulls
    ) as executor:
        futures = {
            executor.submit(pull_with_retries, obj_id): obj_id for obj_id in object_ids
        }

        # Wait for completion
        concurrent.futures.wait(futures)

    if err_to_raise is not None:
        raise err_to_raise

    return results


def inflate_object_from_contents(
    object_id: str,
    object_contents: dict[str, bytes],
    *,
    keep_object_contents: bool = False,
    objects: Optional[dict[str, InflatableObject]] = None,
) -> InflatableObject:
    """Inflate an object from object contents.

    Parameters
    ----------
    object_id : str
        The ID of the object to inflate.
    object_contents : dict[str, bytes]
        A dictionary mapping object IDs to their contents as bytes.
        All descendant objects must be present in this dictionary.
    keep_object_contents : bool (default: False)
        If `True`, the object content will be kept in the `object_contents`
        dictionary after inflation. If `False`, the object content will be
        removed from the dictionary to save memory.
    objects : Optional[dict[str, InflatableObject]] (default: None)
        No need to provide this parameter. A dictionary to store already
        inflated objects, mapping object IDs to their corresponding
        `InflatableObject` instances.

    Returns
    -------
    InflatableObject
        The inflated object.
    """
    if objects is None:
        # Initialize objects dictionary
        objects = {}

    if object_id in objects:
        # If the object is already in the objects dictionary, return it
        return objects[object_id]

    # Extract object class and object_ids of children
    object_content = object_contents[object_id]
    obj_type, children_obj_ids, _ = get_object_head_values_from_object_content(
        object_content=object_contents[object_id]
    )

    # Remove the object content from the dictionary to save memory
    if not keep_object_contents:
        del object_contents[object_id]

    # Resolve object class
    cls_type = inflatable_class_registry[obj_type]

    # Inflate all children objects
    children: dict[str, InflatableObject] = {}
    for child_obj_id in children_obj_ids:
        children[child_obj_id] = inflate_object_from_contents(
            child_obj_id,
            object_contents,
            keep_object_contents=keep_object_contents,
            objects=objects,
        )

    # Inflate object passing its children
    obj = cls_type.inflate(object_content, children=children)
    del object_content  # Free memory after inflation
    objects[object_id] = obj
    return obj


def validate_object_content(content: bytes) -> None:
    """Validate the deflated content of an InflatableObject."""
    try:
        # Check if there is a head-body divider
        index = content.find(HEAD_BODY_DIVIDER)
        if index == -1:
            raise ValueError(
                "Unexpected format for object content. Head and body "
                "could not be split."
            )

        head = _get_object_head(content)

        # check if the head has three parts:
        # <object_type> <children_ids> <object_body_len>
        head_decoded = head.decode(encoding="utf-8")
        head_parts = head_decoded.split(HEAD_VALUE_DIVIDER)

        if len(head_parts) != 3:
            raise ValueError("Unexpected format for object head.")

        obj_type, children_str, body_len = head_parts

        # Check that children IDs are valid IDs
        children = children_str.split(",")
        for children_id in children:
            if children_id and not is_valid_sha256_hash(children_id):
                raise ValueError(
                    f"Detected invalid object ID ({children_id}) in children."
                )

        # Check that object type is recognized
        if obj_type not in inflatable_class_registry:
            if obj_type != "CustomDataClass":  # to allow for the class in tests
                raise ValueError(f"Object of type {obj_type} is not supported.")

        # Check if the body length in the head matches that of the body
        actual_body_len = len(content) - len(head) - len(HEAD_BODY_DIVIDER)
        if actual_body_len != int(body_len):
            raise ValueError(
                f"Object content length expected {body_len} bytes but got "
                f"{actual_body_len} bytes."
            )

    except ValueError as err:
        raise UnexpectedObjectContentError(
            object_id=get_object_id(content), reason=str(err)
        ) from err
