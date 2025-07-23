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
"""Utils for ObjectStore."""


from typing import Union

from flwr.proto.appio_pb2 import PushAppMessagesRequest  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import PushMessagesRequest  # pylint: disable=E0611
from flwr.proto.message_pb2 import ObjectIDs  # pylint: disable=E0611

from . import ObjectStore


def store_mapping_and_register_objects(
    store: ObjectStore, request: Union[PushAppMessagesRequest, PushMessagesRequest]
) -> dict[str, ObjectIDs]:
    """Store Message object to descendants mapping and preregister objects."""
    if not request.messages_list:
        return {}

    objects_to_push: dict[str, ObjectIDs] = {}

    # Get run_id from the first message in the list
    # All messages of a request should in the same run
    run_id = request.messages_list[0].metadata.run_id

    # Keep track of unique object IDs registered so far
    # This is to avoid requesting to push the same object multiple times.
    unique_object_ids_so_far: set[str] = set()
    for object_tree in request.message_object_trees:
        # Preregister
        object_ids_just_registered = set(store.preregister(run_id, object_tree))

        # Record objects that need to be pushed (this is
        # equivalent to the set difference between the newly registered
        # object IDs and the unique object IDs seen so far)
        objects_to_push[object_tree.object_id] = ObjectIDs(
            object_ids=object_ids_just_registered - unique_object_ids_so_far
        )

        # Add to the set of unique object IDs if not already present
        unique_object_ids_so_far.update(object_ids_just_registered)

    return objects_to_push
