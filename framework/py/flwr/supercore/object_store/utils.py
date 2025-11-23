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


from flwr.proto.appio_pb2 import PushAppMessagesRequest  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import PushMessagesRequest  # pylint: disable=E0611

from . import ObjectStore


def store_mapping_and_register_objects(
    store: ObjectStore, request: PushAppMessagesRequest | PushMessagesRequest
) -> set[str]:
    """Store Message object to descendants mapping and preregister objects."""
    if not request.messages_list:
        return set()
    objects_to_push: set[str] = set()
    # Get run_id from the first message in the list
    # All messages of a request should in the same run
    run_id = request.messages_list[0].metadata.run_id

    for object_tree in request.message_object_trees:
        # Preregister
        unavailable_obj_ids = store.preregister(run_id, object_tree)
        # Keep track of objects that need to be pushed
        objects_to_push |= set(unavailable_obj_ids)

    return objects_to_push
