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


from .constant import HEAD_BODY_DIVIDER, HEAD_VALUE_DIVIDER
from .inflatable import (
    UnexpectedObjectContentError,
    _get_object_head,
    get_object_id,
    is_valid_sha256_hash,
)
from .inflatable_grpc_utils import inflatable_class_registry


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
