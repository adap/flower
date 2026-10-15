# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Wire-format tests for compacted-context protobuf messages."""

from flwr.proto.runseries_pb2 import CompactedContext  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetCompactedContextsRequest,
    GetCompactedContextsResponse,
    PutCompactedContextRequest,
    PutCompactedContextResponse,
)


def test_compacted_context_round_trip() -> None:
    """Keep context IDs, content, and timestamp across the wire."""
    context = CompactedContext(
        id=2**40,
        series_id=12,
        through_event_id=2**40 + 1,
        text="Summary",
        created_at="2026-10-15T12:00:00Z",
    )

    parsed = CompactedContext.FromString(context.SerializeToString())

    assert parsed == context


def test_get_compacted_contexts_round_trip() -> None:
    """Keep pagination presence and repeated contexts across the wire."""
    omitted = GetCompactedContextsRequest.FromString(
        GetCompactedContextsRequest().SerializeToString()
    )
    assert not omitted.HasField("limit")
    assert not omitted.HasField("before_context_id")

    request = GetCompactedContextsRequest(limit=2, before_context_id=2**40)
    parsed_request = GetCompactedContextsRequest.FromString(request.SerializeToString())
    assert parsed_request == request
    assert parsed_request.HasField("limit")
    assert parsed_request.HasField("before_context_id")

    response = GetCompactedContextsResponse(
        contexts=[CompactedContext(id=3), CompactedContext(id=2)]
    )
    assert GetCompactedContextsResponse.FromString(response.SerializeToString()) == (
        response
    )


def test_put_compacted_context_round_trip() -> None:
    """Keep the nested context in both write messages."""
    context = CompactedContext(through_event_id=2**40, text="Summary")
    request = PutCompactedContextRequest(context=context)
    response = PutCompactedContextResponse(context=context)

    assert PutCompactedContextRequest.FromString(request.SerializeToString()) == request
    assert PutCompactedContextResponse.FromString(response.SerializeToString()) == (
        response
    )
