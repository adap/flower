from __future__ import annotations

from flwr.demo.proto import (
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    SerdeHelper,
    ServerAppIoStub,
)
from flwr.demo.serializable import Array, ArrayRecord, Serializable, get_object_class
from flwr.demo.utils import get_object_body


def serialize_and_push_object(obj: Serializable, stub: ServerAppIoStub) -> None:
    """Serialize an object and push it to the server."""
    if obj.children:
        for child in obj.children:
            serialize_and_push_object(child, stub)

    object_content = obj.serialize()

    stub.PushObject(
        PushObjectRequest(
            object_id=obj.object_id,
            object_content=object_content,
            has_children=obj.children is not None,
        )
    )


def pull_and_deserialize_object(object_id: str, stub: ServerAppIoStub) -> Serializable:
    proto: PullObjectResponse = stub.PullObject(PullObjectRequest(object_id=object_id))
    obj_cls = get_object_class(proto.object_content)

    if not proto.has_children:
        return obj_cls.deserialize(proto.object_content)

    children: list[Serializable] = []
    refs = SerdeHelper.FromString(get_object_body(proto.object_content))
    for child_object_id in refs.children_ids:
        children.append(pull_and_deserialize_object(child_object_id, stub))

    return obj_cls.deserialize(proto.object_content, children)


stub = ServerAppIoStub()  # Assuming you have a gRPC stub instance

arr = Array(
    dtype="float32",
    shape=[2, 3],
    stype="ndarray",
    # Data length: 11 bytes
    data=b"Mock data 1",
)
arr2 = Array(
    dtype="float32",
    shape=[2, 3],
    stype="ndarray",
    # Data length: 76 bytes
    data=b"Hello world! This is a long string that will be chunked into smaller pieces.",
)
arr_record = ArrayRecord(data={"arr": arr, "other": arr2})
print(f"Initial object:\n{arr_record}\n")

print("Serializing and pushing object to server...")
serialize_and_push_object(arr_record, stub)

print("\nPulling and deserializing object from server...")
retrieved_obj = pull_and_deserialize_object(arr_record.object_id, stub)
print(f"\nRetrieved object:\n{retrieved_obj}\n")
