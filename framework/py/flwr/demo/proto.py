from __future__ import annotations

import pickle
from dataclasses import dataclass

from flwr.demo.utils import get_object_head


@dataclass
class PullObjectRequest:
    object_id: str


@dataclass
class PullObjectResponse:
    object_content: bytes
    has_children: bool


@dataclass
class PushObjectRequest:
    object_id: str
    object_content: bytes
    has_children: bool


@dataclass
class PushObjectResponse:
    pass


@dataclass
class ObjectRef:

    names: list[str]
    ids: list[str]

    @staticmethod
    def FromString(data: bytes) -> ObjectRef:
        names, ids = pickle.loads(data)
        return ObjectRef(names=names, ids=ids)

    def SerializeToString(self) -> bytes:
        return pickle.dumps((self.names, self.ids))


class ServerAppIoStub:

    store: dict[str, tuple[bytes, bool]] = {}

    def PullObject(self, request: PullObjectRequest) -> PullObjectResponse:
        # Simulate a gRPC call
        content, has_children = self.store.get(request.object_id, (b"", False))
        _type, _len = get_object_head(content)
        print(f"Pulled '{_type}' object ({_len} bytes) with ID '{request.object_id}'")
        return PullObjectResponse(
            object_content=content,
            has_children=has_children,
        )

    def PushObject(self, request: PushObjectRequest) -> PushObjectResponse:
        # Simulate a gRPC call
        self.store[request.object_id] = (request.object_content, request.has_children)
        _type, _len = get_object_head(request.object_content)
        print(f"Pushed '{_type}' object ({_len} bytes) with ID '{request.object_id}'")
        return PushObjectResponse()
