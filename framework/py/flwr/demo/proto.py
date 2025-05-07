from __future__ import annotations

import pickle
from dataclasses import dataclass, field

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
class SerdeHelper:

    extra: bytes = b""
    children_ids: list[str] = field(default_factory=list)

    @staticmethod
    def FromString(data: bytes) -> SerdeHelper:
        extra, ids = pickle.loads(data)
        return SerdeHelper(extra=extra, children_ids=ids)

    def SerializeToString(self) -> bytes:
        return pickle.dumps((self.extra, self.children_ids))


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
