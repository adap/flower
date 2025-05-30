from __future__ import annotations

import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from flwr.demo.proto import SerdeHelper
from flwr.demo.utils import add_object_head, get_object_body, get_object_head

MAX_CHUNK_SIZE = 30  # Maximum size of a chunk in bytes


class Serializable(ABC):
    """Abstract base class for serializable objects."""

    @property
    @abstractmethod
    def children(self) -> list[Serializable] | None:
        """Return a list of child objects."""

    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize the object to bytes.

        Returns
        -------
        bytes
            The serialized object content.
        """

    @classmethod
    @abstractmethod
    def deserialize(
        cls, object_content: bytes, children: list[Serializable] | None = None
    ) -> Serializable:
        """Deserialize the object from bytes.

        Parameters
        ----------
        object_content : bytes
            The serialized object content.
        children : list[Serializable] | None
            A list of child objects, if any. Defaults to None.

        Returns
        -------
        Serializable
            The deserialized object.
        """

    @property
    def object_id(self) -> str:
        """Return the object ID."""
        return hashlib.sha256(self.serialize()).hexdigest()


@dataclass
class Array(Serializable):

    dtype: str
    shape: list[int]
    stype: str
    data: bytes

    @property
    def children(self) -> list[Serializable] | None:
        """Return a list of child objects."""
        if len(self.data) < MAX_CHUNK_SIZE:
            return None
        return [
            Chunk(self.data[i : i + MAX_CHUNK_SIZE])
            for i in range(0, len(self.data), MAX_CHUNK_SIZE)
        ]

    def serialize(self) -> bytes:
        """Serialize the object to bytes."""
        chunks: list[Chunk] = self.children
        if chunks is not None:
            children_ids = [chunk.object_id for chunk in chunks]
            extra = pickle.dumps((self.dtype, self.shape, self.stype))
        else:
            children_ids = []
            extra = pickle.dumps((self.dtype, self.shape, self.stype, self.data))
        object_body = SerdeHelper(
            extra=extra, children_ids=children_ids
        ).SerializeToString()
        object_content = add_object_head(Array, object_body)
        return object_content

    @classmethod
    def deserialize(
        cls,
        object_content: bytes,
        children: list[Serializable] | None = None,
    ) -> Array:
        """Deserialize the object from bytes."""
        object_body = get_object_body(object_content)
        object_id = hashlib.sha256(object_content).hexdigest()
        proto = SerdeHelper.FromString(object_body)
        if proto.children_ids:
            id_to_child: dict[str, Chunk] = {
                child.object_id: child for child in children
            }
            data = b"".join(
                [id_to_child[child_id].data for child_id in proto.children_ids]
            )
            dtype, shape, stype = pickle.loads(proto.extra)
        else:
            dtype, shape, stype, data = pickle.loads(proto.extra)
        array = cls(dtype=dtype, shape=shape, stype=stype, data=data)
        array.object_id = object_id
        return array

    def __getattribute__(self, name):
        if name == "shape":  # Only `shape` is mutable
            self.__dict__.pop("_object_id", None)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name != "__dict__":  # Escape the __dict__ attribute
            self.__dict__.pop("_object_id", None)
        super().__setattr__(name, value)

    @property
    def object_id(self) -> str:
        """Return the object ID."""
        if "_object_id" not in self.__dict__:
            self.__dict__["_object_id"] = super().object_id
        return self.__dict__["_object_id"]

    @object_id.setter
    def object_id(self, value: str):
        """Set the object ID."""
        self.__dict__["_object_id"] = value


@dataclass
class ArrayRecord(Serializable):
    """ArrayRecord class for serializing and deserializing Array objects."""

    data: dict[str, Array] = field(default_factory=dict)

    @property
    def children(self) -> list[Serializable] | None:
        """Return a list of child objects."""
        return list(self.data.values())

    def serialize(self) -> bytes:
        """Serialize the object to bytes."""
        extra = pickle.dumps(list(self.data.keys()))
        children_ids = [child.object_id for child in self.data.values()]
        proto = SerdeHelper(extra=extra, children_ids=children_ids)
        object_content = add_object_head(ArrayRecord, proto.SerializeToString())
        return object_content

    @classmethod
    def deserialize(
        cls,
        object_content: bytes,
        children: list[Serializable] | None = None,
    ) -> ArrayRecord:
        """Deserialize the object from bytes."""
        if children is None:
            raise ValueError("Children must be provided for deserialization.")

        object_body = get_object_body(object_content)
        proto = SerdeHelper.FromString(object_body)

        keys = pickle.loads(proto.extra)
        id_to_child = {child.object_id: child for child in children}
        data = {
            name: id_to_child[child_id]
            for name, child_id in zip(keys, proto.children_ids)
        }
        return cls(data=data)


@dataclass
class Chunk(Serializable):
    """Chunk class for serializing and deserializing Array objects."""

    data: bytes

    @property
    def children(self) -> list[Serializable] | None:
        """Return a list of child objects."""
        return None

    def serialize(self) -> bytes:
        """Serialize the object to bytes."""
        object_body = SerdeHelper(extra=self.data).SerializeToString()
        object_content = add_object_head(Chunk, object_body)
        return object_content

    @classmethod
    def deserialize(cls, object_content: bytes) -> Chunk:
        """Deserialize the object from bytes."""
        object_body = get_object_body(object_content)
        extra = SerdeHelper.FromString(object_body).extra
        return cls(data=extra)


serializable_class_registry: dict[str, type[Serializable]] = {
    Array.__qualname__: Array,
    ArrayRecord.__qualname__: ArrayRecord,
    Chunk.__qualname__: Chunk,
}


def get_object_class(object_content: bytes) -> type[Serializable]:
    """Get object class from the serialized object."""
    object_type, _ = get_object_head(object_content)
    return serializable_class_registry[object_type]
