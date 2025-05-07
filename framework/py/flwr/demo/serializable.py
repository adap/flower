from __future__ import annotations

import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from flwr.demo.proto import ObjectRef
from flwr.demo.utils import add_object_head, get_object_body, get_object_head


class Serializable(ABC):
    """Abstract base class for serializable objects."""

    @property
    @abstractmethod
    def children(self) -> list[Serializable] | None:
        """Return a list of child objects."""

    @abstractmethod
    def serialize(self) -> tuple[bytes, str]:
        """Serialize the object to bytes.

        Returns
        -------
        tuple[bytes, str]
            A tuple containing:
            - The object content as bytes
            - The object ID as a string
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
        return None

    def serialize(self) -> bytes:
        """Serialize the object to bytes."""
        object_body = pickle.dumps((self.dtype, self.shape, self.stype, self.data))
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
        dtype, shape, stype, data = pickle.loads(object_body)
        return cls(dtype=dtype, shape=shape, stype=stype, data=data)

    def __getattribute__(self, name):
        if name != "__dict__":  # Escape the __dict__ attribute
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
        keys = list(self.data.keys())
        children_ids = [child.object_id for child in self.data.values()]
        proto = ObjectRef(names=keys, ids=children_ids)
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
        proto = ObjectRef.FromString(object_body)

        id_to_child = {child.object_id: child for child in children}
        data = {
            name: id_to_child[child_id]
            for name, child_id in zip(proto.names, proto.ids)
        }
        return cls(data=data)


serializable_class_registry: dict[str, type[Serializable]] = {
    Array.__qualname__: Array,
    ArrayRecord.__qualname__: ArrayRecord,
}


def get_object_class(object_content: bytes) -> type[Serializable]:
    """Get object class from the serialized object."""
    object_type, _ = get_object_head(object_content)
    return serializable_class_registry[object_type]
