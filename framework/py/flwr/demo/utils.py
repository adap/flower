def add_object_head(
    object_class: type,
    object_body: bytes,
) -> bytes:
    """Add object head to the serialized object."""
    object_type = object_class.__qualname__
    object_body_len = str(len(object_body))
    head = f"{object_type} {object_body_len}".encode("utf-8") + b"\x00"
    return head + object_body


def get_object_body(object_content: bytes) -> bytes:
    """Get object body from the serialized object."""
    _, body = object_content.split(b"\x00", 1)
    return body


def get_object_head(object_content: bytes) -> tuple[str, int]:
    """Get object head from the serialized object."""
    head, _ = object_content.split(b"\x00", 1)
    object_type, body_len = head.decode("utf-8").split(" ", 1)
    return object_type, int(body_len)
