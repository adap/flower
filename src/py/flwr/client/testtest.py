import socket


def is_port_free(host: str, port: int) -> bool:
    """Checks whether a specific port is free (not in use) on the specified host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Attempt to connect to the specified host and port
        result = sock.connect_ex((host, port))
        # If connect_ex returns 0, the port is in use; otherwise, it's free
        return result != 0


address = "0.0.0.0:9092"
_host, _port = [v for v in address.split(":")]
_port = int(_port)
print(is_port_free(_host, _port))
