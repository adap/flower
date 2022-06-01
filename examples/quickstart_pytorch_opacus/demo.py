import multiprocessing as mp

from client import start_client
from server import start_server


def main():
    """Launch a server and multiple clients for the demo."""
    num_clients = 2
    server_process = mp.Process(target=start_server)
    server_process.start()
    with mp.Pool(num_clients) as pool:
        pool.map(start_client, range(num_clients))
    server_process.kill()


if __name__ == "__main__":
    main()
    # TODO: provide server side test function to server
    # TODO: log weights and metrics to a file for the demo
