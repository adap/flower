import click
import flwr as fl

@click.command()
def cli():
    fl.server.start_server(config={"num_rounds": 1})