import click
from flwr.cli.server.server import server

@click.group()
def app():
    pass

app.add_command(server)