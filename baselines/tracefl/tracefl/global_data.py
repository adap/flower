"""Global Data Management Module.

This module provides functionality for managing global data structures and state across
the federated learning system. It includes functions for tracking round-level data and
maintaining global state information.
"""

# tracefl/global_data.py

round_data = {}


def update_round_data(
    server_round, initial_parameters, client2ws, client2num_examples, client2class
):
    """Update the global round data with new information.

    Args:
        server_round: The round number
        initial_parameters: The initial parameters for the round
        client2ws: A dictionary mapping clients to their workspace
        client2num_examples: A dictionary mapping clients to the number of examples
        client2class: A dictionary mapping clients to their class
    """
    round_data[server_round] = {
        "initial_parameters": initial_parameters,
        "client2ws": client2ws,
        "client2num_examples": client2num_examples,
        "client2class": client2class,
    }
