# tracefl/global_data.py

round_data = {}


def update_round_data(
    server_round, initial_parameters, client2ws, client2num_examples, client2class
):
    round_data[server_round] = {
        "initial_parameters": initial_parameters,
        "client2ws": client2ws,
        "client2num_examples": client2num_examples,
        "client2class": client2class,
    }
