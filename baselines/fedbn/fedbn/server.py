"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

def get_on_fit_config():
    def fit_config_fn(server_round: int):
        # resolve and convert to python dict
        fit_config = {}
        fit_config["round"] = server_round  # add round info
        return fit_config

    return fit_config_fn