"""fedprox: A Flower Baseline."""

from collections import defaultdict

from easydict import EasyDict


def context_to_easydict(context):
    """Parser to generate internal config files once you are given are context.

    This is to facilitate easier sharing and better config management rather than using
    a dict.
    """
    configs_to_parse = {
        "run_config": _extract_run_configs_per_type(context.run_config),
        "node_config": _extract_run_configs_per_type(context.node_config),
    }
    return EasyDict(configs_to_parse)


def _extract_run_configs_per_type(config):
    parsed_configs = defaultdict(defaultdict)

    for key, value in config.items():
        if "." in key:
            category, name = key.split(".")
            parsed_configs[category][name.replace("-", "_")] = value
        else:
            parsed_configs[key.replace("-", "_")] = value
    return parsed_configs
