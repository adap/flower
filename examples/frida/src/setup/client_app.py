import torch
from copy import deepcopy
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .app_common import check_config, get_model
from .data_loader import load_partition_offline, get_subsetloaders_offline, load_local_data_offline
from .client import Client, ClientSubset, FreeRider, FreeRiderSubset


def client_fn(context: Context):
    cfg = _make_cfg(context.run_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = check_config(cfg, cfg.attack_types)
    net = deepcopy(get_model(cfg, device)).to(device)

    node_cfg = context.node_config
    is_simulation = "partition-id" in node_cfg and "num-partitions" in node_cfg

    if is_simulation:
        partition_id = int(node_cfg["partition-id"])
        trainloader, valloader = load_partition_offline(cfg, partition_id)
    else:
        # flwr-supernode --node-config "data_path='/path/to/node_data'"
        data_path = node_cfg.get("data_path", cfg.path_to_local_dataset)
        trainloader, valloader = load_local_data_offline(cfg, data_path)

    if is_simulation:
        cid = str(partition_id)
        is_freerider = partition_id < cfg.num_freeriders
    else:
        cid = str(context.node_id)
        is_freerider = False 

    if cfg.canary:
        subsetloaders = get_subsetloaders_offline(cfg)
        if is_freerider and cfg.freerider_canary:
            client = FreeRiderSubset(cid, net, len(trainloader.dataset), cfg.freerider_type, subsetloaders, device, cfg)
        elif is_freerider:
            client = FreeRider(cid, net, len(trainloader.dataset), cfg.freerider_type, device, cfg)
        else:
            client = ClientSubset(cid, net, trainloader, valloader, subsetloaders, device, cfg)
    else:
        if is_freerider:
            client = FreeRider(cid, net, len(trainloader.dataset), cfg.freerider_type, device, cfg)
        else:
            client = Client(cid, net, trainloader, valloader, device, cfg)

    return client.to_client()


def _make_cfg(run_config: dict):
    class Cfg(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        def get(self, key, default=None):
            return super().get(key, default)
    
    cfg = Cfg(run_config)
    
    if isinstance(cfg.attack_types, str):
        cfg.attack_types = [x.strip() for x in cfg.attack_types.split(",")]
    
    if isinstance(cfg.n_encoder_layers, str):
        cfg.n_encoder_layers = [int(x.strip()) for x in cfg.n_encoder_layers.split(",")]
    
    if isinstance(cfg.n_gmm_layers, str):
        cfg.n_gmm_layers = [int(x.strip()) for x in cfg.n_gmm_layers.split(",")]
    
    if cfg.name_layer_grads == "None":
        cfg.name_layer_grads = None

    return cfg


app = ClientApp(client_fn=client_fn)
