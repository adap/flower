import torch
import flwr as fl
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context

from .app_common import check_config, get_model
from .model import get_parameters
from .server import PrivacyAttacksForDefense, PrivacyAttacksForDefenseFedProx
from .data_loader import load_partition_offline, get_subsetloaders_offline


def server_fn(context: Context):
    cfg = _make_cfg(context.run_config)
    cfg = check_config(cfg, cfg.attack_types)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_model(cfg, device).to(device)
    params = fl.common.ndarrays_to_parameters(get_parameters(net))

    freeriders = [i < cfg.num_freeriders for i in range(cfg.num_clients)]

    training_datasets = []
    validation_datasets = []
    for i in range(cfg.num_clients):
        tr_i, val_i = load_partition_offline(cfg, i)
        training_datasets.append(tr_i)
        validation_datasets.append(val_i)

    subsets = get_subsetloaders_offline(cfg) if cfg.canary else None

    use_fedprox = bool(cfg.get("use_fedprox", False))
    StrategyClass = PrivacyAttacksForDefenseFedProx if use_fedprox else PrivacyAttacksForDefense

    kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=cfg.num_clients,
        min_evaluate_clients=0,
        min_available_clients=cfg.num_clients,
        initial_parameters=params,
        evaluate_fn=None,       
        training_datasets=training_datasets,
        validation_datasets=validation_datasets,
        subsets=subsets,
        net=net,
        freeriders=freeriders,
        cfg=cfg,
        device=device,
    )
    if use_fedprox:
        kwargs["proximal_mu"] = float(cfg.get("proximal_mu", 0.1))

    strategy = StrategyClass(**kwargs)

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=cfg.num_rounds),
    )


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


app = ServerApp(server_fn=server_fn)