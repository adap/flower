"""statavg: A Flower Baseline."""

from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder, StandardScaler

from flwr.common import Context
from flwr.server import (
    ServerApp,
    ServerAppComponents,
    ServerConfig,
    SimpleClientManager,
)
from statavg.dataset import prepare_dataset
from statavg.model import get_model
from statavg.server import ResultsSaverServer, save_results_and_clean_dir
from statavg.strategy import define_server_strategy


def get_evaluate_fn(cfg: DictConfig):
    """Return evaluate_fn used in strategy."""
    _, testset = prepare_dataset(
        cfg.num_clients, cfg.path_to_dataset, cfg.include_test, cfg.testset_ratio
    )

    def evalaute_fn(server_round, parameters, scaler):
        """Evaluate the test set (if provided)."""
        _ = server_round
        if testset.empty:
            # this implies that testset is not used
            # and thus, included_testset from config file is False
            return None, {"accuracy": None}

        y_test = testset[["type"]]
        enc_y = LabelEncoder()
        y_test = enc_y.fit_transform(y_test.to_numpy().reshape(-1))
        x_test = testset.drop(["type"], axis=1).to_numpy()

        # normalization
        # Check if the directory of the scaler exists and pick a scaler
        # of an arbitrary user. It's the same for all users.
        if scaler:
            scaler = scaler[1]
            x_test = scaler.transform(x_test)
        else:
            scaler = StandardScaler()
            x_test = scaler.fit_transform(x_test)

        model = get_model(cfg.input_shape, cfg.num_classes)
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, {"accuracy": accuracy}

    return evalaute_fn


def server_fn(context: Context):
    """Define standards for simulation."""
    cfg = OmegaConf.create(context.run_config)
    print(cfg)
    strategy_class = define_server_strategy(cfg)

    def fit_config_fn(server_round: int):
        """Return the server's config file."""
        config = {
            "current_round": server_round,
        }

        return config

    strategy = strategy_class(
        min_fit_clients=cfg.num_clients,
        on_fit_config_fn=fit_config_fn,
        evaluate_fn=get_evaluate_fn(cfg),
    )
    config = ServerConfig(num_rounds=int(cfg.num_server_rounds))
    client_manager = SimpleClientManager()
    server = ResultsSaverServer(
        strategy=strategy,
        client_manager=client_manager,
        results_saver_fn=save_results_and_clean_dir,
        run_config=cfg,
    )
    return ServerAppComponents(server=server, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
