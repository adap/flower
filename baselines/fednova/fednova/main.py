from dataset import load_datasets
from client import gen_client_fn
import hydra
from omegaconf import DictConfig, OmegaConf
from strategy import FedNova, weighted_average
from client import FlowerClient
from flwr.common import ndarrays_to_parameters
import flwr as fl
from utils import fit_config
from hydra.utils import instantiate


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
	"""Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
	# 1. Print parsed config
	print(OmegaConf.to_yaml(cfg))

	# 2. Prepare your dataset

	trainloaders, testloader, data_ratios = load_datasets(cfg)

	# 3. Define your clients

	client_fn = gen_client_fn(num_epochs=cfg.num_epochs,
							  trainloaders=trainloaders,
							  testloader=testloader,
							  data_ratios=data_ratios,
							  model=cfg.model,
							  exp_config=cfg)

	init_parameters = [layer_param.cpu().numpy() for _,layer_param in instantiate(cfg.model).state_dict().items()]
	init_parameters = ndarrays_to_parameters(init_parameters)

	# 4. Define your strategy
	strategy = FedNova(exp_config=cfg,
					   evaluate_metrics_aggregation_fn=weighted_average,
					   accept_failures=False,
					   on_evaluate_config_fn=fit_config,
					   initial_parameters=init_parameters)

	# 5. Start Simulation

	history = fl.simulation.start_simulation(client_fn=client_fn,
											 num_clients=cfg.num_clients,
											 config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
											 strategy=strategy,
											 client_resources=cfg.client_resources,
											 ray_init_args={"ignore_reinit_error": True, "num_cpus": 6})

	round, loss = history.losses_distributed[-1]
	round, accuracy = history.metrics_distributed["accuracy"][-1]
	print("---------------------Round: {} Test loss: Test Accuracy {}----------------------".format(round, loss, accuracy))


# 6. Save your results
# Here you can save the `history` returned by the simulation and include
# also other buffers, statistics, info needed to be saved in order to later
# on generate the plots you provide in the README.md. You can for instance
# access elements that belong to the strategy for example:
# data = strategy.get_my_custom_data() -- assuming you have such method defined.
# Hydra will generate for you a directory each time you run the code. You
# can retrieve the path to that directory with this:
# save_path = HydraConfig.get().runtime.output_dir


if __name__ == "__main__":
	main()
