from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
)
from flwr.server.client_proxy import ClientProxy
from logging import WARNING, DEBUG
from flwr.common.logger import log
import torch
from flwr.server.strategy import FedAvg
import wandb
from utils.utils_mnist import (
    VAE,
    VAE_CNN,
    visualize_plotly_latent_representation,
    vae_loss,
    vae_rec_loss,
)
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class VisualiseFedGuide(FedAvg):
    """Adapted FedGuide strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        lr_g=1e-3,
        steps_g=10,
        num_classes=10,
        alignment_dataloader=None,
        device=None,
        lambda_align_g=1.0,
        latent_dim=2,
        initial_generator_params: Optional[Parameters] = None,
        gen_model=None,
        global_eval_data=None,
        folder_name=None,
    ) -> None:

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.lr_g = lr_g
        self.steps_g = steps_g
        self.num_classes = num_classes
        self.alignment_dataloader = alignment_dataloader
        self.device = device
        self.lambda_align_g = lambda_align_g
        self.latent_dim = latent_dim
        self.initial_generator_params = initial_generator_params
        self.gen_model = gen_model
        self.local_param_dict = {
            f"client_param_{cid}": self.initial_parameters
            for cid in range(self.min_available_clients)
        }
        self.global_eval_data = global_eval_data
        self.folder_name = folder_name

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedGuide(accept_failures={self.accept_failures})"
        return rep

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        evalloader = DataLoader(self.global_eval_data, batch_size=64, shuffle=True)

        with torch.no_grad():
            # load generator weights
            guide_model = VAE_CNN(z_dim=self.latent_dim, encoder_only=True).to(
                self.device
            )
            gen_params_dict = zip(
                self.gen_model.state_dict().keys(),
                parameters_to_ndarrays(self.initial_generator_params),
            )
            gen_state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in gen_params_dict}
            )
            guide_model.load_state_dict(gen_state_dict, strict=True)

            # local decorders
            temp_local_models = [
                VAE_CNN(z_dim=self.latent_dim).to(self.device)
                for _ in range(5)  # TODO: change to num_clients
            ]
            for eval_img, _ in evalloader:
                eval_img = eval_img.to(self.device)
                save_image(
                    eval_img.view(64, 1, 28, 28),
                    f"{self.folder_name}/eval_true_img.png",
                )
                for idx, temp_local_model in enumerate(temp_local_models):
                    params_dict = zip(
                        temp_local_model.state_dict().keys(),
                        parameters_to_ndarrays(
                            self.local_param_dict[f"client_param_{idx}"]
                        ),
                    )
                    state_dict = OrderedDict(
                        {k: torch.tensor(v) for k, v in params_dict}
                    )
                    temp_local_model.load_state_dict(state_dict, strict=True)
                    z_g, mu_g, logvar_g = guide_model(eval_img)
                    gen_img = temp_local_model.decoder(z_g)
                    save_image(
                        gen_img.view(64, 1, 28, 28),
                        f"{self.folder_name}/eval_gen_img_{idx}.png",
                    )
                break
        decoded_imgs = {
            f"global_eval_true_img": f"{self.folder_name}/eval_true_img.png",
            **{
                f"global_eval_gen_img_{idx}": f"{self.folder_name}/eval_gen_img_{idx}.png"
                for idx in range(5)
            },
        }
        # parameters = self.local_param_dict["client_param_0"]
        # TODO:chk the agg model with params
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        print(f"to compare with agg model: {parameters_ndarrays[7]}")
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {**decoded_imgs})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
            # config["gen_params"] = self.gen_stats
            config["gen_params"] = self.initial_generator_params
            config["local_param_dict"] = self.local_param_dict
            print(
                f'pre-train 0: {parameters_to_ndarrays(config["local_param_dict"]["client_param_0"])[7]}'
            )
            print(
                f'pre-train 1: {parameters_to_ndarrays(config["local_param_dict"]["client_param_1"])[7]}'
            )
            print(
                f'pre-train 2: {parameters_to_ndarrays(config["local_param_dict"]["client_param_2"])[7]}'
            )
            print(
                f'pre-train 3: {parameters_to_ndarrays(config["local_param_dict"]["client_param_3"])[7]}'
            )
            print(
                f'pre-train 4: {parameters_to_ndarrays(config["local_param_dict"]["client_param_4"])[7]}'
            )

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
            config["local_param_dict"] = self.local_param_dict
            print(
                f'post-train 0: {parameters_to_ndarrays(config["local_param_dict"]["client_param_0"])[7]}'
            )
            print(
                f'post-train 1: {parameters_to_ndarrays(config["local_param_dict"]["client_param_1"])[7]}'
            )
            print(
                f'post-train 2: {parameters_to_ndarrays(config["local_param_dict"]["client_param_2"])[7]}'
            )
            print(
                f'post-train 3: {parameters_to_ndarrays(config["local_param_dict"]["client_param_3"])[7]}'
            )
            print(
                f'post-train 4: {parameters_to_ndarrays(config["local_param_dict"]["client_param_4"])[7]}'
            )

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        temp_local_models = [
            VAE_CNN(z_dim=self.latent_dim).to(self.device)
            for _ in range(len(weights_results))
        ]
        # load generator weights
        gen_params_dict = zip(
            self.gen_model.state_dict().keys(),
            parameters_to_ndarrays(self.initial_generator_params),
        )
        gen_state_dict = OrderedDict({k: torch.tensor(v) for k, v in gen_params_dict})
        self.gen_model.load_state_dict(gen_state_dict, strict=True)
        optimizer = torch.optim.Adam(self.gen_model.parameters(), lr=self.lr_g)

        for temp_local_model in temp_local_models:
            temp_local_model.eval()

        for ep_g in range(self.steps_g):
            for step, (align_img, _) in enumerate(self.alignment_dataloader):
                loss = []

                optimizer.zero_grad()
                align_img = align_img.to(self.device)

                for idx, (weights, _) in enumerate(weights_results):
                    params_dict = zip(
                        temp_local_models[idx].state_dict().keys(), weights
                    )
                    state_dict = OrderedDict(
                        {k: torch.tensor(v) for k, v in params_dict}
                    )
                    temp_local_models[idx].load_state_dict(state_dict, strict=True)
                    z_g, mu_g, logvar_g = self.gen_model(align_img)

                    # loss_ = vae_loss(
                    #     temp_local_models[idx].decoder(z_g),
                    #     align_img,
                    #     mu_g,
                    #     logvar_g,
                    #     self.lambda_align_g,
                    # )
                    loss_ = vae_rec_loss(
                        temp_local_models[idx].decoder(z_g), align_img, cnn=True
                    )

                    loss.append(loss_)
                recon_loss = torch.stack(loss).mean(dim=0).mean()
                # KL divergence loss
                kld_loss = -0.5 * torch.sum(
                    1 + logvar_g - mu_g.pow(2) - logvar_g.exp(), dim=1
                )
                # Take the mean along dimension 0 (mean over images in the batch)
                kld_loss = torch.mean(kld_loss)

                total_loss = recon_loss + kld_loss * self.lambda_align_g

                threshold = 1e-6  # Define a threshold for the negligible loss
                log(DEBUG, f"guide update loss at ep {ep_g} step {step}: {total_loss}")

                if (
                    total_loss.item() > threshold
                ):  # Check if the total_loss is greater than the threshold
                    total_loss.backward()
                    optimizer.step()
                else:
                    log(
                        DEBUG,
                        f"Skipping optimization at step {step} due to negligible total_loss",
                    )
        guide_fig = visualize_plotly_latent_representation(
            self.gen_model,
            self.alignment_dataloader,
            self.device,
            use_PCA=False,
            num_class=self.num_classes,
        )
        wandb.log(
            data={
                f"final_guide_encoder_loss": total_loss.item(),
                "client_round": server_round,
                "guide_latent_space": guide_fig,
            },
            step=server_round,
        )

        self.initial_generator_params = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.gen_model.state_dict().items()]
        )
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        local_param_dict = {}
        for _, fit_res in results:
            local_param_dict[f"client_param_{fit_res.metrics['cid']}"] = (
                fit_res.parameters
            )
        self.local_param_dict = local_param_dict
        print(
            f'agg post 0: {parameters_to_ndarrays(self.local_param_dict["client_param_0"])[7]}'
        )
        print(
            f'agg post 1: {parameters_to_ndarrays(self.local_param_dict["client_param_1"])[7]}'
        )
        print(
            f'agg post 2: {parameters_to_ndarrays(self.local_param_dict["client_param_2"])[7]}'
        )
        print(
            f'agg post 3: {parameters_to_ndarrays(self.local_param_dict["client_param_3"])[7]}'
        )
        print(
            f'agg post 4: {parameters_to_ndarrays(self.local_param_dict["client_param_4"])[7]}'
        )
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]

        for fit_metric in fit_metrics:
            data = {}

            for key, value in fit_metric[1].items():
                if key in ["cid", "gen_image", "true_image", "latent_rep"]:
                    continue
                data = {
                    f"train_{key}_{fit_metric[1]['cid']}": value,
                }

            data[f"train_num_examples_{fit_metric[1]['cid']}"] = fit_metric[0]
            data[f"train_true_image_{fit_metric[1]['cid']}"] = wandb.Image(
                fit_metric[1]["true_image"]
            )
            data[f"train_gen_image_{fit_metric[1]['cid']}"] = wandb.Image(
                fit_metric[1]["gen_image"]
            )
            data[f"train_latent_rep_{fit_metric[1]['cid']}"] = fit_metric[1][
                "latent_rep"
            ]

            wandb.log(data=data, step=server_round)
        print(f"agg overall: {aggregate(weights_results)[7]}")
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        eval_metrics = [(res.num_examples, res.metrics) for _, res in results]

        for eval_metric in eval_metrics:
            data = {}
            for key, value in eval_metric[1].items():
                if key in ["cid"]:
                    continue
                data = {
                    f"eval_{key}_{eval_metric[1]['cid']}": value,
                }
            data[f"eval_num_examples_{eval_metric[1]['cid']}"] = eval_metric[0]

            wandb.log(data=data, step=server_round)
        data_agg = {
            f"eval_{key}_aggregated": value for key, value in metrics_aggregated.items()
        }
        wandb.log(
            data={
                "eval_loss_aggregated": loss_aggregated,
                **data_agg,
            },
            step=server_round,
        )
        return loss_aggregated, metrics_aggregated
