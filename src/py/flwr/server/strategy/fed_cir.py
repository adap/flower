from logging import WARNING, DEBUG
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.utils import Generator, ResNet18, VAE, CVAE
from .aggregate import aggregate, weighted_loss_avg
from .fedavg import FedAvg
import wandb
import matplotlib

matplotlib.use("Agg")

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""
import torch
import torch.nn as nn
from collections import OrderedDict


def vae_loss(recon_img, img, mu, logvar, beta=1.0):
    # Reconstruction loss using binary cross-entropy
    condition = (recon_img >= 0.0) & (recon_img <= 1.0)
    # assert torch.all(condition), "Values should be between 0 and 1"
    if not torch.all(condition):
        ValueError("Values should be between 0 and 1")
        recon_img = torch.clamp(recon_img, 0.0, 1.0)
    recon_loss = F.binary_cross_entropy(
        recon_img, img.view(-1, img.shape[2] * img.shape[3]), reduction="sum"
    )

    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total VAE loss
    total_loss = recon_loss + kld_loss * beta

    return total_loss


class FedCiR(FedAvg):
    """Configurable FedCiRs strategy implementation."""

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
        initial_generator_params: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        lr_g=1e-3,
        steps_g=10,
        gen_stats=None,
        num_classes=10,
        alignment_dataloader=None,
        device=None,
        lambda_align_g=1.0,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`,
            `min_evaluate_clients` will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.initial_generator_params = initial_generator_params
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.lr_gen = lr_g
        self.num_classes = num_classes
        self.steps_gen = steps_g
        self.gen_stats = gen_stats
        self.device = device
        self.gen_model = VAE(encoder_only=True).to(self.device)
        self.alignment_loader = alignment_dataloader
        self.ref_mu, self.ref_logvar = self.compute_ref_stats()
        self.lambda_align_g = lambda_align_g

    def compute_ref_stats(self):
        ref_model = CVAE(z_dim=2).to(self.device)
        opt_ref = torch.optim.Adam(ref_model.parameters(), lr=1e-3)
        for ep in range(5000):
            for images, labels in self.alignment_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                opt_ref.zero_grad()
                recon_images, mu, logvar = ref_model(images, labels)
                vae_loss1 = vae_loss(recon_images, images, mu, logvar, 0.01)
                vae_loss1.backward()
                opt_ref.step()
            if ep % 100 == 0:
                log(DEBUG, f"Epoch {ep}, Loss {vae_loss1.item()}")

                log(DEBUG, f"--------------------------------------------------")
        ref_model.eval()
        with torch.no_grad():
            for images, labels in self.alignment_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                _, ref_mu, ref_logvar = ref_model(images, labels)
        return ref_mu, ref_logvar

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedCiRs(accept_failures={self.accept_failures})"
        return rep

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # config = {
        #     "z_g": self.gen_stats.tensors[0],
        #     "mu_g": self.gen_stats.tensors[1],
        #     "log_var_g": self.gen_stats.tensors[2],
        # }
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
            config["gen_params"] = self.gen_stats
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
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        def vae_loss_connect(recon_img, img, mu, logvar, mu_ref, logvar_ref):
            # Reconstruction loss using binary cross-entropy
            condition = (recon_img >= 0.0) & (recon_img <= 1.0)
            # assert torch.all(condition), "Values should be between 0 and 1"
            if not torch.all(condition):
                ValueError("Values should be between 0 and 1")
                recon_img = torch.clamp(recon_img, 0.0, 1.0)
            recon_loss = F.binary_cross_entropy(
                recon_img, img.view(-1, img.shape[2] * img.shape[3]), reduction="sum"
            )

            # KL divergence loss
            loss_align = 0.5 * (logvar_ref - logvar - 1) + (
                logvar.exp() + (mu - mu_ref).pow(2)
            ) / (2 * logvar_ref.exp())
            loss_align_reduced = loss_align.sum(dim=1).sum()
            # Total VAE loss
            total_loss = self.lambda_align_g * loss_align_reduced + recon_loss

            return total_loss

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        temp_local_models = [VAE().to(self.device) for _ in range(len(weights_results))]
        optimizer = torch.optim.Adam(self.gen_model.parameters(), lr=self.lr_gen)

        for temp_local_model in temp_local_models:
            temp_local_model.eval()

        for ep_g in range(self.steps_gen):
            for step, (align_img, _) in enumerate(self.alignment_loader):
                preds = []
                # mu_s = []
                # logvar_s = []
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
                    preds.append(temp_local_models[idx].decoder(z_g))
                    # _, mu, logvar = temp_local_models[idx](align_img)
                    # mu_s.append(mu)
                    # logvar_s.append(logvar)

                # loss = vae_loss(
                #     torch.stack(preds).mean(dim=0),
                #     align_img,
                #     mu_g,
                #     logvar_g,
                # )
                loss = vae_loss_connect(
                    torch.stack(preds).mean(dim=0),
                    align_img,
                    mu_g,
                    logvar_g,
                    self.ref_mu,
                    self.ref_logvar,
                    # torch.stack(mu_s).mean(dim=0),
                    # torch.stack(logvar_s).mean(dim=0),
                )
                threshold = 1e-6  # Define a threshold for the negligible loss
                log(DEBUG, f"generator loss at ep {ep_g} step {step}: {loss}")

                if (
                    loss.item() > threshold
                ):  # Check if the loss is greater than the threshold
                    loss.backward(retain_graph=True)
                    optimizer.step()
                else:
                    log(
                        DEBUG,
                        f"Skipping optimization at step {step} due to negligible loss",
                    )
        wandb.log(data={f"final_gen_loss": loss.item(), "client_round": server_round})
        # z_g, mu_g, log_var_g = self.gen_model(one_hot_all_labels)
        self.gen_stats = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.gen_model.state_dict().items()]
        )
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]

        for fit_metric in fit_metrics:
            data = {
                f"train_num_examples_{fit_metric[1]['cid']}": fit_metric[0],
                f"train_vae_loss_term_{fit_metric[1]['cid']}": fit_metric[1][
                    "vae_loss_term"
                ],
                f"train_reg_term_{fit_metric[1]['cid']}": fit_metric[1]["reg_term"],
                f"train_align_term_{fit_metric[1]['cid']}": fit_metric[1]["align_term"],
                f"train_true_image_{fit_metric[1]['cid']}": wandb.Image(
                    fit_metric[1]["true_image"]
                ),
                f"train_gen_image_{fit_metric[1]['cid']}": wandb.Image(
                    fit_metric[1]["gen_image"]
                ),
                f"train_latent_rep_{fit_metric[1]['cid']}": wandb.Image(
                    fit_metric[1]["latent_rep"]
                ),
                "client_round": fit_metric[1]["client_round"],
            }

            wandb.log(data)
            plt.close("all")
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        eval_metrics = [(res.num_examples, res.metrics) for _, res in results]

        for eval_metric in eval_metrics:
            data = {
                f"eval_num_examples_{eval_metric[1]['cid']}": eval_metric[0],
                f"eval_accuracy_{eval_metric[1]['cid']}": eval_metric[1]["accuracy"],
                f"eval_local_val_loss_{eval_metric[1]['cid']}": eval_metric[1][
                    "local_val_loss"
                ],
                "client_round": server_round,
            }

            wandb.log(data)

        return loss_aggregated, metrics_aggregated
