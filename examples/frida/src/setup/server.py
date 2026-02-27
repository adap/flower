from typing import List, Tuple
from flwr.common import Metrics, FitIns
from typing import List, Tuple
from typing import Union, Optional, Dict
import flwr as fl

from flwr.common import (
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from .model import get_parameters, set_parameters

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from .attacks import (
    YeomAttack,
    YeomAttackEfficient,
    CosineAttack,
    CosineAttackEfficient,
    DAGMMAttack,
    CosineSimilarityAttack,
    STDAttack,
    L2NormAttack,
    DistScoreAttack,
    InconsistencyAttack,
    STDDAGMMAttack,
)

from time import time
from .data_loader import load_attack_sets_from_noise

import wandb
import torch
from copy import deepcopy
import numpy as np
import random
import psutil
import tracemalloc


def check_config(cfg, attack_types):
    if "cosine" not in attack_types and "yeom" not in attack_types:
        cfg.canary = False
        cfg.noise = False
    else:
        if cfg.noise:
            if not cfg.canary and not cfg.noise:
                raise ValueError("ALERT: Canary and noise are both false!")
        else:
            cfg.canary = True
            cfg.dynamic_canary = True
            cfg.single_training = False

    if cfg.dataset == "cifar100":
        cfg.num_classes = 100
        cfg.image_label = "fine_label"
        cfg.input_size = 32

    elif cfg.dataset == "cifar10":
        cfg.num_classes = 10
        cfg.image_label = "label"
        cfg.input_size = 32

    return cfg


def get_evaluate_fn(model, valloader, device, cfg):
    """Return an evaluation function for server-side evaluation."""
    valloader = random.choice(valloader)

    def evaluate_fn(server_round, parameters, config):
        set_parameters(model, parameters)  # Update model with the latest parameters
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        model.eval().to(device)

        with torch.no_grad():
            for batch in valloader:
                
                if cfg.dataset == "shakespeare":
                    inputs, labels = (
                        batch[cfg.text_input].to(device),
                        batch[cfg.text_label].to(device),
                    )
                    
                    outputs = model(inputs)
                    
                    batch_loss = criterion(outputs, labels)
                    loss += batch_loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                else:
                    images, labels = (
                        batch[cfg.image_name].to(device),
                        batch[cfg.image_label].to(device),
                    )
                    
                    outputs = model(images)
                    
                    batch_loss = criterion(outputs, labels)
                    loss += batch_loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        if total == 0:
            print("Warning: Evaluation finished with total=0. Valloader might be empty.")
            return 0.0, {"accuracy": 0.0, "loss": 0.0}

        loss /= total
        accuracy = correct / total

        try:
            wandb.log(
                {"server_accuracy": accuracy, "server_loss": loss},
            )
        except Exception as e:
            print(f"Wandb logging failed (server evaluation): {e}")

        return loss, {"accuracy": accuracy, "loss": loss}

    return evaluate_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"weighted_accuracy": sum(accuracies) / sum(examples)}


class PrivacyAttacksForDefense(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Parameters,
        evaluate_fn=None,
        training_datasets=[],
        validation_datasets=[],
        subsets=[],
        net,
        freeriders,
        cfg,
        device="cpu",
        proximal_mu=0.1,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
        )
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.validation_datasets = validation_datasets
        self.training_datasets = training_datasets
        self.device = device
        self.attack_types = cfg.attack_types
        self.net = net
        self.freeriders = freeriders
        self.average_rounds = cfg.average_rounds
        self.cfg = cfg
        self.prev_parameters = None
        self.prev_prev_parameters = None
        self.prev_grads = None
        self.params_0 = None
        self.params_1 = None
        self.history_detection_results = []
        self.cfg = check_config(self.cfg, self.attack_types)

        self.round_metrics = []
        self.total_runtime = 0
        self.total_attack_time = 0
        self.total_aggregation_time = 0
        
        self.attacks = {}

        if self.training_datasets:
            first_batch = next(iter(self.training_datasets[0]))
            if cfg.dataset == "shakespeare":
                input_key = cfg.text_input
                input_shape = first_batch[input_key].shape[1:] 
            else:
                input_key = cfg.image_name
                input_shape = first_batch[input_key].shape[1:] # e.g., (3, 32, 32)
        else:
            input_shape = (0,) 

        for attack_type in self.attack_types:
            if attack_type == "yeom":
                attack = YeomAttack(self.net, training_datasets, validation_datasets, device, subsets, cfg)

            elif attack_type == "cosine":
                self.gamma = cfg.gamma
                attack = CosineAttack(self.net, validation_datasets, device, subsets, cfg)

            elif attack_type == "stddagmm":
                cfg.n_encoder_layers.insert(0, sum(p.numel() for p in self.net.parameters() if p.requires_grad))
                attack = STDDAGMMAttack(cfg.n_encoder_layers, cfg.n_gmm_layers, device, cfg.dagmm_batch_size, cfg.dagmm_epochs)

            elif attack_type == "dagmm":
                cfg.n_encoder_layers.insert(0, sum(p.numel() for p in self.net.parameters() if p.requires_grad))
                attack = DAGMMAttack(cfg.n_encoder_layers, cfg.n_gmm_layers, device, cfg.dagmm_batch_size, cfg.dagmm_epochs)

            elif attack_type == "cosine_similarity":
                attack = CosineSimilarityAttack()

            elif attack_type == "std":
                attack = STDAttack()

            elif attack_type == "l2":
                attack = L2NormAttack()
            
            elif attack_type == "dist_score":
                attack = DistScoreAttack(
                    device,
                    load_attack_sets_from_noise(
                        input_shape, cfg.num_classes, cfg.sample_per_label, cfg.decaf_batch_size, cfg
                    ),
                )
            elif attack_type == "inconsistency":
                attack = InconsistencyAttack(
                    device,
                    load_attack_sets_from_noise(
                        input_shape, cfg.num_classes, cfg.sample_per_label, cfg.decaf_batch_size, cfg
                    ),
                )
            else:
                print("[ERROR] No attack.")
            
            self.attacks[attack_type] = attack


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

        config_start_time = time()
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        if server_round > 1:
            sum_abs_diff = 0
            abs_diffs = []
            for tens_A, tens_B in zip(parameters_to_ndarrays(parameters), self.prev_parameters):
                sum_abs_diff += torch.sum(torch.abs(torch.tensor(tens_A) - torch.tensor(tens_B)))
                abs_diffs.append(torch.abs(torch.tensor(tens_A) - torch.tensor(tens_B)))
            num_params = sum([np.prod(tensor.size()) for tensor in list(self.net.parameters())])
            std_noise = float(sum_abs_diff / num_params)
        
        else:
            std_noise = self.cfg.std
            std_noise_vector = []
            for tens in parameters_to_ndarrays(parameters):
                std_noise_vector.append(torch.ones(tens.shape) * self.cfg.std)

        standard_config = {"server_round": server_round, "std_noise": std_noise}

        if self.cfg.freerider_type == "gradient_noiser" or self.cfg.freerider_type == "advanced" or self.cfg.freerider_type == "gradient_noiser_old":
            standard_config["prev_params"] = self.prev_parameters
            standard_config["prev_prev_params"] = self.prev_prev_parameters
            standard_config["params0"] = self.params_0
            standard_config["params1"] = self.params_1
            # standard_config["all_prev_params"] = self.all_prev_params

        fit_configurations = []
        for idx, client in enumerate(clients):
            fit_configurations.append((client, FitIns(parameters, standard_config)))

        # Track configuration time
        config_time = time() - config_start_time
        
        print(f"\n{'='*60}")
        print(f"FL ROUND {server_round} - CONFIGURATION PHASE")
        print(f"{'='*60}")
        print(f"Number of clients selected: {len(clients)}")
        print(f"Configuration time: {config_time:.3f} seconds")
        
        if server_round == 1:
            self.params_0 = parameters_to_ndarrays(parameters)
        elif server_round == 2:
            self.params_1 = parameters_to_ndarrays(parameters)

        self.prev_prev_parameters = self.prev_parameters
        self.prev_parameters = parameters_to_ndarrays(parameters)
        if self.prev_parameters is not None and self.prev_prev_parameters is not None:
            globalmodel = torch.cat([torch.flatten(param) for param in [torch.tensor(param, device=self.device) for param in self.prev_parameters]])
            prev = torch.cat([torch.flatten(param) for param in [torch.tensor(param, device=self.device) for param in self.prev_prev_parameters]])
            l2_norm = torch.norm(globalmodel-prev, p=2)

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average and get Yeom probabilities."""
        
        tracemalloc.start()
        round_start_time = time()
        
        aggregation_start_time = time()
        
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), int(fit_res.metrics["cid"])) for _, fit_res in results]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        
        aggregation_time = time() - aggregation_start_time
        
        weights_results = sorted(weights_results, key=lambda x: x[1])
        num_examples = [(fit_res.num_examples, int(fit_res.metrics["cid"])) for _, fit_res in results]
        num_examples.sort(key=lambda x: int(x[1]))
        num_examples = [x[0] for x in num_examples]
        
        attack_start_time = time()
        self.net.train()
        
        print(f"\n{'='*60}")
        print(f"FL ROUND {server_round} - ATTACK DETECTION PHASE")
        print(f"{'='*60}")
 
        attack_results = {}
        individual_attack_times = {}
        original_state = get_parameters(self.net)

        for attack_type in self.attack_types:
            single_attack_start = time()
            attack_results[attack_type] = {}
            set_parameters(self.net, original_state)

            if attack_type == "yeom":
                detection_results, detection_metrics = self.attacks[attack_type].do_attack(
                    self.net, weights_results, self.cfg.zscore_threshold, server_round
                )
            elif attack_type == "cosine":
                detection_results, detection_metrics = self.attacks[attack_type].do_attack(
                    self.net, weights_results, server_round
                )
            elif attack_type == "dagmm":
                detection_results, detection_metrics = self.attacks[attack_type].do_attack(
                    get_parameters(self.net), weights_results, self.cfg.zscore_threshold
                )
            elif attack_type == "stddagmm":
                detection_results, detection_metrics = self.attacks[attack_type].do_attack(
                    get_parameters(self.net), weights_results, self.cfg.zscore_threshold
                )
            elif attack_type == "l2" or attack_type == "std" or attack_type == "cosine_similarity":
                detection_results, detection_metrics = self.attacks[attack_type].do_attack(
                    self.net, weights_results, self.cfg.zscore_threshold
                )
            elif attack_type in ("inconsistency", "dist_score"):
                detection_results, detection_metrics, pia_results = self.attacks[attack_type].do_attack(
                    self.net, self.prev_parameters, weights_results, num_examples, self.cfg, self.cfg.zscore_threshold
                )
                attack_results[attack_type]["pia_results"] = pia_results
            else:
                print("[ERROR]: unknown attack")
                continue
        
            individual_attack_times[attack_type] = time() - single_attack_start
            print(f"  {attack_type.upper()} attack execution time: {individual_attack_times[attack_type]:.3f} seconds")
            
            attack_results[attack_type]["detection_results"] = detection_results
            attack_results[attack_type]["detection_metrics"] = detection_metrics

        total_attack_time = time() - attack_start_time

        # Combine different detection results if more than one attack
        if len(self.attack_types) > 1:
            combined_detection_results = []
            for client in range(len(detection_results)):
                count = 0
                for attack_type in self.attack_types:
                    detection_results = attack_results[attack_type]["detection_results"]
                    if detection_results[client]:
                        count += 1
                combined_detection_results.append(count >= len(self.attack_types)/2)
            self.history_detection_results.append(combined_detection_results)
        else:
            self.history_detection_results.append(detection_results)
            
        print(f"\nFree-riders detected: {detection_results}")
        print(f"Total attack detection time: {total_attack_time:.3f} seconds")

        # Mitigation phase timing
        mitigation_start_time = time()
        
        if self.cfg.mitigation:
            parameters_aggregated = ndarrays_to_parameters(
                aggregate([weight_result for (detection, weight_result) in zip(detection_results, weights_results) if not detection])
            )
        
        mitigation_time = time() - mitigation_start_time

        # Calculate metrics
        freerider_metrics = self.attacks[self.attack_types[0]].get_metrics(detection_results, self.freeriders)
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate total round time
        total_round_time = time() - round_start_time
        self.total_runtime += total_round_time
        self.total_attack_time += total_attack_time
        self.total_aggregation_time += aggregation_time
        
        # Print comprehensive timing information
        print(f"\n{'='*60}")
        print(f"FL ROUND {server_round} - PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Aggregation time: {aggregation_time:.3f} seconds")
        print(f"Attack detection time: {total_attack_time:.3f} seconds")
        if self.cfg.mitigation:
            print(f"Mitigation time: {mitigation_time:.3f} seconds")
        if self.cfg.average_detection > 1:
            print(f"Max voting time: {maxvoting_time:.3f} seconds")
        print(f"Total round time: {total_round_time:.3f} seconds")
        print(f"Memory usage: {current / 10**6:.2f} MB (peak: {peak / 10**6:.2f} MB)")
        # print(f"CPU usage: {psutil.cpu_percent()}%") # psutil import required
        print(f"\nDetection Performance:")
        print(f"  Accuracy: {freerider_metrics['Accuracy']:.4f}")
        print(f"  Precision: {freerider_metrics['Precision']:.4f}")
        print(f"  Recall: {freerider_metrics['Recall']:.4f}")
        print(f"  F1 Score: {freerider_metrics['F1 Score']:.4f}")
        print(f"  FPR: {freerider_metrics['False Positive Rate']:.4f}")
        print(f"  FNR: {freerider_metrics['False Negative Rate']:.4f}")
        
        # Store round metrics for later analysis
        round_metrics = {
            "round": server_round,
            "aggregation_time": aggregation_time,
            "attack_detection_time": total_attack_time,
            "individual_attack_times": individual_attack_times,
            "mitigation_time": mitigation_time if self.cfg.mitigation else 0,
            "maxvoting_time": maxvoting_time if self.cfg.average_detection > 1 else 0,
            "total_round_time": total_round_time,
            "memory_usage_mb": current / 10**6,
            "peak_memory_mb": peak / 10**6,
            # "cpu_percent": psutil.cpu_percent(),
            "num_clients": len(weights_results),
            "freeriders_detected": sum(detection_results),
        }
        self.round_metrics.append(round_metrics)

        if "yeom" in self.attack_types:
            freerider_metrics["detection_loss"] = attack_results["yeom"]["detection_metrics"]

        if "cosine" in self.attack_types:
            freerider_metrics["cosine_metrics"] = attack_results["cosine"]["detection_metrics"]

        if "l2" in self.attack_types:
            freerider_metrics["l2_metrics"] = attack_results["l2"]["detection_metrics"]

        if "std" in self.attack_types:
            freerider_metrics["std_metrics"] = attack_results["std"]["detection_metrics"]

        if "cosine_similarity" in self.attack_types:
            freerider_metrics["cosim_metrics"] = attack_results["cosine_similarity"]["detection_metrics"]
        
        if any(k in self.attack_types for k in ("dist_score", "inconsistency")):
            freerider_metrics["pia"] = (
                (("dist_score" in self.attack_types) and (attack_results.get("dist_score") or {}).get("pia_results"))
                or (("inconsistency" in self.attack_types) and (attack_results.get("inconsistency") or {}).get("pia_results"))
            )

        freerider_metrics["historical_detection_results"] = self.history_detection_results
        freerider_metrics["round_metrics"] = round_metrics

        set_parameters(self.net, parameters_to_ndarrays(parameters_aggregated))

        if server_round % 10 == 0:
            self.print_cumulative_stats(server_round)

        return parameters_aggregated, freerider_metrics
    

    def print_cumulative_stats(self, server_round):
        """Print cumulative statistics every 10 rounds"""
        print(f"\n{'='*60}")
        print(f"CUMULATIVE STATISTICS (Rounds 1-{server_round})")
        print(f"{'='*60}")
        print(f"Total runtime: {self.total_runtime:.2f} seconds")
        print(f"Average time per round: {self.total_runtime/server_round:.3f} seconds")
        print(f"Total attack detection time: {self.total_attack_time:.2f} seconds")
        print(f"Average attack time per round: {self.total_attack_time/server_round:.3f} seconds")
        print(f"Total aggregation time: {self.total_aggregation_time:.2f} seconds")
        print(f"Average aggregation time per round: {self.total_aggregation_time/server_round:.3f} seconds")
        
        if self.round_metrics:
            avg_memory = sum(m["memory_usage_mb"] for m in self.round_metrics) / len(self.round_metrics)
            max_memory = max(m["peak_memory_mb"] for m in self.round_metrics)
            print(f"Average memory usage: {avg_memory:.2f} MB")
            print(f"Peak memory usage: {max_memory:.2f} MB")
            
            if self.attack_types and "individual_attack_times" in self.round_metrics[0]:
                print(f"\nPer-Attack Average Times:")
                for attack_type in self.attack_types:
                    avg_time = sum(
                        m["individual_attack_times"].get(attack_type, 0) 
                        for m in self.round_metrics
                    ) / len(self.round_metrics)
                    print(f"  {attack_type.upper()}: {avg_time:.3f} seconds")
        
        print(f"{'='*60}\n")


class PrivacyAttacksForDefenseFedProx(PrivacyAttacksForDefense):
    """FedProx version of PrivacyAttacksForDefense strategy."""
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Parameters,
        evaluate_fn=None,
        training_datasets=[],
        validation_datasets=[],
        subsets=[],
        net,
        freeriders,
        cfg,
        device="cpu",
        proximal_mu=0.1,  # FedProx parameter
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            training_datasets=training_datasets,
            validation_datasets=validation_datasets,
            subsets=subsets,
            net=net,
            freeriders=freeriders,
            cfg=cfg,
            device=device,
        )
        self.proximal_mu = proximal_mu

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure fit with proximal_mu parameter."""
        
        config_start_time = time()
        print("Configure fit starts")
        
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        if server_round > 1:
            sum_abs_diff = 0
            abs_diffs = []
            for tens_A, tens_B in zip(parameters_to_ndarrays(parameters), self.prev_parameters):
                sum_abs_diff += torch.sum(torch.abs(torch.tensor(tens_A) - torch.tensor(tens_B)))
                abs_diffs.append(torch.abs(torch.tensor(tens_A) - torch.tensor(tens_B)))
            num_params = sum([np.prod(tensor.size()) for tensor in list(self.net.parameters())])
            std_noise = float(sum_abs_diff / num_params)
        else:
            std_noise = self.cfg.std
            std_noise_vector = []
            for tens in parameters_to_ndarrays(parameters):
                std_noise_vector.append(torch.ones(tens.shape) * self.cfg.std)

        standard_config = {
            "server_round": server_round, 
            "std_noise": std_noise,
            "proximal_mu": self.proximal_mu  # Add this line
        }

        if self.cfg.freerider_type == "gradient_noiser" or self.cfg.freerider_type == "advanced" or self.cfg.freerider_type == "gradient_noiser_old":
            standard_config["prev_params"] = self.prev_parameters
            standard_config["prev_prev_params"] = self.prev_prev_parameters
            standard_config["params0"] = self.params_0
            standard_config["params1"] = self.params_1

        fit_configurations = []
        for idx, client in enumerate(clients):
            fit_configurations.append((client, FitIns(parameters, standard_config)))

        # Track configuration time
        config_time = time() - config_start_time
        
        print(f"\n{'='*60}")
        print(f"FL ROUND {server_round} - CONFIGURATION PHASE (FedProx μ={self.proximal_mu})")
        print(f"{'='*60}")
        print(f"Number of clients selected: {len(clients)}")
        print(f"Configuration time: {config_time:.3f} seconds")
        
        if server_round == 1:
            self.params_0 = parameters_to_ndarrays(parameters)
        elif server_round == 2:
            self.params_1 = parameters_to_ndarrays(parameters)

        self.prev_prev_parameters = self.prev_parameters
        self.prev_parameters = parameters_to_ndarrays(parameters)
        
        if self.prev_parameters is not None and self.prev_prev_parameters is not None:
            globalmodel = torch.cat([torch.flatten(param) for param in [torch.tensor(param, device=self.device) for param in self.prev_parameters]])
            prev = torch.cat([torch.flatten(param) for param in [torch.tensor(param, device=self.device) for param in self.prev_prev_parameters]])
            l2_norm = torch.norm(globalmodel-prev, p=2)

        return fit_configurations


