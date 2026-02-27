import random
from copy import deepcopy
from time import time

import numpy as np
import scipy
import scipy.stats
import torch
import torch.nn.functional as F
from scipy import spatial
from torch.utils.data import DataLoader, TensorDataset

from .dagmm import (
    DAGMM,
    STDDAGMM,
    get_energy,
    get_gmm_parameters,
    train_dagmm,
    train_std_dagmm,
)
from .model import get_parameters, set_parameters
from .old_pia import old_pia
from .train import test, train


@torch.no_grad()
def set_parameters_inplace(net, parameters):
    for p, v in zip(net.parameters(), parameters):
        if not torch.is_tensor(v):
            v = torch.from_numpy(v)
        p.copy_(v.to(device=p.device, dtype=p.dtype, non_blocking=True))


class YeomAttack:
    def __init__(
        self, net, training_datasets, validation_datasets, device, subsets, cfg
    ):
        self.training_datasets = training_datasets
        self.validation_datasets = validation_datasets
        self.canary = subsets
        self.device = device
        self.net = net.to(device)
        self.cfg = cfg

    def do_attack(self, parameters, weight_results, threshold=1.0, server_round=None):
        self.net = parameters
        detection_results = []
        detection_losses = []

        if self.cfg.yeom_type == "serverside":
            if self.cfg.dynamic_canary:
                canary = self.canary[server_round - 1]
            else:
                canary = self.canary

            aux_net = deepcopy(self.net)
            train(
                aux_net,
                canary,
                self.device,
                self.cfg,
                epochs=self.cfg.server_epochs,
                verbose=True,
            )

            global_loss, _ = test(aux_net, canary, self.device, self.cfg)
            canary_losses = [
                self._evaluate_client(weights[0], canary)[0]
                for weights in weight_results
            ]

            for i, weight_result in enumerate(weight_results):
                detection_result, detection_loss = self._detect_validation(
                    global_loss, canary_losses[i]
                )
                detection_losses.append([weight_result[1], detection_loss])
                detection_results.append([weight_result[1], detection_result])

        elif self.cfg.yeom_type == "clientside":
            if self.cfg.dynamic_canary:
                canary = self.canary[server_round - 1]
            else:
                canary = self.canary
            canary_losses = [
                self._evaluate_client(weights[0], canary)[0]
                for weights in weight_results
            ]
            for i, weight_result in enumerate(weight_results):
                detection_result, detection_loss = self._detect(
                    canary_losses[i], canary_losses, threshold
                )
                detection_losses.append([weight_result[1], detection_loss])
                detection_results.append([weight_result[1], detection_result])

        detection_results.sort(key=lambda x: int(x[0]))
        detection_losses.sort(key=lambda x: int(x[0]))

        return [x[1] for x in detection_results], [x[1] for x in detection_losses]

    def _evaluate_client(self, parameters, dataset):
        aux_net = deepcopy(self.net)
        set_parameters(aux_net, parameters)
        loss, acc = test(aux_net, dataset, self.device, self.cfg)
        return loss, acc

    def _detect_validation(self, global_val_loss, canary_loss):
        if canary_loss > global_val_loss:
            decision = True
        else:
            decision = False

        return decision, canary_loss

    def _detect(self, local_loss, canary_losses, threshold):
        mean = np.mean(canary_losses)
        std = np.std(canary_losses)
        z_score = (local_loss - mean) / std

        if abs(z_score) > threshold:
            decision = True
        else:
            decision = False

        return decision, local_loss

    def get_metrics(self, detection_results, freeriders):
        return compute_metrics(detection_results, freeriders)


def compute_metrics(detection_results, freeriders):
    detection_results = np.array(detection_results)
    freeriders = np.array(freeriders)

    true_positives = np.sum(np.logical_and(detection_results, freeriders))
    false_positives = np.sum(
        np.logical_and(detection_results, np.logical_not(freeriders))
    )
    true_negatives = np.sum(
        np.logical_and(np.logical_not(detection_results), np.logical_not(freeriders))
    )
    false_negatives = np.sum(
        np.logical_and(np.logical_not(detection_results), freeriders)
    )

    accuracy = (true_positives + true_negatives) / len(freeriders)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0.0
    )
    fnr = (
        false_negatives / (false_negatives + true_positives)
        if (false_negatives + true_positives) > 0
        else 0.0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr,
    }


class CosineAttack:
    def __init__(self, net, validation_datasets, device, canary, cfg):

        self.validation_datasets = validation_datasets
        self.device = device
        self.net = net
        self.cfg = cfg
        self.canary = canary

        self.gamma = cfg.gamma
        self.name_layer_grads = cfg.name_layer_grads
        self.average = cfg.average_rounds
        self.cosine_members = []
        self.cosine_nonmembers = []

    def do_attack(self, global_model, weight_results, server_round=None):
        grads_clients = [
            [w[1], compute_update(global_model, w[0], self.name_layer_grads)]
            for w in weight_results
        ]
        grads_clients.sort(key=lambda x: int(x[0]))
        if self.cfg.dynamic_canary:
            canary = self.canary[server_round - 1]
        else:
            canary = self.canary
        canary = DataLoader(
            canary.dataset, batch_size=self.cfg.canary_batch_size, shuffle=True
        )

        time_start_nonmember = time()
        distributions_nonmember = []
        for client_id in range(len(grads_clients)):
            validation_subset = self.validation_datasets[client_id]
            validation_subset = DataLoader(
                validation_subset.dataset,
                batch_size=self.cfg.canary_batch_size,
                shuffle=True,
            )
            distribution_nonmember = self._attack(
                global_model, validation_subset, grads_clients[client_id][1]
            )
            distributions_nonmember.append([client_id, distribution_nonmember])
        distributions_nonmember.sort(key=lambda x: int(x[0]))

        print(f"Time dist nonmember: {(time() - time_start_nonmember):.2f} seconds")

        # compute the distribution of cosine similarity for each client on member data
        time_start_member = time()
        distributions_member = []
        for client_id in range(len(grads_clients)):
            # distribution_member = self._attack(global_model, self.training_datasets[client_id], grads_clients[client_id][1])
            distribution_member = self._attack(
                global_model, canary, grads_clients[client_id][1]
            )
            distributions_member.append([client_id, distribution_member])
        distributions_member.sort(key=lambda x: int(x[0]))

        distributions_member = [x[1] for x in distributions_member]
        self.cosine_members.append(distributions_member)
        distributions_nonmember = [x[1] for x in distributions_nonmember]
        self.cosine_nonmembers.append(distributions_nonmember)

        print(f"Time dist member: {(time() - time_start_member):.2f} seconds")

        detection_results = []
        for client_id in range(len(distributions_nonmember)):
            if self.average > 1:
                distr_nonmember = self._average_distributions(
                    self.cosine_nonmembers, client_id
                )
                distr_member = self._average_distributions(
                    self.cosine_members, client_id
                )
            else:
                distr_nonmember = distributions_nonmember[client_id]
                distr_member = distributions_member[client_id]

            stat, pvalue = scipy.stats.ttest_ind(
                distr_nonmember, distr_member, equal_var=False
            )
            if pvalue < 0.05:
                detection_results.append([client_id, False])

            else:
                detection_results.append([client_id, True])
            print("client {} -- stat: {}, pvalue: {}".format(client_id, stat, pvalue))

        return [x[1] for x in detection_results], (
            distributions_member,
            distributions_nonmember,
        )

    def _attack(self, model, dataset, grad_client):
        model.eval()
        if isinstance(grad_client, torch.Tensor):
            grad_client_cpu = grad_client.detach().to("cpu", copy=True).float()
        else:
            grad_client_cpu = torch.as_tensor(grad_client, dtype=torch.float32)

        gc_norm = torch.norm(grad_client_cpu)
        if gc_norm > 0:
            grad_client_cpu = grad_client_cpu / gc_norm

        all_cos_sims = []
        total_samples = 0

        model.to(self.device)

        for batch in dataset:
            if total_samples >= self.cfg.subset_samples:
                break

            input_key = (
                self.cfg.text_input
                if self.cfg.dataset == "shakespeare"
                else self.cfg.image_name
            )
            label_key = (
                self.cfg.text_label
                if self.cfg.dataset == "shakespeare"
                else self.cfg.image_label
            )
            inputs = batch[input_key].to(self.device, non_blocking=True)
            labels = batch[label_key].to(self.device, non_blocking=True)

            batch_grads = self._compute_batch_gradients(model, inputs, labels)

            with torch.no_grad():
                for g in batch_grads:
                    g_cpu = g.detach().to("cpu", copy=True).float()
                    n = g_cpu.norm()
                    if n > 0:
                        g_cpu.div_(n)
                    all_cos_sims.append(float(torch.dot(g_cpu, grad_client_cpu)))
                    total_samples += 1
                    if total_samples >= self.cfg.subset_samples:
                        break

            del batch_grads, inputs, labels
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.tensor(all_cos_sims, dtype=torch.float32)  # stays on CPU

    def _average_distributions(self, distributions, client_id):
        num_epochs = min(self.average, len(distributions))
        tensors_to_average = [
            distributions[-i][client_id].numpy() for i in range(1, num_epochs + 1)
        ]
        average_distribution = np.mean(tensors_to_average, axis=0)
        return average_distribution

    def _compute_batch_gradients(self, model, inputs, labels):
        model.train()
        batch_grads = []

        for i in range(inputs.size(0)):
            model.zero_grad()
            output = model(inputs[i].unsqueeze(0))
            loss = F.cross_entropy(output, labels[i].unsqueeze(0))
            loss.backward()

            sample_grads = []
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    sample_grads.append(param.grad.view(-1))

            batch_grads.append(torch.cat(sample_grads))

        model.zero_grad()

        return torch.stack(batch_grads)

    def _compute_gradients_labels(self, model, x, num_labels):
        grads = []
        for label in range(num_labels):
            y = torch.tensor([label] * x.size(0))  # Fake label tensor
            grads.append(self._compute_gradients(model, x, y))

        return torch.sum(torch.stack(grads), dim=0)

    def _evaluate_client(self, parameters, subset):
        set_parameters(self.net, parameters)
        dataset = subset if subset else self.training_datasets
        loss, acc = test(self.net, dataset, self.device, self.cfg)
        return loss, acc

    def get_metrics(self, detection_results, freeriders):
        return compute_metrics(detection_results, freeriders)


def compute_update(global_model, weights_client, name_layer_grads=None):
    global_params = get_parameters(global_model)
    model_grads = []
    for i, (layer, param) in enumerate(global_model.named_parameters()):
        if param.requires_grad:
            para_diff = weights_client[i] - global_params[i]
            model_grads.append(torch.tensor(para_diff).flatten())

    model_grads = torch.cat(model_grads, -1)

    return model_grads


def zscore_detect(samples, lower_threshold=1, higher_threshold=1):
    zscores = (samples - np.mean(samples)) / np.std(samples)
    return (
        [
            True if z > higher_threshold or -z > lower_threshold else False
            for z in zscores
        ],
        zscores,
    )


def compute_update(global_model, weights_client, name_layer_grads=None):
    global_params = get_parameters(global_model)
    model_grads = []
    for i, (layer, param) in enumerate(global_model.named_parameters()):
        if param.requires_grad:
            para_diff = weights_client[i] - global_params[i]
            model_grads.append(torch.tensor(para_diff).flatten())

    model_grads = torch.cat(model_grads, -1)

    return model_grads


class DAGMMAttack:
    def __init__(
        self, n_encoder_layers, n_gmm_layers, device, batch_size=16, epochs=100
    ):
        self.n_encoder_layers = n_encoder_layers
        self.n_gmm_layers = n_gmm_layers
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size

    def do_attack(self, prev_params, weight_results):
        dagmm = DAGMM(self.n_encoder_layers, self.n_gmm_layers).to(self.device)
        grads_clients = [
            [
                w[1],
                np.concatenate(
                    [(new - old).flatten() for old, new in zip(prev_params, w[0])]
                ),
            ]
            for w in weight_results
        ]
        grads_clients.sort(key=lambda x: int(x[0]))
        grads = torch.tensor(np.array([g[1] for g in grads_clients]))
        dataset = TensorDataset(grads)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        train_dagmm(dagmm, data_loader, self.device, self.epochs)
        dagmm.eval()
        data_loader = DataLoader(dataset, batch_size=len(grads))

        for batch in data_loader:
            data = batch[0].to(self.device)
            membership_estimations, combined_features, _ = dagmm(data)
            mixture_probabilities, means, covariances = get_gmm_parameters(
                membership_estimations, combined_features
            )
            energies = [
                get_energy(f, mixture_probabilities, means, covariances)
                .detach()
                .cpu()
                .numpy()
                for f in combined_features
            ]
            detection_results, zscores = zscore_detect(energies, lower_threshold=np.inf)
        return detection_results, (energies, zscores)

    def get_metrics(self, detection_results, freeriders):
        return compute_metrics(detection_results, freeriders)


class STDDAGMMAttack:
    def __init__(
        self, n_encoder_layers, n_gmm_components, device, batch_size=16, epochs=100
    ):
        self.n_encoder_layers = n_encoder_layers
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_gmm_components = 3

    def do_attack(self, prev_params, weight_results, zscore_threshold):
        std_dagmm = STDDAGMM(
            self.n_encoder_layers, n_gmm_components=self.n_gmm_components
        ).to(self.device)

        grads_clients = [
            [
                w[1],
                np.concatenate(
                    [(new - old).flatten() for old, new in zip(prev_params, w[0])]
                ),
            ]
            for w in weight_results
        ]
        grads_clients.sort(key=lambda x: int(x[0]))
        grads = torch.tensor(
            np.array([g[1] for g in grads_clients]), dtype=torch.float32
        )

        dataset = TensorDataset(grads)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        train_std_dagmm(std_dagmm, data_loader, self.device, self.epochs)

        std_dagmm.eval()
        with torch.no_grad():
            data = grads.to(self.device)
            gamma, features, _ = std_dagmm(data)

            phi, mu, sigma = std_dagmm.criterion._get_gmm_parameters(gamma, features)
            energies = std_dagmm.criterion._compute_energy(features, phi, mu, sigma)
            energies = energies.detach().cpu().numpy()

        detection_results = self._adaptive_threshold_detection(energies)

        return detection_results, (energies, None)

    def _adaptive_threshold_detection(self, energies):
        mean = np.mean(energies)
        std = np.std(energies)

        median = np.median(energies)
        mad = np.median(np.abs(energies - median))

        z_scores = (energies - mean) / (std + 1e-8)
        mad_scores = (energies - median) / (mad + 1e-8)

        combined_scores = 0.7 * np.abs(z_scores) + 0.3 * np.abs(mad_scores)
        threshold = 1.0

        return [score > threshold for score in combined_scores]

    def get_metrics(self, detection_results, freeriders):
        return compute_metrics(detection_results, freeriders)


class CosineSimilarityAttack:
    def do_attack(self, global_model, weight_results, threshold=1.0, server_round=None):
        grads_clients = [
            [w[1], compute_update(global_model, w[0])] for w in weight_results
        ]
        grads_clients.sort(key=lambda x: int(x[0]))
        grads_clients = [g[1] for g in grads_clients]

        chosen_idx = random.randrange(len(grads_clients))
        chosen_grads = grads_clients.pop(chosen_idx)

        similarities = [spatial.distance.cosine(chosen_grads, g) for g in grads_clients]
        detection_results, zscores = zscore_detect(similarities, threshold, threshold)
        detection_results.insert(chosen_idx, False)

        return detection_results, (similarities, zscores)

    def get_metrics(self, detection_results, freeriders):
        return compute_metrics(detection_results, freeriders)


class L2NormAttack:
    def do_attack(self, global_model, weight_results, threshold=1.0, server_round=None):
        grads_clients = [
            [w[1], compute_update(global_model, w[0])] for w in weight_results
        ]
        grads_clients.sort(key=lambda x: int(x[0]))
        grads_clients = [g[1].detach().numpy() for g in grads_clients]
        l2norms = [np.linalg.norm(g, 2) for g in grads_clients]
        detection_results, zscores = zscore_detect(l2norms, threshold, threshold)

        return detection_results, (l2norms, zscores)

    def get_metrics(self, detection_results, freeriders):
        return compute_metrics(detection_results, freeriders)


class STDAttack:
    def do_attack(self, global_model, weight_results, threshold=1.0, server_round=None):
        grads_clients = [
            [w[1], compute_update(global_model, w[0])] for w in weight_results
        ]
        grads_clients.sort(key=lambda x: int(x[0]))
        grads_clients = [g[1].detach().numpy() for g in grads_clients]
        stds = [np.std(g) for g in grads_clients]
        detection_results, zscores = zscore_detect(stds, threshold, threshold)

        return detection_results, (stds, zscores)

    def get_metrics(self, detection_results, freeriders):
        return compute_metrics(detection_results, freeriders)


def pia(
    global_model, prev_params, weight_results, num_examples, cfg, attack_loaders, device
):
    return old_pia(
        global_model,
        prev_params,
        weight_results,
        num_examples,
        attack_loaders,
        cfg,
        device,
    )


class DistScoreAttack:
    def __init__(self, device, attack_loaders):
        self.device = device
        self.attack_loaders = attack_loaders
        self.global_labels_normalised = None

    def do_attack(
        self,
        global_model,
        prev_params,
        weight_results,
        num_examples,
        cfg,
        threshold=1.0,
    ):
        labels = pia(
            global_model,
            prev_params,
            weight_results,
            num_examples,
            cfg,
            self.attack_loaders,
            self.device,
        )
        detection_results, (errors, zscores) = self.detect(labels, threshold)
        return (
            detection_results,
            (errors, zscores),
            {
                cfg.pia_type: labels,
            },
        )

    def detect(self, labels, threshold=1, verbose=True):
        global_labels = np.average(labels, axis=0)
        labels_normalised = labels / np.expand_dims(np.sum(labels, axis=1), 1)
        if self.global_labels_normalised is None:
            self.global_labels_normalised = global_labels / np.sum(global_labels)
        if np.any(np.isnan(self.global_labels_normalised)):
            self.global_labels_normalised = np.zeros_like(self.global_labels_normalised)
        no_grads_labels_normalised = np.ones_like(global_labels) / np.size(
            global_labels, axis=0
        )
        basis = np.stack((self.global_labels_normalised, no_grads_labels_normalised))
        grads_component = []
        no_grads_components = []
        errors = []
        for l in labels_normalised:
            if np.any(np.isnan(l)):
                l = np.zeros_like(l)
            decomposition = decompose(basis, l)
            grads_component.append(decomposition[0])
            no_grads_components.append(decomposition[1])
            difference = decomposition @ basis - l
            errors.append(np.linalg.norm(difference, ord=1))
        errors = np.array(errors)
        errors = np.log(np.clip(errors, a_min=1e-300, a_max=np.inf))
        detection_results, zscores = zscore_detect(errors, threshold, threshold)

        self.global_labels_normalised = global_labels / np.sum(global_labels)
        if verbose:
            print(
                "detection results distance defense: {}, zscores: {}".format(
                    detection_results, zscores
                )
            )
        return detection_results, (errors, zscores)

    def get_metrics(self, detection_results, freeriders):
        return compute_metrics(detection_results, freeriders)


class InconsistencyAttack:
    def __init__(self, device, attack_loaders):
        self.device = device
        self.attack_loaders = attack_loaders
        self.labels_normalised = None

    def do_attack(
        self, global_model, prev_params, weight_results, num_examples, cfg, threshold
    ):
        labels = pia(
            global_model,
            prev_params,
            weight_results,
            num_examples,
            cfg,
            self.attack_loaders,
            self.device,
        )
        detection_results, (inconsistencies, zscores) = self.detect(labels, threshold)
        return (
            detection_results,
            (inconsistencies, zscores),
            {
                cfg.pia_type: labels,
            },
        )

    def detect(self, labels, threshold=1.0, verbose=True):
        labels_normalised = labels / np.expand_dims(np.sum(labels, axis=1), 1)
        if self.labels_normalised is None:
            self.labels_normalised = np.expand_dims(labels_normalised, 0)
            return [False] * np.size(labels, axis=0), (
                [0] * np.size(labels, axis=0),
                [0] * np.size(labels, axis=0),
            )
        else:
            self.labels_normalised = np.append(
                self.labels_normalised, np.expand_dims(labels_normalised, 0), axis=0
            )
        if np.size(self.labels_normalised, 0) > 2:
            np.delete(self.labels_normalised, 0, axis=0)

        inconsistencies = np.mean(
            np.linalg.norm(np.diff(self.labels_normalised, axis=0), ord=2, axis=2),
            axis=0,
        )
        detection_results, zscores = zscore_detect(
            inconsistencies, threshold, threshold
        )
        if verbose:
            print(
                "detection results inconsistency defense: {}, zscores: {}".format(
                    detection_results, zscores
                )
            )

        return detection_results, (inconsistencies, zscores)

    def get_metrics(self, detection_results, freeriders):
        return compute_metrics(detection_results, freeriders)


def decompose(basis, target):
    return np.linalg.pinv(basis @ np.transpose(basis)) @ basis @ target


def iqr_detect(
    samples,
    lower_precentile=25,
    higher_precentile=75,
    lower_threshold=1.5,
    higher_threshold=1.5,
):
    lower_sample = np.percentile(samples, lower_precentile)
    higher_sample = np.percentile(samples, higher_precentile)
    diff = higher_sample - lower_sample
    upper_bound = higher_sample + diff * higher_threshold
    lower_bound = lower_sample - diff * lower_threshold
    return [True if x < lower_bound or x > upper_bound else False for x in samples]


def cluster_detect(samples, threshold_ratio=1.5):
    ordered = np.sort(samples)
    min_idx = np.size(ordered) // 2
    max_idx = min_idx
    threshold = 0
    stop_size = np.size(ordered) / 2
    while max_idx - min_idx + 1 < stop_size:
        if min_idx > 0 and max_idx < np.size(ordered) - 1:
            if (
                ordered[min_idx] - ordered[min_idx - 1]
                < ordered[max_idx + 1] - ordered[max_idx]
            ):
                threshold = np.max((ordered[min_idx] - ordered[min_idx - 1], threshold))
                min_idx -= 1
            else:
                threshold = np.max((ordered[max_idx + 1] - ordered[max_idx], threshold))
                max_idx += 1
        elif min_idx > 0:
            threshold = np.max((ordered[min_idx] - ordered[min_idx - 1], threshold))
            min_idx -= 1
        else:
            threshold = np.max((ordered[max_idx + 1] - ordered[max_idx], threshold))
            max_idx += 1
    threshold = threshold * threshold_ratio
    while min_idx > 0 and ordered[min_idx] - ordered[min_idx - 1] <= threshold:
        threshold = np.max(
            (threshold, (ordered[min_idx] - ordered[min_idx - 1]) * threshold_ratio)
        )
        min_idx -= 1
    while (
        max_idx < np.size(ordered) - 1
        and ordered[max_idx + 1] - ordered[max_idx] <= threshold
    ):
        threshold = np.max(
            (threshold, (ordered[max_idx + 1] - ordered[max_idx]) * threshold_ratio)
        )
        max_idx += 1
    lower_bound = ordered[min_idx]
    upper_bound = ordered[max_idx]
    return [True if x < lower_bound or x > upper_bound else False for x in samples]


def cluster_zscore_detect(samples):
    ordered = np.sort(samples)
    min_idx = np.size(ordered) // 2
    max_idx = min_idx
    threshold = 0
    stop_size = np.size(ordered) / 2
    while max_idx - min_idx + 1 < stop_size:
        if min_idx > 0 and max_idx < np.size(ordered) - 1:
            if (
                ordered[min_idx] - ordered[min_idx - 1]
                < ordered[max_idx + 1] - ordered[max_idx]
            ):
                threshold = np.max((ordered[min_idx] - ordered[min_idx - 1], threshold))
                min_idx -= 1
            else:
                threshold = np.max((ordered[max_idx + 1] - ordered[max_idx], threshold))
                max_idx += 1
        elif min_idx > 0:
            threshold = np.max((ordered[min_idx] - ordered[min_idx - 1], threshold))
            min_idx -= 1
        else:
            threshold = np.max((ordered[max_idx + 1] - ordered[max_idx], threshold))
            max_idx += 1
    zscores = (
        2
        * (samples - np.mean(ordered[min_idx : max_idx + 1]))
        / (ordered[max_idx] - ordered[min_idx])
    )
    return ([True if z > 4 or -z > 4 else False for z in zscores], zscores)


def modified_zscore_detect(samples, lower_threshold=2, higher_threshold=2):
    denominator = np.median(np.abs(samples - np.median(samples)))
    if denominator == 0:
        zscores = np.zeros_like(samples)
    else:
        zscores = 0.6745 * (samples - np.median(samples)) / denominator
    return (
        [
            True if z > higher_threshold or -z > lower_threshold else False
            for z in zscores
        ],
        zscores,
    )
