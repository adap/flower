import flwr as fl
from .train import train, test
from .model import get_parameters, set_parameters
import torch
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler, Dataset
import numpy as np
from copy import deepcopy
import json


def set_parameters(net, parameters):
    params = zip(net.parameters(), parameters)
    for p, new_p in params:
        p.data = torch.tensor(new_p, dtype=p.dtype, device=p.device)


def get_parameters(net):
    return [p.cpu().detach().numpy() for p in net.parameters()]


def clip_gradients(net, norm_threshold):
        total_norm = 0.0
        for param in net.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = max(1.0, total_norm / norm_threshold)
        for param in net.parameters():
            if param.grad is not None:
                param.grad.data.mul_(1/clip_coef)


def add_gaussian_noise(net, cfg, device, sensitivity=1.0):
    for param in net.parameters():
        std = cfg.sigma * cfg.clip_threshold**2
        noise = torch.normal(mean=0.0, std=std, size=param.size(), device=device)
        param.data += noise
        param.data /= cfg.batch_size


class FreeRider(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader_len, freerider_type, device, cfg):
        self.freerider_type = freerider_type
        self.net = net.to(device)
        self.cid = cid
        self.device = device
        self.cfg = cfg
        self.trainloader_len = trainloader_len
    

    def fit(self, parameters, config):
        if self.freerider_type == "advanced_disguised":
            set_parameters(self.net, parameters)
            self.linear_noising(config["server_round"], config["std_noise"])
            new_parameters = get_parameters(self.net)

        elif self.freerider_type == "gradient_noiser_old":
            if config["prev_params"] == None:
                prenoise_parameters = parameters
                set_parameters(self.net, prenoise_parameters)
                self.linear_noising(config["server_round"], config["std_noise"])

            else:
                theta_t = [torch.tensor(param, device=self.device) for param in parameters]
                theta_t_1 = [torch.tensor(param, device=self.device) for param in config["prev_params"]]
                prenoise_parameters = [2 * new - old for new, old in zip(parameters, config["prev_params"])]
                set_parameters(self.net, prenoise_parameters)
                self.linear_noising(config["server_round"], config["std_noise"])

            new_parameters = get_parameters(self.net)

        elif self.freerider_type == "gradient_noiser":
            if config["prev_params"] == None or config["prev_prev_params"] == None:
                prenoise_parameters = parameters
                set_parameters(self.net, prenoise_parameters)
                self.linear_noising(config["server_round"], config["std_noise"])

            else:
                theta_t = [torch.tensor(param, device=self.device) for param in parameters]
                theta_t_1 = [torch.tensor(param, device=self.device) for param in config["prev_params"]]
                theta_t_2 = [torch.tensor(param, device=self.device) for param in config["prev_prev_params"]]

                theta_t_flat = torch.cat([torch.flatten(param) for param in theta_t])
                theta_t_1_flat = torch.cat([torch.flatten(param) for param in theta_t_1])
                theta_t_2_flat = torch.cat([torch.flatten(param) for param in theta_t_2])
                l_t = torch.norm(theta_t_flat - theta_t_1_flat, p=2)
                l_t_2 = torch.norm(theta_t_1_flat - theta_t_2_flat, p=2)
                scale = l_t / l_t_2

                prenoise_parameters = [(new + (new - old)*scale.item()) for new, old in zip(parameters, config["prev_params"])]
                set_parameters(self.net, prenoise_parameters)
                iteration = config["server_round"] 
                prob = 1

                for param, t, t1, t2 in zip(self.net.parameters(), theta_t, theta_t_1, theta_t_2):
                    variance = torch.var(t-t1) 
                    if torch.rand(1).item() < prob:
                        std = variance.sqrt().item()
                        noise = torch.normal(mean=0.0, std=std, size=param.size(), device=self.device)
                        param.data += noise

            new_parameters = get_parameters(self.net)

        elif self.freerider_type == "advanced":
            if config["prev_params"] is None or config["prev_prev_params"] is None:
                new_parameters = parameters
                set_parameters(self.net, new_parameters)
                new_parameters = get_parameters(self.net)
            else:
                new_parameters = self.advanced_attack(parameters, config)
            set_parameters(self.net, new_parameters)
            new_parameters = get_parameters(self.net)

        return new_parameters, self.trainloader_len, {"cid": self.cid}

    def advanced_attack(self, parameters, config):
        theta0 = [torch.tensor(param, device=self.device) for param in config["params0"]]
        theta1 = [torch.tensor(param, device=self.device) for param in config["params1"]]
        theta_t = [torch.tensor(param, device=self.device) for param in parameters]
        theta_t_1 = [torch.tensor(param, device=self.device) for param in config["prev_params"]]
        theta_t_2 = [torch.tensor(param, device=self.device) for param in config["prev_prev_params"]]

        theta_0_flat = torch.cat([torch.flatten(param) for param in theta0])
        theta_1_flat = torch.cat([torch.flatten(param) for param in theta1])
        theta_t_flat = torch.cat([torch.flatten(param) for param in theta_t])
        theta_t_1_flat = torch.cat([torch.flatten(param) for param in theta_t_1])
        theta_t_2_flat = torch.cat([torch.flatten(param) for param in theta_t_2])

        l_tnorm2= torch.norm(theta_t_flat - theta_t_1_flat, p=2)
        l_t_2norm2 = torch.norm(theta_t_1_flat - theta_t_2_flat, p=2)
        scale = l_tnorm2 / l_t_2norm2

        l_t = torch.norm(theta_t_flat - theta_t_1_flat, p=1)
        l_1 = torch.norm(theta_1_flat - theta_0_flat, p=1)

        d = 0.6

        theta_update = [curr+(curr - prev)*scale.item() for curr, prev in zip(theta_t, theta_t_1)]
        theta_update_flat = torch.cat([torch.flatten(param) for param in theta_update])
        C = (torch.norm(theta_0_flat-theta_t_flat, p=2) / torch.norm(theta_update_flat-theta_t_flat, p=2)).item()

        lambda_t = np.log((l_t.item() / l_1.item()) ** (1 / (config["server_round"] - 1)))
        expected_cos_beta = (C ** 2) / (C ** 2 + np.exp(2*lambda_t*config["server_round"]).item())
        U_f = [(curr - prev)*scale.item() for curr, prev in zip(theta_t, theta_t_1)]

        n = self.cfg.num_clients
        numerator = ((n ** 2) / (n + (n**2 - n) * expected_cos_beta)) - 1
        phi_t = torch.sqrt(torch.tensor(numerator, device=self.device)) * torch.norm(theta_t_flat - theta_t_1_flat, p=1)

        l = theta_t_flat.shape[0]*d
        noisy_update = []
        for u_f in U_f:
            mask = (torch.rand(u_f.shape, device=self.device) < d).float()
            noise = torch.normal(0, phi_t.item() / l, size=u_f.shape, device=self.device)
            noisy_u_f = u_f + mask * noise
            noisy_update.append(noisy_u_f)

        noisy_update = [curr + noise_update for curr, noise_update in zip(theta_t, noisy_update)]

        return noisy_update

    def get_n_params(self):
        return sum(np.prod(param.size()) for param in self.net.parameters())

    def linear_noising(self, iteration, std_noise):
        for param in self.net.parameters():
            std = self.cfg.multiplicator * std_noise * iteration ** (-self.cfg.power)
            noise = torch.normal(mean=0.0, std=std, size=param.size(), device=self.device)
            param.data += noise

    def linear_noising_random(self, iteration, std_noise, prob):
        for param in self.net.parameters():
            if torch.rand(1).item() < prob:
                std = self.cfg.multiplicator * std_noise * iteration ** (-self.cfg.power)
                noise = torch.normal(mean=0.0, std=std, size=param.size(), device=self.device)
                param.data += noise

    def linear_noising_parameter(self, iteration, std_noise_vector):
        for param, std_noise in zip(self.net.parameters(), std_noise_vector):
            std = self.cfg.multiplicator * std_noise * iteration ** (-self.cfg.power)
            noise = torch.normal(mean=0.0, std=std, size=param.size(), device=self.device)
            param.data += noise

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device, self.cfg)
        return float(loss), len(self.valloader.sampler), {"accuracy": float(accuracy)}

    def advanced_attack(self, parameters, config):
        theta0 = [torch.tensor(param, device=self.device) for param in config["params0"]]
        theta1 = [torch.tensor(param, device=self.device) for param in config["params1"]]
        theta_t = [torch.tensor(param, device=self.device) for param in parameters]
        theta_t_1 = [torch.tensor(param, device=self.device) for param in config["prev_params"]]
        theta_t_2 = [torch.tensor(param, device=self.device) for param in config["prev_prev_params"]]

        theta_0_flat = torch.cat([torch.flatten(param) for param in theta0])
        theta_1_flat = torch.cat([torch.flatten(param) for param in theta1])
        theta_t_flat = torch.cat([torch.flatten(param) for param in theta_t])
        theta_t_1_flat = torch.cat([torch.flatten(param) for param in theta_t_1])
        theta_t_2_flat = torch.cat([torch.flatten(param) for param in theta_t_2])

        l_tnorm2= torch.norm(theta_t_flat - theta_t_1_flat, p=2)
        l_t_2norm2 = torch.norm(theta_t_1_flat - theta_t_2_flat, p=2)
        scale = l_tnorm2 / l_t_2norm2

        l_t = torch.norm(theta_t_flat - theta_t_1_flat, p=1)
        l_1 = torch.norm(theta_1_flat - theta_0_flat, p=1)

        d = 0.6

        theta_update = [curr+(curr - prev)*scale.item() for curr, prev in zip(theta_t, theta_t_1)]
        theta_update_flat = torch.cat([torch.flatten(param) for param in theta_update])
        C = (torch.norm(theta_0_flat-theta_t_flat, p=2) / torch.norm(theta_update_flat-theta_t_flat, p=2)).item()

        lambda_t = np.log((l_t.item() / l_1.item()) ** (1 / (config["server_round"] - 1)))
        expected_cos_beta = (C ** 2) / (C ** 2 + np.exp(2*lambda_t*config["server_round"]).item())
        U_f = [(curr - prev)*scale.item() for curr, prev in zip(theta_t, theta_t_1)]

        n = self.cfg.num_clients
        numerator = ((n ** 2) / (n + (n**2 - n) * expected_cos_beta)) - 1
        phi_t = torch.sqrt(torch.tensor(numerator, device=self.device)) * torch.norm(theta_t_flat - theta_t_1_flat, p=1)

        l = theta_t_flat.shape[0]*d
        noisy_update = []
        for u_f in U_f:
            mask = (torch.rand(u_f.shape, device=self.device) < d).float()
            noise = torch.normal(0, phi_t.item() / l, size=u_f.shape, device=self.device)
            noisy_u_f = u_f + mask * noise
            noisy_update.append(noisy_u_f)

        noisy_update = [curr + noise_update for curr, noise_update in zip(theta_t, noisy_update)]

        return noisy_update
        

class FreeRiderSubset(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader_len, freerider_type, subsetloaders, device, cfg):
        self.freerider_type = freerider_type
        self.net = net.to(device)
        self.cid = cid
        self.device = device
        self.cfg = cfg
        self.trainloader_len = trainloader_len
        self.subsetloaders = subsetloaders


    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        subset = self.subsetloaders[config["server_round"] - 1] 
        train(self.net, subset, self.device, self.cfg, epochs=self.cfg.canary_epochs)

        if self.freerider_type == "advanced_disguised":
            set_parameters(self.net, parameters)
            self.linear_noising(config["server_round"], config["std_noise"])
            new_parameters = get_parameters(self.net)

        elif self.freerider_type == "gradient_noiser":
            if config["prev_params"] == None:
                prenoise_parameters = parameters
            else:
                prenoise_parameters = [2 * new - old for new, old in zip(parameters, config["prev_params"])]
            set_parameters(self.net, prenoise_parameters)
            self.linear_noising(config["server_round"], config["std_noise"])
            new_parameters = get_parameters(self.net)

        elif self.freerider_type == "advanced":
            if config["prev_params"] is None or config["prev_prev_params"] is None:
                new_parameters = parameters
                set_parameters(self.net, new_parameters)
                new_parameters = get_parameters(self.net)
            else:
                new_parameters = self.advanced_attack(parameters, config)
            set_parameters(self.net, new_parameters)
            new_parameters = get_parameters(self.net)
        
        return new_parameters, self.trainloader_len + len(subset), {"cid": self.cid}

    def get_n_params(self):
        return sum(np.prod(param.size()) for param in self.net.parameters())

    def linear_noising(self, iteration, std_noise):
        for param in self.net.parameters():
            # std = std_noise
            std = self.cfg.multiplicator * std_noise * iteration ** (-self.cfg.power)
            noise = torch.normal(mean=0.0, std=std, size=param.size(), device=self.device)
            param.data += noise

    def linear_noising_parameter(self, iteration, std_noise_vector):
        for param, std_noise in zip(self.net.parameters(), std_noise_vector):
            std = self.cfg.multiplicator * std_noise * iteration ** (-self.cfg.power)
            noise = torch.normal(mean=0.0, std=std, size=param.size(), device=self.device)
            param.data += noise

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device, self.cfg)
        return float(loss), len(self.valloader.sampler), {"accuracy": float(accuracy)}


class Client(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, device, cfg):
        self.cid = cid
        self.net = net.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.cfg = cfg

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        
        proximal_mu = config.get("proximal_mu", 0.0)
        global_params = None
        if proximal_mu > 0:
            global_params = [p.clone().detach() for p in self.net.parameters()]
        
        train(self.net, self.trainloader, self.device, self.cfg, 
              epochs=self.cfg.local_epochs, proximal_mu=proximal_mu, 
              global_params=global_params)

        if self.cfg.dp:
            clip_gradients(self.net, self.cfg.clip_threshold)
            add_gaussian_noise(self.net, self.cfg, self.device)

        return get_parameters(self.net), len(self.trainloader.sampler), {"cid": self.cid}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device, self.cfg)
        return float(loss), len(self.valloader.sampler), {"accuracy": float(accuracy)}


class ClientSubset(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, subsetloaders, device, cfg):
        self.net = net.to(device)
        self.subsetloaders = subsetloaders
        self.cfg = cfg
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

    def fit(self, parameters, config):
        subset = self.subsetloaders[config["server_round"] - 1]
        set_parameters(self.net, parameters)

        proximal_mu = config.get("proximal_mu", 0.0)
        global_params = None
        if proximal_mu > 0.0:
            global_params = [p.clone().detach() for p in self.net.parameters()]
        
        train(self.net, self.trainloader, self.device, self.cfg, 
              epochs=self.cfg.local_epochs, proximal_mu=proximal_mu, 
              global_params=global_params)
              
        train(self.net, subset, self.device, self.cfg, 
              epochs=self.cfg.canary_epochs, proximal_mu=proximal_mu, 
              global_params=global_params)
              
        if self.cfg.dp:
            clip_gradients(self.net, self.cfg.clip_threshold)  # Clip gradients here
            add_gaussian_noise(self.net, self.cfg, self.device)

        return get_parameters(self.net), len(self.trainloader) + len(subset), {"cid": self.cid}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device, self.cfg)
        return float(loss), len(self.valloader.sampler), {"accuracy": float(accuracy)}


def generate_client_fn(net, trainloaders, valloaders, vector_freeriders, cfg, device):
    def client_fn(cid: str):
        cid_int = int(cid)
        if vector_freeriders[cid_int]:
            return FreeRider(cid, deepcopy(net), len(trainloaders[cid_int].sampler), cfg.freerider_type, device, cfg).to_client()
        else:
            return Client(cid, deepcopy(net), trainloaders[cid_int], valloaders[cid_int], device, cfg).to_client()

    return client_fn


def generate_client_fn_subset(net, trainloaders, valloaders, vector_freeriders, subsetloaders, cfg, device):
    def client_fn(cid: str):
        cid_int = int(cid)
        if vector_freeriders[cid_int]:
            if cfg.freerider_canary:
                return FreeRiderSubset(cid, deepcopy(net), len(trainloaders[cid_int].sampler), cfg.freerider_type, subsetloaders, device, cfg).to_client()
            else:
                return FreeRider(cid, deepcopy(net), len(trainloaders[cid_int].sampler), cfg.freerider_type, device, cfg).to_client()

        else:
            return ClientSubset(
                cid, deepcopy(net), trainloaders[cid_int], valloaders[cid_int], subsetloaders, device, cfg
            ).to_client()

    return client_fn
