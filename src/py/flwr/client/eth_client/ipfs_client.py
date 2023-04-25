import io

import ipfshttpclient
import torch
import os
import numpy as np
from collections import OrderedDict
import flwr.common.typing


class IPFSClient:

    # class attribute so that IPFSClient's on the same machine can all benefit
    _cached_models = {}

    _model_constructor = None

    def __init__(self, ipfs_api):
        self._ipfs_api = ipfs_api
        self.model = None

    def set_model(self,net):
        self.model = net

    def get_model(self, model_cid):
        if model_cid is None:
            return self.model
        if model_cid in self._cached_models:
            print("get cached model")
            # make a deep copy from cache
            self.model.load_state_dict(
                self._cached_models[model_cid])
        else:
            # download from IPFS
            print("download from IPFS")
            with ipfshttpclient.connect(self._ipfs_api) as ipfs:
                print("ipfs connection done")
                print("check cid",model_cid)
                model_bytes = ipfs.cat(model_cid,timeout=60)
                print("upload done")
            buffer = io.BytesIO(model_bytes)
            self.model.load_state_dict(torch.load(buffer))
            self._cached_models[model_cid] = self.model.state_dict()
        return self.model

    def add_model(self, model):
        # params = list(model.parameters())
        # param = params[0]
        buffer = io.BytesIO()
        if isinstance(model, torch.nn.Module):
            check = model.state_dict()
            torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        with ipfshttpclient.connect(self._ipfs_api) as ipfs:
            model_cid = ipfs.add_bytes(buffer.read())

        return model_cid

    def architecture_to_ipfs(self,model):
        state_dict = model.state_dict()
        architecture = {}
        for key in state_dict.keys():
            if key.endswith(".weight"):
                module_path = key[:-len(".weight")]
                module = model
                for part in module_path.split("."):
                    module = getattr(module, part)
                architecture[module_path] = module
            # Save architecture to file and upload to IPFS
        buffer = io.BytesIO()
        torch.save(architecture, buffer)
        buffer.seek(0)
        with ipfshttpclient.connect(self._ipfs_api) as ipfs:
            arch_cid = ipfs.add_bytes(buffer.read())
        return arch_cid

    def load_architecture(self,cid):
        # Load architecture from IPFS
        with ipfshttpclient.connect(self._ipfs_api) as ipfs:
            buffer = io.BytesIO(ipfs.cat(cid))
            architecture = torch.load(buffer)
        # Create model from architecture
        self.model = torch.nn.Module()
        for module_path, module in architecture.items():
            parent = self.model
            parts = module_path.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], module)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
