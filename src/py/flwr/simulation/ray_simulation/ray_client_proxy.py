# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ray-based Flower ClientProxy implementation."""

import traceback
from typing import Dict

import ray

from flwr import common
from flwr.client import Client
from flwr.server.client_proxy import ClientProxy


class RayClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(
        self, cid: str, client_type: Client, resources: Dict[str, int], fed_dir: str
    ):
        super().__init__(cid)
        self.client_type = client_type
        self.resources = resources
        self.fed_dir = fed_dir

    def get_parameters(self) -> common.ParametersRes:
        """Return the current local model parameters."""

        # spawn client and run get_parameters()
        params = launch_and_get_params.options(**self.resources).remote(
            self.client_type, self.cid, self.fed_dir
        )
        parameters = ray.get(params)
        parameters = common.weights_to_parameters(parameters)

        return common.ParametersRes(parameters=parameters)

    def fit(self, ins: common.FitIns) -> common.FitRes:
        """Refine the provided weights using the locally held dataset."""

        weights = common.parameters_to_weights(ins.parameters)

        # spawn client and run fit()
        try:
            remote_fit = launch_and_fit.options(**self.resources).remote(
                self.client_type, self.cid, self.fed_dir, weights, ins.config
            )
        except ray.exception.RayTaskError as ex:
            print("The following error occured:\n" + str(ex.cause()))
            raise ex
        parameters, num_examples, metrics = ray.get(remote_fit)

        parameters = common.weights_to_parameters(parameters)
        return common.FitRes(parameters, num_examples=num_examples, metrics=metrics)

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""

        weights = common.parameters_to_weights(ins.parameters)

        # spawn client and run evaluate()
        try:
            remote_eval = launch_and_eval.options(**self.resources).remote(
                self.client_type, self.cid, self.fed_dir, weights, ins.config
            )
        except Exception as ex:
            print("The following error occured:\n")
        #            raise ex
        loss, num_examples, metrics = ray.get(remote_eval)
        return common.EvaluateRes(loss, num_examples, metrics=metrics)

    def reconnect(self, reconnect: common.Reconnect) -> common.Disconnect:
        """Disconnect and (optionally) reconnect later."""

        # Nothing to do here.
        disconnect: common.Disconnect = None

        return disconnect


@ray.remote
def launch_and_get_params(client_type, cid, fed_dir):
    client = client_type(cid, fed_dir)
    return client.get_parameters()


@ray.remote
def launch_and_fit(client_type, cid, fed_dir, parameters, config):
    client = client_type(cid, fed_dir)
    try:
        return client.fit(parameters, config)
    except Exception as ex:
        print("An exception occured in fit_round:")
        traceback.print_exc()
        raise ex


@ray.remote
def launch_and_eval(client_type, cid, fed_dir, parameters, config):
    client = client_type(cid, fed_dir)
    try:
        return client.evaluate(parameters, config)
    except Exception as ex:
        print("An exception occured in evaluate_round:")
        traceback.print_exc()
        raise ex
