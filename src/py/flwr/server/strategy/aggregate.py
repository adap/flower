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
"""Aggregation functions for strategy implementations."""


from functools import reduce
from math import isnan
from typing import List, Optional, Tuple, Dict

import numpy as np

from flwr.server.client_proxy import ClientProxy
from flwr.common import Weights, Parameters

from logging import INFO, DEBUG
from flwr.common.logger import log


def aggregate(results: List[Tuple[Weights, int]]) -> Weights:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    # model.coef_ = results[0][0][0]
    # model.intercept_ = results[0][0][1]        

    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: Weights = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

def aggregate_byzantine(results: List[Tuple[Weights, ClientProxy]], server_weights: Weights, byzantine_config: Dict) -> Weights:
    """Compute Byzantine average."""

    # Maintain client data in dict to avoid recalcuating it
    client_data = {}
    client_trust = byzantine_config["client_trust"]

    # Flatten list of server weight matrices
    server_flat = np.concatenate([weight_matrix.flatten() for weight_matrix in server_weights])
    length = np.shape(server_flat)[0]
    log(INFO, f"Weight vector has {length} entries and norm={round(np.linalg.norm(server_flat), 6)}")

    # median trust is the trust assigned to people who do not yet have trust
    median_trust = 1 if len(client_trust.values()) == 0 else np.median(list(client_trust.values()))

    # Determine the trust-weighted update direction
    unit_base = np.zeros(length)
    for client_weights, client in results:
        client_flat = np.concatenate([weight_matrix.flatten() for weight_matrix in client_weights])
        client_delta = client_flat - server_flat

        trust_factor = median_trust if client.cid not in client_trust.keys() else client_trust[client.cid]
        if np.linalg.norm(client_delta):
            unit_base += trust_factor * client_delta / np.linalg.norm(client_delta)

        client_data[client.cid] = {
            "trust_factor": trust_factor,
            "client_flat": client_flat,
            "client_delta": client_delta
        }

    # Convert to unit norm
    if np.linalg.norm(unit_base) == 0:
        unit_base[0] = 1
    else:
        unit_base = unit_base / np.linalg.norm(unit_base)

    # Collect trust, magnitudes, and angles for each client
    trusts = []
    magnitudes = []
    angles = []
    server_delta = np.zeros(length)
    for client_weights, client in results:        
        # Calculate client update
        client_flat, client_delta, trust_factor = client_data[client.cid]["client_flat"], client_data[client.cid]["client_delta"], client_data[client.cid]["trust_factor"]        

        # Caclualte client update magnitude
        mag = np.linalg.norm(client_delta)
        magnitudes.append(mag)
        client_data[client.cid]["magnitude"] = mag
        
        # calcualte client update angle // relative to average
        ang = 0
        if mag != 0:
            unit_delta = client_delta / mag
            dot_product = min(np.dot(unit_base, unit_delta), 1)
            ang = np.arccos(dot_product) * 180 / np.pi
            angles.append(ang)
            server_delta += trust_factor * unit_delta  # update server direction
        angles.append(ang)
        client_data[client.cid]["angle"] = ang 

        # add client contribution to new server weight
        trusts.append(trust_factor)

        log(INFO, f"cid={client.cid} mag={round(mag, 6)} ang={round(ang, 6)}")

    # Calculate server update
    if np.linalg.norm(server_delta) != 0:
        server_delta /= np.linalg.norm(server_delta)
    f = byzantine_config["max_byzantine_clients"] if byzantine_config["max_byzantine_clients"] < len(results) else 0
    threshold = sorted(magnitudes)[-f] if f else np.inf
    agg_mag = agg_trust = 0
    for mag, t in zip(magnitudes, trusts):
        if mag >= threshold:
            continue
        agg_mag += mag * t
        agg_trust += t
    server_update = agg_mag / agg_trust * server_delta
    
    # Calcualte updated server weights
    ang = 0
    if np.linalg.norm(server_delta):
        dot_product = min(np.dot(unit_base, server_delta) / np.linalg.norm(server_delta), 1)
        ang = np.arccos(dot_product) * 180 / np.pi
    new_server_flat = server_flat + server_update
    log(DEBUG, f"update_step  ang={round(ang, 6)} mag={round(np.linalg.norm(server_update), 6)} new_norm={round(np.linalg.norm(new_server_flat), 6)}")

    # Log metrics about average magnitude and angle
    median_mag, stdev_mag = np.median(magnitudes), np.std(magnitudes)
    mean_ang, stdev_ang = np.mean(angles), np.std(angles)
    log(DEBUG, f"mag={round(median_mag, 6)} ({round(stdev_mag, 6)})   ang={round(mean_ang, 6)} ({round(stdev_ang, 6)})")

    # Update Trusts
    total_trust = 0
    for _, data in client_data.items():
        new_trust_factor = data["trust_factor"]
        magnitude_delta = (data["magnitude"] - median_mag) / stdev_mag if stdev_mag else 0
        new_trust_factor *= byzantine_config["magnitude_trust_fn"](magnitude_delta)
        angle_delta = (data["angle"] - mean_ang) / stdev_ang if stdev_ang else 0
        new_trust_factor *= byzantine_config["angle_trust_fn"](angle_delta)
        total_trust += new_trust_factor

        # Update client data
        data["trust_factor"] = new_trust_factor
        data["magnitude_delta"] = magnitude_delta
        data["angle_delta"] = angle_delta

    # Calculate excess trust
    excess_trust = 0
    ctfs = []
    for cid, data in client_data.items():
        data["trust_factor"] /= total_trust
        if data["trust_factor"] > byzantine_config["max_trust"]:
            client_trust[cid] = byzantine_config["max_trust"]
            excess_trust += data["trust_factor"] - byzantine_config["max_trust"]
        elif data["trust_factor"] < byzantine_config["min_trust"]:
            client_trust[cid] = byzantine_config["min_trust"]
            excess_trust += data["trust_factor"] - byzantine_config["min_trust"]
        else:
            client_trust[cid] = data["trust_factor"]
            ctfs.append((cid, data["trust_factor"]))
    if excess_trust > 0:
        ctfs = sorted(ctfs, key=lambda x: x[1], reverse=True)
        for cid, tf in ctfs:
            if excess_trust < byzantine_config["max_trust"] - tf:
                client_trust[cid] += excess_trust
                excess_trust = 0
                break
            else:
                client_trust[cid] = byzantine_config["max_trust"]
                excess_trust -= byzantine_config["max_trust"] - tf

    for cid, data in client_data.items():
        log(INFO, f"cid={cid} trust={round(client_trust[cid], 6)} ang={round(data['angle_delta'], 6)} mag={round(data['magnitude_delta'], 6)}")
        print(f"round={byzantine_config['round']} cid={cid} trust={client_trust[cid]} ang={data['angle_delta']} mag={data['magnitude_delta']}")

    # reshape trust weights into Weights object
    new_server_weights: Weights = []
    index = 0
    for weight_matrix in server_weights:
        shape = np.shape(weight_matrix)
        num_elements = reduce(lambda x, y: x * y, shape)
        new_server_weights.append(np.reshape(new_server_flat[index : index + num_elements], shape))
        index += num_elements

    return new_server_weights

def weighted_loss_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _, _ in results]
    )
    weighted_losses = [num_examples * loss for num_examples, loss, _ in results]
    return sum(weighted_losses) / num_total_evaluation_examples

def aggregate_qffl(
    weights: Weights, deltas: List[Weights], hs_fll: List[Weights]
) -> Weights:
    """Compute weighted average based on  Q-FFL paper."""
    demominator = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_weights = [(u - v) * 1.0 for u, v in zip(weights, updates)]
    return new_weights
