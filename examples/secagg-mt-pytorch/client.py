from typing import Dict
from logging import ERROR, WARNING, INFO, DEBUG

import flwr as fl
from flwr.client.message_handler.handler_registry import register_handler
from flwr.common import FitIns, FitRes, NDArrays, Scalar
from flwr.common.typing import Task, SecureAggregationMessage
from flwr.common.secure_aggregation.sec_agg_protocol import *
from flwr.common.logger import log

from task import (
    Net,
    DEVICE,
    load_data,
    get_parameters,
    set_parameters,
    train,
    test,
)


# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.Client):
    # def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
    #     return get_parameters(net)

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        results = train(net, trainloader, testloader, epochs=1, device=DEVICE)
        return get_parameters(net), len(trainloader.dataset), results

    # def evaluate(self, parameters, config):
    #     set_parameters(net, parameters)
    #     loss, accuracy = test(net, testloader)
    #     return loss, len(testloader.dataset), {"accuracy": accuracy}
    stages = ["setup", "share keys", "collect masked input", "unmasking"]
    current_stage = "unmasking"

    @register_handler("sec_agg")
    def handle_secagg(self, task: Task) -> Task:
        stage = task.secure_aggregation_message.named_scalars.pop("stage")
        if stage == "setup":
            if self.current_stage != "unmasking":
                log(WARNING, "restart from setup stage")
            self.current_stage = stage
            return setup(self, task)
        # if stage is not "setup", the new stage should be the next stage
        expected_new_stage = self.stages[self.stages.index(self.current_stage) + 1]
        if stage == expected_new_stage:
            self.current_stage = stage
        else:
            raise ValueError(
                "Abort secure aggregation: "
                f"expect {expected_new_stage} stage, but receive {stage} stage"
            )

        if stage == "share keys":
            return share_keys(self, task)
        if stage == "collect masked input":
            return collect_masked_input(self, task)
        if stage == "unmasking":
            return unmasking(self, task)
        raise ValueError(f"Unknown secagg stage: {stage}")

        # return Task(secure_aggregation_message=SecureAggregationMessage(
        #     named_scalars={'secagg_msg':
        #         f"Hello driver! I got a secagg instruction for SA round {task.secure_aggregation_message.named_scalars['secagg_round']}!"}))


def setup(self: FlowerClient, task: Task) -> Task:
    # Assigning parameter values to object fields
    sec_agg_param_dict = task.secure_aggregation_message.named_scalars
    self.sample_num = sec_agg_param_dict["share_num"]
    self.sid = sec_agg_param_dict["secure_id"]
    # self.sec_agg_id = sec_agg_param_dict["secure_id"]
    log(INFO, f"Client {self.sid}: starting stage 0...")

    self.share_num = sec_agg_param_dict["share_num"]
    self.threshold = sec_agg_param_dict["threshold"]
    self.drop_flag = sec_agg_param_dict["test_drop"]
    self.clipping_range = 3
    self.target_range = 1 << 16
    self.mod_range = 1 << 24

    # key is the secure id of another client (int)
    # value is the secret share we possess that contributes to the client's secret (bytes)
    self.b_share_dict = {}
    self.sk1_share_dict = {}
    self.shared_key_2_dict = {}
    # Create 2 sets private public key pairs
    # One for creating pairwise masks
    # One for encrypting message to distribute shares
    self.sk1, self.pk1 = generate_key_pairs()
    self.sk2, self.pk2 = generate_key_pairs()

    self.sk1, self.pk1 = private_key_to_bytes(self.sk1), public_key_to_bytes(self.pk1)
    self.sk2, self.pk2 = private_key_to_bytes(self.sk2), public_key_to_bytes(self.pk2)
    log(INFO, f"Client {self.sid}: stage 0 completes. uploading public keys...")
    return Task(
        secure_aggregation_message=SecureAggregationMessage(
            named_scalars={"pk1": self.pk1, "pk2": self.pk2}
        )
    )


def share_keys(self: FlowerClient, task: Task) -> Task:
    key_dict = task.secure_aggregation_message.named_scalars
    key_dict = {int(sid): (pk1, pk2) for sid, (pk1, pk2) in key_dict.items()}
    log(INFO, f"Client {self.sid}: starting stage 1...")
    # Distribute shares for private mask seed and first private key
    # share_keys_dict:
    self.public_keys_dict = key_dict
    # check size is larger than threshold
    if len(self.public_keys_dict) < self.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # check if all public keys received are unique
    pk_list: List[bytes] = []
    for i in self.public_keys_dict.values():
        pk_list.append(i[0])
        pk_list.append(i[1])
    if len(set(pk_list)) != len(pk_list):
        raise Exception("Some public keys are identical")

    # sanity check that own public keys are correct in dict
    if (
        self.public_keys_dict[self.sid][0] != self.pk1
        or self.public_keys_dict[self.sid][1] != self.pk2
    ):
        raise Exception(
            "Own public keys are displayed in dict incorrectly, should not happen!"
        )

    # Generate private mask seed
    self.b = rand_bytes(32)

    # Create shares
    b_shares = create_shares(self.b, self.threshold, self.share_num)
    sk1_shares = create_shares(self.sk1, self.threshold, self.share_num)

    srcs, dsts, ciphertexts = [], [], []

    for idx, p in enumerate(self.public_keys_dict.items()):
        client_sid, client_public_keys = p
        if client_sid == self.sid:
            self.b_share_dict[self.sid] = b_shares[idx]
            self.sk1_share_dict[self.sid] = sk1_shares[idx]
        else:
            shared_key = generate_shared_key(
                bytes_to_private_key(self.sk2),
                bytes_to_public_key(client_public_keys[1]),
            )
            self.shared_key_2_dict[client_sid] = shared_key
            plaintext = share_keys_plaintext_concat(
                self.sid, client_sid, b_shares[idx], sk1_shares[idx]
            )
            ciphertext = encrypt(shared_key, plaintext)
            srcs.append(self.sid)
            dsts.append(client_sid)
            ciphertexts.append(ciphertext)

    log(INFO, f"Client {self.sid}: stage 1 completes. uploading key shares...")
    return Task(
        secure_aggregation_message=SecureAggregationMessage(
            named_scalars={"dsts": dsts, "ciphertexts": ciphertexts}
        )
    )


def collect_masked_input(self: FlowerClient, task: Task) -> Task:
    log(INFO, f"Client {self.sid}: starting stage 2...")
    # Receive shares and fit model
    available_clients: List[int] = []
    msg = task.secure_aggregation_message
    ciphertexts = msg.named_scalars["ciphertexts"]
    srcs = msg.named_scalars["srcs"]
    assert isinstance(ciphertexts, list)
    if len(ciphertexts) + 1 < self.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # decode all packets and verify all packets are valid. Save shares received
    for src, ct in zip(srcs, ciphertexts):
        shared_key = self.shared_key_2_dict[src]
        plaintext = decrypt(shared_key, ct)
        _src, dst, b_share, sk1_share = share_keys_plaintext_separate(plaintext)
        available_clients.append(src)
        if src != _src:
            raise Exception(
                f"Client {self.sid}: received ciphertext from {_src} instead of {src}"
            )
        if dst != self.sid:
            ValueError(
                f"Client {self.sid}: received an encrypted message for Client {dst} from Client {src}"
            )
        self.b_share_dict[src] = b_share
        self.sk1_share_dict[src] = sk1_share

    # fit client
    # IMPORTANT ASSUMPTION: ASSUME ALL CLIENTS FIT SAME AMOUNT OF DATA
    if self.drop_flag:
        # log(ERROR, "Force dropout due to testing!!")
        raise Exception("Force dropout due to testing")

    weights, weights_factor, _ = self.fit(msg.named_arrays["parameters"], {})

    # weights = [np.zeros(10000)]
    # weights_factor = 1

    # Quantize weight update vector
    quantized_weights = quantize(weights, self.clipping_range, self.target_range)

    quantized_weights = weights_multiply(quantized_weights, weights_factor)
    quantized_weights = factor_weights_combine(weights_factor, quantized_weights)

    dimensions_list: List[Tuple] = [a.shape for a in quantized_weights]

    # add private mask
    private_mask = pseudo_rand_gen(self.b, self.mod_range, dimensions_list)
    quantized_weights = weights_addition(quantized_weights, private_mask)

    for client_id in available_clients:
        # add pairwise mask
        shared_key = generate_shared_key(
            bytes_to_private_key(self.sk1),
            bytes_to_public_key(self.public_keys_dict[client_id][0]),
        )
        # print('shared key length: %d' % len(shared_key))
        pairwise_mask = pseudo_rand_gen(shared_key, self.mod_range, dimensions_list)
        if self.sid > client_id:
            quantized_weights = weights_addition(quantized_weights, pairwise_mask)
        else:
            quantized_weights = weights_subtraction(quantized_weights, pairwise_mask)

    # Take mod of final weight update vector and return to server
    quantized_weights = weights_mod(quantized_weights, self.mod_range)
    # return ndarrays_to_parameters(quantized_weights)
    log(INFO, f"Client {self.sid}: stage 2 completes. uploading masked weights...")
    return Task(
        secure_aggregation_message=SecureAggregationMessage(
            named_arrays={"masked_weights": quantized_weights}
        )
    )


def unmasking(self: FlowerClient, task: Task) -> Task:
    msg = task.secure_aggregation_message
    active_sids, dead_sids = (
        msg.named_scalars["active_sids"],
        msg.named_scalars["dead_sids"],
    )
    # Send private mask seed share for every avaliable client (including itclient)
    # Send first private key share for building pairwise mask for every dropped client
    if len(active_sids) < self.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    sids, shares = [], []
    sids += active_sids
    shares += [self.b_share_dict[sid] for sid in active_sids]
    sids += dead_sids
    shares += [self.sk1_share_dict[sid] for sid in dead_sids]

    return Task(
        secure_aggregation_message=SecureAggregationMessage(
            named_scalars={"sids": sids, "shares": shares}
        )
    )


# Start Flower client
fl.client.start_client(
    server_address="0.0.0.0:9092",
    client=FlowerClient(),
    transport="grpc-rere",
)
