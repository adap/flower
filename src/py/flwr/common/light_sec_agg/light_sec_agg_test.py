import flwr as fl
import tensorflow as tf
import time
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from multiprocessing import Process
from flwr.server.strategy.light_sec_agg_fedavg import LightSecAggFedAvg
from flwr.client.sa_client_wrapper import SAClient
from flwr.common.typing import LightSecAggSetupConfigIns, LightSecAggSetupConfigRes, AskEncryptedEncodedMasksIns, \
    EncryptedEncodedMasksPacket, AskMaskedModelsIns, \
    AskAggregatedEncodedMasksIns, SAServerMessageCarrier, SAClientMessageCarrier
'''weights: Weights = [np.array([[-0.2, -0.5, 1.9], [0.0, 2.4, -1.9]]),
                    np.array([[0.2, 0.5, -1.9], [0.0, -2.4, 1.9]])]
quantized_weights = sec_agg_primitives.quantize(
    weights, 3, 10)
quantized_weights = sec_agg_primitives.weights_divide(quantized_weights, 4)
print(quantized_weights)'''


class MyLightSecAggWrapper(SAClient):
    """Wrapper which adds LightSecAgg methods."""

    def sa_respond(self, ins: SAServerMessageCarrier) -> SAClientMessageCarrier:
        if ins.identifier == '0':
            new_ins = LightSecAggSetupConfigIns(ins.str2scalar)
            res = self.setup_config(new_ins)
            return SAClientMessageCarrier(identifier='0', bytes_list=[res.pk])
        if ins.identifier == '1':
            public_keys_dict = dict([(int(k), LightSecAggSetupConfigRes(v)) for k, v in ins.str2scalar.items()])
            new_ins = AskEncryptedEncodedMasksIns(public_keys_dict)
            res = self.ask_encrypted_encoded_masks(new_ins)
            packet_dict = dict([(str(p.destination), p.ciphertext) for p in res.packet_list])
            return SAClientMessageCarrier(identifier='1', str2scalar=packet_dict)
        if ins.identifier == '2':
            packets = [EncryptedEncodedMasksPacket(int(k), self.id, v) for k, v in ins.str2scalar.items()]
            new_ins = AskMaskedModelsIns(packets, ins.fit_ins)
            res = self.ask_masked_models(new_ins)
            return SAClientMessageCarrier(identifier='2', parameters=res.parameters)
        if ins.identifier == '3':
            new_ins = AskAggregatedEncodedMasksIns(ins.numpy_ndarray_list[0].tolist())
            res = self.ask_aggregated_encoded_masks(new_ins)
            print("beep beep!!!")
            return SAClientMessageCarrier(identifier='3', parameters=res.aggregated_encoded_mask)

# Testing
# Define Flower client

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):  # type: ignore
        return model.get_weights()

    def fit(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


def test_start_server(sample_num=10, T=4, U=7, p=(1 << 31) - 1, vector_dimension=100000, dropout_value=0,
                      num_rounds=1):
    fl.server.start_server("localhost:8080", config={
        "num_rounds": num_rounds, "sec_agg": 2},
                           strategy=LightSecAggFedAvg(fraction_fit=1, min_fit_clients=sample_num,
                                                      min_available_clients=sample_num,
                                                      cfg_dict={"sample_num": sample_num,
                                                                "privacy_guarantee": T,
                                                                "min_clients": U,
                                                                "prime_number": p,
                                                                "clipping_range": 4,
                                                                "target_range": 1 << 16,
                                                                "test": 1,
                                                                "test_vector_dimension": vector_dimension,
                                                                "test_dropout_value": dropout_value}))


def test_start_client(server_address: str,
                      client,
                      grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH, ):
    fl.client.start_numpy_client(server_address, client, grpc_max_message_length, sa_wrapper=MyLightSecAggWrapper)


def test_start_simulation(sample_num=10, T=4, U=7, p=(1 << 31) - 1, vector_dimension=100000, dropout_value=0,
                          num_rounds=1):
    """Start a FL simulation."""
    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(
        target=test_start_server, args=(
            sample_num, T, U, p, vector_dimension, dropout_value, num_rounds)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)

    # Start all the clients
    for i in range(sample_num):
        client_process = Process(target=test_start_client,
                                 args=("localhost:8080", CifarClient()))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()

