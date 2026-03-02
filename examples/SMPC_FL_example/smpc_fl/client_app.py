"""SMPC Federated Learning Client App with P2P gRPC."""

from typing import List, Dict, Tuple
import logging
import numpy as np
import threading
import time
from concurrent import futures
import grpc
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr_datasets import FederatedDataset
from smpc_fl.utils import load_model
from smpc_fl.smpc_client import SMPCProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRPC_AVAILABLE = False
smpc_pb2 = None
smpc_pb2_grpc = None


def _load_grpc_modules():
    """Lazy load gRPC modules."""
    global GRPC_AVAILABLE, smpc_pb2, smpc_pb2_grpc
    if GRPC_AVAILABLE:
        return True
    
    import sys
    import os
    
    # Add smpc_fl directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        import smpc_pb2 as pb2
        import smpc_pb2_grpc as pb2_grpc
        smpc_pb2 = pb2
        smpc_pb2_grpc = pb2_grpc
        GRPC_AVAILABLE = True
        logger.info("gRPC modules loaded successfully")
        return True
    except ImportError:
        try:
            import smpc_fl.smpc_pb2 as pb2
            import smpc_fl.smpc_pb2_grpc as pb2_grpc
            smpc_pb2 = pb2
            smpc_pb2_grpc = pb2_grpc
            GRPC_AVAILABLE = True
            logger.info("gRPC modules loaded (absolute)")
            return True
        except ImportError:
            try:
                from . import smpc_pb2 as pb2
                from . import smpc_pb2_grpc as pb2_grpc
                smpc_pb2 = pb2
                smpc_pb2_grpc = pb2_grpc
                GRPC_AVAILABLE = True
                logger.info("gRPC modules loaded (relative)")
                return True
            except ImportError as e:
                logger.warning(f"gRPC not available: {e}")
                return False


def _get_smpc_servicer_class():
    """Get SMPCServicer class if gRPC is available."""
    if not _load_grpc_modules():
        return None
    
    class SMPCServicer(smpc_pb2_grpc.SMPCServicer):
        def __init__(self, client):
            self.client = client

        def SendShares(self, request, context):
            client_id = request.client_id
            shares = []
            for share in request.shares:
                shape = tuple(share.shape)
                data = np.frombuffer(share.data, dtype=np.float32)
                if data.size != np.prod(shape):
                    return smpc_pb2.AckResponse(status="ERROR")
                shares.append(data.reshape(shape))
            self.client.receive_shares(client_id, shares)
            return smpc_pb2.AckResponse(status="ACK")
    
    return SMPCServicer


class SMPCFlowerClient(NumPyClient):
    """Flower client with SMPC protocol and P2P communication."""
    
    def __init__(self, model, x_train, y_train, x_test, y_test, client_id: int, num_clients: int, client_port: int):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.client_id = client_id
        self.num_clients = num_clients
        self.client_port = client_port
        self.smpc = SMPCProtocol(num_clients)
        
        self.received_shares = {}
        self.all_shares_received = threading.Event()
        self.peer_stubs = {}
        self.own_shares = None
        
        # Start gRPC server for receiving shares
        if _load_grpc_modules():
            SMPCServicer = _get_smpc_servicer_class()
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            smpc_pb2_grpc.add_SMPCServicer_to_server(SMPCServicer(self), self.server)
            self.server.add_insecure_port(f"[::]:{self.client_port}")
            self.server.start()
            logger.info(f"Client {client_id} gRPC server started on port {client_port}")
        else:
            self.server = None
            logger.warning(f"Client {client_id}: gRPC not available, P2P SMPC disabled")

    def connect_to_peers(self, peer_addresses: List[str]):
        """Connect to peer clients via gRPC."""
        if not _load_grpc_modules():
            return
        for i, peer_address in enumerate(peer_addresses):
            if i != self.client_id:
                try:
                    channel = grpc.insecure_channel(peer_address)
                    self.peer_stubs[i] = smpc_pb2_grpc.SMPCStub(channel)
                    logger.info(f"Client {self.client_id} connected to peer {i} at {peer_address}")
                except Exception as e:
                    logger.error(f"Failed to connect to peer {i}: {e}")

    def send_shares_to_peers(self, secret_shares: Dict[int, List[np.ndarray]]):
        """Send secret shares to peer clients."""
        for peer_id, stub in self.peer_stubs.items():
            shares = secret_shares[peer_id]
            share_protos = []
            for matrix in shares:
                flattened_data = matrix.flatten()
                share_protos.append(smpc_pb2.Share(data=flattened_data.astype(np.float32).tobytes(), shape=list(matrix.shape)))
            request = smpc_pb2.SharesRequest(client_id=self.client_id, shares=share_protos)
            try:
                response = stub.SendShares(request)
                if response.status == "ACK":
                    logger.info(f"Client {self.client_id} sent shares to peer {peer_id}")
            except grpc.RpcError as e:
                logger.error(f"Failed to send shares to peer {peer_id}: {e}")

    def receive_shares(self, client_id: int, shares: List[np.ndarray]):
        """Receive secret shares from a peer client."""
        logger.info(f"Client {self.client_id} received shares from client {client_id}")
        self.received_shares[client_id] = shares
        expected_shares = self.num_clients - 1
        if len(self.received_shares) >= expected_shares:
            logger.info(f"Client {self.client_id} received all shares")
            self.all_shares_received.set()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=16, verbose=0)
        
        # Get peer addresses from config (comma-separated string)
        peer_addresses_str = config.get("peer_addresses", "")
        peer_addresses = peer_addresses_str.split(",") if peer_addresses_str else []
        
        if not _load_grpc_modules() or not peer_addresses or len(peer_addresses) < 2:
            logger.warning(f"Client {self.client_id}: P2P not available, using standard FedAvg")
            return self.model.get_weights(), len(self.x_train), {}
        
        # Connect to peers if not already connected
        if not self.peer_stubs:
            self.connect_to_peers(peer_addresses)
            time.sleep(1)
        
        # SMPC: split weights into shares
        weights = self.model.get_weights()
        shares = self.smpc.split_weights_to_shares(weights, self.num_clients)
        self.own_shares = shares[self.client_id]
        
        # Send shares to peers
        other_shares = {k: v for k, v in shares.items() if k != self.client_id}
        self.send_shares_to_peers(other_shares)
        
        # Wait for shares from other clients
        if self.all_shares_received.wait(timeout=30):
            all_shares = [self.own_shares] + list(self.received_shares.values())
            aggregated_shares = self.smpc.reconstruct_weights(all_shares)
            self.all_shares_received.clear()
            self.received_shares.clear()
            return aggregated_shares, len(self.x_train), {}
        else:
            logger.error(f"Client {self.client_id}: Timeout waiting for shares")
            return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict[str, float]]:
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def load_data_simulation(partition_id: int, num_partitions: int):
    """Load data for simulation mode using flwr_datasets."""
    fds = FederatedDataset(dataset="mnist", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42 + partition_id)
    
    x_train = np.array(partition_train_test["train"]["image"]).astype("float32") / 255.0
    y_train = np.array(partition_train_test["train"]["label"])
    x_test = np.array(partition_train_test["test"]["image"]).astype("float32") / 255.0
    y_test = np.array(partition_train_test["test"]["label"])
    
    return x_train, y_train, x_test, y_test


def load_data_deployment(data_path: str):
    """Load data for deployment mode from local path."""
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


def client_fn(context: Context):
    """Construct a Client for both simulation and deployment."""
    
    # Check if running in simulation or deployment mode
    if "partition-id" in context.node_config and "num-partitions" in context.node_config:
        # Simulation mode: use flwr_datasets
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        client_id = partition_id
        num_clients = num_partitions
        
        x_train, y_train, x_test, y_test = load_data_simulation(partition_id, num_partitions)
        logger.info(f"Simulation mode - Client {client_id}: train={len(x_train)}, test={len(x_test)}")
    else:
        # Deployment mode: load from local data path
        data_path = context.node_config.get("data-path", ".")
        client_id = context.node_config.get("client-id", 0)
        num_clients = context.node_config.get("num-clients", 1)
        
        x_train, y_train, x_test, y_test = load_data_deployment(data_path)
        logger.info(f"Deployment mode - Client {client_id}: train={len(x_train)}, test={len(x_test)}")
    
    # Get client port from config
    base_port = context.run_config.get("base-port", 50051)
    client_port = base_port + client_id
    
    # Load model
    model = load_model()
    
    return SMPCFlowerClient(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        client_id=client_id,
        num_clients=num_clients,
        client_port=client_port
    ).to_client()


app = ClientApp(client_fn=client_fn)
