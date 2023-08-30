from logging import INFO
import flwr as fl
import threading
from flwr.common.logger import log
from typing import Dict, List, Optional
import random

from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy

from utils import Model_rate_manager

class client_manager_HeteroFL(fl.server.ClientManager):
    """Provides a pool of available clients."""

    def __init__(self, model_rate_manager : Model_rate_manager = None, clients_to_model_rate_mapping = None , client_label_split = None , seed = 42) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        
        self.is_simulation = False
        if model_rate_manager != None:
            self.is_simulation = True
        
        self.model_rate_manager = model_rate_manager  
        
        # have a common array in simulation to access in the client_fn and server side
        if clients_to_model_rate_mapping != None:
            self.clients_to_model_rate_mapping = clients_to_model_rate_mapping
            ans = self.model_rate_manager.create_model_rate_mapping(len(clients_to_model_rate_mapping))
            # copy self.clients_to_model_rate_mapping , ans
            for i in range(len(ans)):
                self.clients_to_model_rate_mapping[i] = ans[i]
        
        # shall handle in case of not_simulation...
        self.client_label_split = client_label_split

        # control the randomness of sampling the clients
        random.seed(seed)
             
        self._cv = threading.Condition()

    def __len__(self) -> int:
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available.

        Blocks until the requested number of clients is available or until a
        timeout is reached. Current timeout default: 1 day.

        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for, defaults to 86400 (24h).

        Returns
        -------
        success : bool
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        
        # shall look into this, as this might not work as intended
        if(self.is_simulation == False):
            prop = client.get_properties(None, timeout= 86400)
            self.clients_to_model_rate_mapping[int(client.cid)] = prop['model_rate']
            self.client_label_split[int(client.cid)] = prop['label_split']
    
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.clients:
            del self.clients[client.cid]
            del self.clients_to_model_rate_mapping[int(client.cid)]
            del self.client_label_split[int(client.cid)]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients
    
    def get_client_to_model_mapping(self , cid) -> float:
        """Return all available clients to model rate mapping."""
        return self.clients_to_model_rate_mapping[int(cid)]
    
    def get_all_clients_to_model_mapping(self):
        """Return all available clients to model rate mapping."""
        return self.clients_to_model_rate_mapping.copy()
    
    def update(self , server_round):
        if self.is_simulation == True:
            if(server_round == 1 and self. model_rate_manager.model_mode == 'fix') or (self.model_rate_manager.model_mode == 'dynamic'):
                ans = self.model_rate_manager.create_model_rate_mapping(self.num_available)
                # copy self.clients_to_model_rate_mapping , ans
                for i in range(len(ans)):
                    self.clients_to_model_rate_mapping[i] = ans[i]
            return
        

        # to be handled in case of not simulation, i.e. to get the properties again from the clients as they can change the model_rate
        # for i in range(self.num_available):
        #     # need to test this , accumilates the changing model rate of the client
        #     self.clients_to_model_rate_mapping[i] = self.clients[str(i)].get_properties['model_rate']
        # return

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]
