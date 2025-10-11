import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager
from typing import List, Tuple

print("Starting a FAULT-TOLERANT Flower server...")
print("The server will wait for 5 clients and can tolerate 1 client failure per round.")
print("-----------------------------------------------------------------------------")


# Part 1: The Custom ClientManager (Now logs connections AND disconnections)
class LoggingClientManager(SimpleClientManager):
    """A custom client manager that logs connection and disconnection events."""

    def register(self, client: ClientProxy) -> bool:
        """Register a new client and log the event."""
        was_registered = super().register(client)
        if was_registered:
            total_clients = len(self.clients)
            print(f"🎉 Client {client.cid} connected! [{total_clients}/5 clients now online]")
        return was_registered

    # --- NEW METHOD TO LOG DISCONNECTIONS ---
    def unregister(self, client: ClientProxy) -> None:
        """Unregister a client and log the event."""
        super().unregister(client)
        total_clients = len(self.clients)
        print(f"💔 Client {client.cid} disconnected. [{total_clients}/5 clients remain online]")
    # --- END OF NEW METHOD ---


# Part 2: The Custom Strategy (No changes needed here)
class MyCustomStrategy(fl.server.strategy.FedAvg):
    """A custom strategy to log which clients are selected for a round."""
    def configure_fit(
        self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        results = super().configure_fit(server_round, parameters, client_manager)
        if results:
            print(f"\n--- Round {server_round}: Starting training ---")
            print(f"  - Selected {len(results)} clients for this round:")
            for client_proxy, _ in results:
                print(f"    - Client CID: {client_proxy.cid}")
            print("---------------------------------------")
        return results


# Part 3: The Main Execution Block (With updated resilience rules)
if __name__ == "__main__":
    
    client_manager = LoggingClientManager()

    # --- UPDATED STRATEGY FOR FAULT TOLERANCE ---
    strategy = MyCustomStrategy(
        min_available_clients=5,  # Still wait for 5 clients to start the whole process.
        min_fit_clients=4,        # CRUCIAL: Consider a round successful even if only 4 clients return results.
        fraction_fit=1.0,         # Ask 100% of available clients (i.e., all 5) to participate.
    )
    # --- END OF UPDATE ---

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_manager=client_manager,
    )

    print("\nFederated learning process finished.")