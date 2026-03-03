"""fedhomo: Homomorphic FedAvg strategy using CKKS encryption."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tenseal as ts
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import flwr as fl

from fedhomo.crypto import (
    DecryptionError,
    EncryptedAggregator,
    EncryptionError,
    HomomorphicError,
)

log = logging.getLogger(__name__)

PUBLIC_CONTEXT_PATH = Path("keys") / "public_context.pkl"


class HomomorphicFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that aggregates model updates in the encrypted domain using CKKS."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregator = self._initialize_aggregator()
        log.info("Initialized HomomorphicFedAvg")

    def _initialize_aggregator(self) -> EncryptedAggregator:
        """Load the public CKKS context and initialize the aggregator."""
        try:
            with PUBLIC_CONTEXT_PATH.open("rb") as f:
                context = ts.context_from(f.read())
            if not context.is_public():
                raise EncryptionError("Loaded context is not public.")
            log.info("Loaded validated encryption context")
            return EncryptedAggregator(context)
        except FileNotFoundError:
            log.error("Missing public context file: %s", PUBLIC_CONTEXT_PATH)
            raise
        except Exception as e:
            log.error("Aggregator initialization failed: %s", e)
            raise

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate encrypted client updates in the CKKS domain.

        Args:
            server_round: Current federation round number.
            results: List of (client, FitRes) tuples from participating clients.
            failures: List of exceptions from failed clients.

        Returns:
            Tuple of (aggregated encrypted Parameters, metrics dict).
        """
        metrics: Dict[str, Any] = {
            "server_round": server_round,
            "clients_processed": len(results),
            "errors": 0,
        }

        if failures and not self.accept_failures:
            log.error("Aborting aggregation: %d critical failures", len(failures))
            return None, metrics

        if not results:
            log.warning("No results to aggregate")
            return None, metrics

        try:
            client_updates = self._process_client_updates(results, metrics)

            if not client_updates:
                log.warning("No valid client updates after processing")
                return None, metrics

            start = time.time()
            aggregated = self.aggregator.weighted_sum(client_updates)
            serialized = self.aggregator.serialize_vectors(aggregated)

            parameters = ndarrays_to_parameters(
                [np.frombuffer(v, dtype=np.uint8) for v in serialized]
            )

            elapsed = time.time() - start
            log.info("Round %d: aggregated %d clients in %.2fs", server_round, len(client_updates), elapsed)
            metrics["aggregation_time"] = elapsed

            return parameters, metrics

        except HomomorphicError as e:
            log.error("Security critical error during aggregation: %s", e)
            metrics["error"] = str(e)
            return None, metrics
        except Exception as e:
            log.error("Unexpected aggregation error: %s", e)
            metrics["error"] = str(e)
            return None, metrics

    def _process_client_updates(
    self,
    results: List[Tuple[ClientProxy, FitRes]],
    metrics: Dict[str, Any],
) -> List[Tuple[List[ts.CKKSVector], int]]:
        client_updates = []
        for client, res in results:
            try:
                encrypted_params = parameters_to_ndarrays(res.parameters)

                # Deserialize and verify all layers are valid CKKSVectors
                try:
                    vectors = self.aggregator.process_client_update(encrypted_params)
                    log.info(
                        "Client %s: all %d layers verified as CKKSVector ✓",
                        client.cid, len(vectors),
                    )
                except Exception:
                    log.error(
                        "Client %s: parameters are NOT valid CKKSVectors — plaintext leak!",
                        client.cid,
                    )
                    raise EncryptionError(
                        f"Client {client.cid} sent unencrypted parameters"
                    )

                client_updates.append((vectors, res.num_examples))

            except (EncryptionError, DecryptionError) as e:
                log.warning("Invalid update from client %s: %s", client.cid, e)
                metrics["errors"] += 1
                if not self.accept_failures:
                    raise
            except Exception as e:
                log.error("Unexpected error processing client %s: %s", client.cid, e)
                metrics["errors"] += 1
                if not self.accept_failures:
                    raise
        return client_updates
