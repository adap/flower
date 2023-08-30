"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

from logging import DEBUG, INFO
from flwr.common.logger import log
import timeit
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.server import Server, _handle_finished_future_after_evaluate, evaluate_client, evaluate_clients
from flwr.common.typing import Code, EvaluateIns, EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import EvaluateResultsAndFailures
from flwr.server.strategy import Strategy


class PowerOfChoiceServer(Server):

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Estimate local losses of a subset of available clients
            res_first_phase = self.evaluate_first_phase(current_round, timeout=timeout)

            # Set the res_loss of the first phase of the round in the strategy
            self.strategy.set_res_first_phase(res_first_phase)

            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
    
    def evaluate_first_phase(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[List[Tuple[ClientProxy, EvaluateRes]]
    ]:
        """Estimate local losses on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
            first_phase=True,
        )
        if not client_instructions:
            log(INFO, "evaluate_first_phase %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_first_phase %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_first_phase of round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Return results with client and losses
        return results

class PowerOfChoiceCommAndCompVariant(Server):

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting with RPOW variant")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Sample d clients from the available clients
            self.strategy.sample_clients(self._client_manager)

            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, (results, failures) = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
                # Update the Atmp list with the losses returned by the clients
                if results is not None:
                    self.strategy.update_atmp(results)
                    log(
                        INFO,
                        "Atmp updated: %s \n with results \n %s",
                        self.strategy.atmp,
                        results,
                    )
                    

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history