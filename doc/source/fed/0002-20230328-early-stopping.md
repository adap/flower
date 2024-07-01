---
fed-number: 0002
title: Early stopping
authors: ["@adap"]
creation-date: 2023-03-28
last-updated: 2023-03-28
status: provisional
---

# FED Template

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
<!-- - [Drawbacks](#drawbacks)
- [Alternatives Considered](#alternatives-considered)
- [Appendix](#appendix) -->

## Summary

Currently the training for a model will continue until the number of rounds set at the beginning is atteined.
This means that even if a model as achived his highest performance it can continue to train without any improvements for many rounds. 

It would be good to have a way to stop the training after a specified number of rounds where the performance has not imporved.

## Motivation

The main motivation behind this FED is this Slack [discussion](https://friendly-flower.slack.com/archives/C01RM6LMKQA/p1678706923101609). Early stopping is something that is already present in many frameworks, it would therefore be important for Flower to seemlessly integrate with it.

### Goals

* Provide a way to specify a number of rounds after which we stop the training if the model has not improved.

* Write an code example with early stoppage implemented. 

### Non-Goals

* Make breaking changes

* Go into the specific of early stoppage (the most important is to provide a way to do it, the specific of the early stoppage should only appear in the example).

## Proposal

Provide a hook which could receive the latest model, centralized and federated evaluation results and then could return a boolean to indicate to stop early. The strategy/hook author would then be responsible for handling and storing the best recent model.

To implement this, we would need to add a new `bool` return value to the `evaluate` function of the `Strategy` class in `src/py/flwr/server/strategy/strategy.py`. 

```python
# Current strategy.py implementation skipped from line 1 to 163
    @abstractmethod
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[
            Union[
                Tuple[float, Dict[str, Scalar]],
                Tuple[float, Dict[str, Scalar], bool]
            ]
        ]:
# Current strategy.py implementation skipped from line 168 to 185
```

Then the `Server` class in `src/py/flwr/server/server.py` would need to get updated accordingly:

```python
# Current server.py implementation skipped from line 1 to 114
            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                if len(res_cen) == 3:
                    loss_cen, metrics_cen, stop_training = res_cen
                else:
                    loss_cen, metrics_cen, stop_training = (*res_cen, False)
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
                if stop_training:
                    end_time = timeit.default_timer()
                    elapsed = end_time - start_time
                    log(INFO, "FL finished in %s", elapsed)
                    return history
# Current server.py implementation skipped from line 131 to 442
```

To take a concrete strategy example, if we want to add early stopping to `FedAvg`, we would need to modify the `evaluate` function of the `FedAvg` class in `src/py/flwr/server/strategy/fedavg.py` and the `evaluate_fn` argument type:

```python
# Current fedavg.py implementation skipped from line 1 to 55
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar], bool]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
# Current fedavg.py implementation skipped from line 77 to 155
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar], bool]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics, stop_training = eval_res
        return loss, metrics, stop_training
# Current fedavg.py implementation skipped from line 169 to 276
```

From a user perspective, we would need to define a new class holding a function that we will call from our `evaluate` function, and pass this `evaluate` function to our strategy:

```python
class EarlyStop:
    def __init__(self, patience: int):
        self.best_parameters = None
        self.best_accuracy = 0
        self.count = 0
        self.patience = patience

    def callback(
        self,
        parameters: NDArrays,
        res_cen: Tuple[float, float],
    ) -> bool:
        curr_accuracy = res_cen[1]
        if curr_accuracy > self.best_accuracy:
            self.count = 0
            self.best_parameters = parameters
            self.best_accuracy = curr_accuracy
        else:
            self.count += 1
        
        if self.count > self.patience:
            return True
        else:
            return False

early_stopping = EarlyStop(patience=5)

def evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
) -> Optional[Tuple[float, Dict[str, Scalar], bool]]:

    model = Net()
    set_params(model, parameters)
    loss, accuracy = test(model, testloader, device=device)

    stop_training = early_stopping.callback(parameters, (loss, accuracy))

    return loss, {"accuracy": accuracy}, stop_training

strategy = fl.server.strategy.FedAvg(
    evaluate_fn=evaluate,
)
```

## Drawbacks

* Current provided strategies that redefine the `evaluate` function would need to be rewritten.

* Some type checking might fail for some user with the new change.

## Alternatives Considered

### Passing the callback to the `start_server` function

To implement this, we would need to add a `Callable` argument to the `fit` function of the `Server` class in `src/py/flwr/server/server.py`. 

```python
class Server:
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        callback: Optional[
            Callable[
                [
                    Parameters,
                    Tuple[float, Dict[str, Scalar]],
                    Tuple[
                        Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures
                    ],
                ],
                bool,
            ]
        ] = None,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None
        self.callback: Optional[
            Callable[
                [
                    Parameters,
                    Tuple[float, Dict[str, Scalar]],
                    Tuple[
                        Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures
                    ],
                ],
                bool,
            ]
        ] = callback

    # Current server.py implementation skipped from line 67 to 80
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        # Current server.py implementation skipped from line 82 to 103
        for current_round in range(1, num_rounds + 1):
            # Current server.py implementation skipped from line 105 to 142
            if self.callback and self.callback(self.parameters, res_cen, res_fed):
                end_time = timeit.default_timer()
                elapsed = end_time - start_time
                log(INFO, "FL finished in %s", elapsed)
                return history

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
```

This `Callable` would take as inputs `self.parameters` (the latest model), `res_cen` (the centralized evaluation results), and `res_fed` (the federated evaluation results). It would then return a `bool` that would indicate whether or not the training should be stopped.

This `Callable` argument would need to be passed to the `Server` object, via a `start_server` function argument.

From a user perspective, we would need to define a new class holding a function that we will provide to the `start_server` function:

```python
class EarlyStop:
    def __init__(self, patience: int):
        self.best_parameters = None
        self.best_accuracy = 0
        self.count = 0
        self.patience = patience

    def callback(
        self,
        parameters: Parameters, 
        res_cen: Tuple[float, Dict[str, Scalar]], 
        res_fed: Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ) -> bool:
        curr_accuracy = res_cen[0]
        if curr_accuracy > self.best_accuracy:
            self.count = 0
            self.best_parameters = parameters
            self.best_accuracy = curr_accuracy
        else:
            self.count += 1
        
        if self.count > self.patience:
            return True
        else:
            return False

early_stopping = EarlyStop(patience=5)

# Start Flower server
flwr.server.start_server(
    server_address="0.0.0.0:8080",
    config=flwr.server.ServerConfig(num_rounds=3),
    strategy=strategy,
    callback=early_stopping.callback,
)
```


