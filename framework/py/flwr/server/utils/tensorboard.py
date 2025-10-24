# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower TensorBoard utilities."""


import os
from datetime import datetime
from logging import WARN
from typing import Callable, Optional, Union, cast

from flwr.common import EvaluateRes, Scalar
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

try:
    import tensorflow as TF
except ModuleNotFoundError:
    TF = None

MISSING_EXTRA_TF = """
Extra dependency required for using tensorboard are missing.
The program will continue without tensorboard.

In order to use tensorboard, install `tensorflow` with the following command:

    `pip install tensorflow`.
"""


def tensorboard(logdir: str) -> Callable[[Strategy], Strategy]:
    """TensorBoard logger for Flower strategies.

    It will log loss, num_examples and all metrics which are of type float or int.

    This can either be used as a decorator as shown in the example variant 1
    or directly as shown in the example variant 2.

    Example:
        # Variant 1
        @tensorboard(logdir=LOGDIR)
        class CustomStrategy(FedAvg):
            pass

        strategy = CustomStrategy()

        # Variant 2
        strategy = tensorboard(logdir=LOGDIR)(FedAvg)()
    """
    print(
        "\n\t\033[32mStart TensorBoard with the following parameters"
        f"\n\t$ tensorboard --logdir {logdir}\033[39m\n"
    )
    # Create logdir if it does not yet exist
    os.makedirs(logdir, exist_ok=True)

    # To allow multiple runs and group those we will create a subdir
    # in the logdir which is named as number of directories in logdir + 1
    run_id = str(
        len(
            [
                name
                for name in os.listdir(logdir)
                if os.path.isdir(os.path.join(logdir, name))
            ]
        )
    )
    run_id = run_id + "-" + datetime.now().strftime("%Y%m%dT%H%M%S")
    logdir_run = os.path.join(logdir, run_id)

    def decorator(strategy_class: Strategy) -> Strategy:
        """Return overloaded Strategy Wrapper."""
        if TF is None:
            log(WARN, MISSING_EXTRA_TF)
            return strategy_class

        class TBWrapper(strategy_class):  # type: ignore
            """Strategy wrapper that hooks into some methods for TensorBoard logging."""

            def aggregate_evaluate(
                self,
                server_round: int,
                results: list[tuple[ClientProxy, EvaluateRes]],
                failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> tuple[Optional[float], dict[str, Scalar]]:
                """Hooks into aggregate_evaluate for TensorBoard logging purpose."""
                # Execute decorated function and extract results for logging
                # They will be returned at the end of this function but also
                # used for logging
                loss_aggregated, config = super().aggregate_evaluate(
                    server_round,
                    results,
                    failures,
                )

                # Server logs
                writer = TF.summary.create_file_writer(
                    os.path.join(logdir_run, "server")
                )

                # Write aggregated loss
                with writer.as_default(
                    step=server_round
                ):  # pylint: disable=not-context-manager
                    TF.summary.scalar(
                        "server/loss_aggregated", loss_aggregated, step=server_round
                    )
                    writer.flush()

                if len(results) == 0:
                    return loss_aggregated, config

                # Client logs
                for client, evaluate_res in results:
                    loss, num_examples, metrics = (
                        evaluate_res.loss,
                        evaluate_res.num_examples,
                        evaluate_res.metrics,
                    )

                    writer = TF.summary.create_file_writer(
                        os.path.join(logdir_run, "clients", client.cid)
                    )
                    with writer.as_default(  # pylint: disable=not-context-manager
                        step=server_round
                    ):
                        TF.summary.scalar("clients/loss", loss)
                        TF.summary.scalar("clients/num_examples", num_examples)
                        if metrics is not None:
                            for key, value in metrics.items():
                                if type(value) in [int, float]:
                                    TF.summary.scalar(f"clients/{key}", value)
                        writer.flush()

                return loss_aggregated, config

        return cast(Strategy, TBWrapper)

    return decorator
