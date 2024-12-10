"""secaggexample: A Flower with SecAgg+ app."""

from logging import INFO

import flwr.common.recordset_compat as compat
from flwr.common import Context, log, parameters_to_ndarrays, Message
from flwr.common.secure_aggregation.quantization import quantize
from flwr.server import Driver, LegacyContext
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.server.workflow.secure_aggregation.secaggplus_workflow import (
    SecAggPlusWorkflow,
)
from flwr.common.secure_aggregation.secaggplus_constants import Stage
from flwr.server.workflow.secure_aggregation.secaggplus_aggregator import (
    SecAggPlusAggregatorState,
)

from secaggexample.task import get_weights, make_net


class SecAggPlusWorkflowWithLogs(SecAggPlusWorkflow):
    """The SecAggPlusWorkflow augmented for this example.

    This class includes additional logging and modifies one of the FitIns to instruct
    the target client to simulate a dropout.
    """

    node_ids = []

    def __call__(self, driver: Driver, context: Context) -> None:
        first_3_params = get_weights(make_net())[0].flatten()[:3]
        _quantized = quantize(
            [first_3_params for _ in range(5)],
            self.clipping_range,
            self.quantization_range,
        )
        log(INFO, "")
        log(
            INFO,
            "################################ Introduction ################################",
        )
        log(
            INFO,
            "In the example, clients will skip model training and evaluation",
        )
        log(INFO, "for demonstration purposes.")
        log(
            INFO,
            "Client 0 is configured to drop out before uploading the masked vector.",
        )
        log(INFO, "After quantization, the raw vectors will look like:")
        for i in range(1, 5):
            log(INFO, "\t%s... from Client %s", _quantized[i], i)
        log(
            INFO,
            "Numbers are rounded to integers stochastically during the quantization, ",
        )
        log(INFO, "and thus vectors may not be identical.")
        log(
            INFO,
            "The above raw vectors are hidden from the ServerApp through adding masks.",
        )
        log(INFO, "")
        log(
            INFO,
            "########################## Secure Aggregation Start ##########################",
        )

        super().__call__(driver, context)

        paramsrecord = context.state.parameters_records[MAIN_PARAMS_RECORD]
        parameters = compat.parametersrecord_to_parameters(paramsrecord, True)
        ndarrays = parameters_to_ndarrays(parameters)
        log(
            INFO,
            "Weighted average of parameters (dequantized): %s...",
            ndarrays[0].flatten()[:3],
        )
        log(
            INFO,
            "########################### Secure Aggregation End ###########################",
        )
        log(INFO, "")

    def on_stage_complete(
        self, success: bool, state: SecAggPlusAggregatorState
    ) -> None:
        super().on_stage_complete(success, state)
        if not success:
            return
        if state.current_stage == Stage.SETUP:
            self.node_ids = list(state.sampled_node_ids)
            self._nid_to_fitins[self.node_ids[0]].configs_records["fitins.config"][
                "drop"
            ] = True
        elif state.current_stage == Stage.COLLECT_MASKED_VECTORS:
            for node_id in state.sampled_node_ids - state.active_node_ids:
                log(INFO, "Client %s dropped out.", self.node_ids.index(node_id))
            log(
                INFO,
                "Obtained sum of masked parameters: %s...",
                state.aggregated_vector[1].flatten()[:3],
            )
