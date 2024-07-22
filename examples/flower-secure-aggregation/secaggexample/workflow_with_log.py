from logging import DEBUG, INFO

import numpy as np

import flwr.common.recordset_compat as compat
from flwr.common import Context, log, parameters_to_ndarrays
from flwr.common.logger import update_console_handler
from flwr.common.secure_aggregation.quantization import quantize
from flwr.server import Driver, LegacyContext
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.server.workflow.secure_aggregation.secaggplus_workflow import (
    SecAggPlusWorkflow,
    WorkflowState,
)

update_console_handler(DEBUG, True, True)


class SecAggPlusWorkflowWithLogs(SecAggPlusWorkflow):
    """The SecAggPlusWorkflow augmented for this example.

    This class includes additional logging and modifies one of the FitIns to instruct
    the target client to simulate a dropout.
    """

    node_ids = []

    def __call__(self, driver: Driver, context: Context) -> None:
        _quantized = quantize(
            [np.ones(3) for _ in range(5)], self.clipping_range, self.quantization_range
        )
        log(INFO, "")
        log(
            INFO,
            "################################ Introduction ################################",
        )
        log(
            INFO,
            "In the example, each client will upload a vector [1.0, 1.0, 1.0] instead of",
        )
        log(INFO, "model updates for demonstration purposes.")
        log(
            INFO,
            "Client 0 is configured to drop out before uploading the masked vector.",
        )
        log(INFO, "After quantization, the raw vectors will look like:")
        for i in range(1, 5):
            log(INFO, "\t%s from Client %s", _quantized[i], i)
        log(
            INFO,
            "Numbers are rounded to integers stochastically during the quantization",
        )
        log(INFO, ", and thus entries may not be identical.")
        log(
            INFO,
            "The above raw vectors are hidden from the driver through adding masks.",
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
            "Weighted average of vectors (dequantized): %s",
            ndarrays[0],
        )
        log(
            INFO,
            "########################### Secure Aggregation End ###########################",
        )
        log(INFO, "")

    def setup_stage(
        self, driver: Driver, context: LegacyContext, state: WorkflowState
    ) -> bool:
        ret = super().setup_stage(driver, context, state)
        self.node_ids = list(state.active_node_ids)
        state.nid_to_fitins[self.node_ids[0]].configs_records["fitins.config"][
            "drop"
        ] = True
        return ret

    def collect_masked_vectors_stage(
        self, driver: Driver, context: LegacyContext, state: WorkflowState
    ) -> bool:
        ret = super().collect_masked_vectors_stage(driver, context, state)
        for node_id in state.sampled_node_ids - state.active_node_ids:
            log(INFO, "Client %s dropped out.", self.node_ids.index(node_id))
        log(INFO, "Obtained sum of masked vectors: %s", state.aggregate_ndarrays[1])
        return ret
