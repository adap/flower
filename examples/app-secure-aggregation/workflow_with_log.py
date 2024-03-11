from flwr.server import Driver, LegacyContext
from flwr.server.workflow.secure_aggregation.secaggplus_workflow import (
    SecAggPlusWorkflow,
    WorkflowState,
)
import numpy as np
from flwr.common.secure_aggregation.quantization import quantize


class SecAggPlusWorkflowWithLogs(SecAggPlusWorkflow):
    """The SecAggPlusWorkflow augmented for this example.

    This class includes additional logging and modifies one of the FitIns to instruct
    the target client to simulate a dropout.
    """

    node_ids = []

    def setup_stage(
        self, driver: Driver, context: LegacyContext, state: WorkflowState
    ) -> bool:
        _quantized = quantize(
            [np.ones(3) for _ in range(5)], self.clipping_range, self.quantization_range
        )
        print(
            "\n\n################################ Introduction ################################\n"
            "In the example, each client will upload a vector [1.0, 1.0, 1.0] instead of\n"
            "model updates for demonstration purposes.\n"
            "Client 0 is configured to drop out before uploading the masked vector.\n"
            f"After quantization, the raw vectors will look like:"
        )
        for i in range(1, 5):
            print(f"\t{_quantized[i]} from Client {i}")
        print(
            f"Numbers are rounded to integers stochastically during the quantization\n"
            ", and thus entries may not be identical."
        )
        print(
            "The above raw vectors are hidden from the driver through adding masks.\n"
        )
        print(
            "########################## Secure Aggregation Start ##########################"
        )
        print(f"Sending configurations to 5 clients...")
        ret = super().setup_stage(driver, context, state)
        print(f"Received public keys from {len(state.active_node_ids)} clients.")
        self.node_ids = list(state.active_node_ids)
        state.nid_to_fitins[self.node_ids[0]].configs_records["fitins.config"][
            "drop"
        ] = True
        return ret

    def share_keys_stage(
        self, driver: Driver, context: LegacyContext, state: WorkflowState
    ) -> bool:
        print(f"\nForwarding public keys...")
        ret = super().share_keys_stage(driver, context, state)
        print(
            f"Received encrypted key shares from {len(state.active_node_ids)} clients."
        )
        return ret

    def collect_masked_input_stage(
        self, driver: Driver, context: LegacyContext, state: WorkflowState
    ) -> bool:
        print(f"\nForwarding encrypted key shares and requesting masked vectors...")
        ret = super().collect_masked_input_stage(driver, context, state)
        for node_id in state.sampled_node_ids - state.active_node_ids:
            print(f"Client {self.node_ids.index(node_id)} dropped out.")
        for node_id in state.active_node_ids:
            print(
                f"Received masked vectors from Client {self.node_ids.index(node_id)}."
            )
        print(f"Obtained sum of masked vectors: {state.aggregate_ndarrays[1]}")
        return ret

    def unmask_stage(
        self, driver: Driver, context: LegacyContext, state: WorkflowState
    ) -> bool:
        print("\nRequesting key shares to unmask the aggregate vector...")
        ret = super().unmask_stage(driver, context, state)
        print(f"Received key shares from {len(state.active_node_ids)} clients.")

        print(
            f"Weighted average of vectors (dequantized): {state.aggregate_ndarrays[0]}"
        )
        print(
            "########################### Secure Aggregation End ###########################\n\n"
        )
        return ret
