"""Federated Learning Provenance Module.

This module provides functionality for tracking and analyzing provenance in federated
learning systems. It includes functions for computing round-level provenance metrics and
analyzing client contributions to the global model.
"""

import logging
from typing import Any, Dict

from tracefl.fl_provenance_modules import FederatedProvTrue


# --------------------------------------------------------------------
# round_lambda_prov
# --------------------------------------------------------------------`
def round_lambda_prov(
    train_cfg: Any,
    prov_cfg: Any,
    prov_global_model: Any,
    client2model: Dict[str, Any],
    client2num_examples: Dict[str, int],
    *,
    all_rounds_clients2class: Dict[str, Dict[int, int]],
    central_test_data: Any,
    server_round: int,
) -> Dict[str, Any]:
    """Execute a round of provenance analysis for federated learning.

    Args:
        train_cfg: Training configuration
        prov_cfg: Provenance configuration
        prov_global_model: The global model for provenance analysis
        client2model: Dictionary mapping client IDs to their models
        client2num_examples: Dictionary mapping client IDs to number of examples
        all_rounds_clients2class: Dictionary mapping client IDs to class distributions
        central_test_data: Test dataset for evaluation
        server_round: Current server round number

    Returns
    -------
        dict: Provenance analysis results
    """
    round_prov = FederatedProvTrue(
        train_cfg=train_cfg,
        prov_cfg=prov_cfg,
        round_key=str(server_round),
        server_test_data=central_test_data,
        client2model=client2model,
        client2num_examples=client2num_examples,
        prov_global_model=prov_global_model,
        all_rounds_clients2class=all_rounds_clients2class,
        t=None,
    )

    try:
        prov_result_dict = round_prov.run()
    except (ValueError, KeyError, RuntimeError) as e:
        logging.error(
            "Error in running provenance for round %s. Error: %s", server_round, e
        )
        prov_result_dict = {"Error": str(e)}

    return prov_result_dict
