# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Legacy default workflows."""


import timeit

from logging import INFO
from flwr.common import log
from typing import Optional
from ..typing import Workflow
from ..compat.legacy_context import LegacyContext
from ..driver import Driver
from flwr.common import Context


class DefaultWorkflow:
    """Default FL workflow factory in Flower."""

    def __init__(
        self,
        fit_workflow: Optional[Workflow] = None,
        evaluate_workflow: Optional[Workflow] = None,
    ):
        ...

    def __call__(self, driver: Driver, context: Context) -> None:
        """Create the workflow."""
        if not isinstance(context, LegacyContext):
            raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")
        
        # Initialize parameters
        ...

        # Run federated learning for num_rounds
		log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, state.num_rounds + 1):
            state.current_round = current_round

            # Fit round
			self.fit_workflow(driver, context)

            # Centralized evaluation
            ...
 
            # Evaluate round
			self.evaluate_workflow(driver, context)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)

