"""feature-election: Federated feature selection with Flower.

ServerApp entry point that configures and launches the FeatureElectionStrategy.
"""

import logging

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from feature_election.strategy import FeatureElectionStrategy

logger = logging.getLogger(__name__)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the Feature Election + FL ServerApp."""
    # Read Configuration
    run_config = context.run_config

    skip_feature_election = bool(run_config.get("skip-feature-election", False))
    auto_tune = bool(run_config.get("auto-tune", False))
    tuning_rounds = int(str(run_config.get("tuning-rounds", 0)))
    freedom_degree = float(run_config.get("freedom-degree", 0.5))
    aggregation_mode = str(run_config.get("aggregation-mode", "weighted"))

    num_rounds = int(str(run_config.get("num-rounds", 5)))
    fraction_train = float(run_config.get("fraction-train", 1.0))
    fraction_evaluate = float(run_config.get("fraction-evaluate", 1.0))
    min_train_nodes = int(str(run_config.get("min-train-nodes", 2)))
    min_evaluate_nodes = int(str(run_config.get("min-evaluate-nodes", 2)))

    # Initialize Strategy with all configuration
    strategy = FeatureElectionStrategy(
        freedom_degree=freedom_degree,
        tuning_rounds=tuning_rounds if (auto_tune and not skip_feature_election) else 0,
        aggregation_mode=aggregation_mode,
        auto_tune=auto_tune,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=min_evaluate_nodes,
        accept_failures=True,
        skip_feature_election=skip_feature_election,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(),
        num_rounds=num_rounds,
        timeout=3600,
    )

    # Log final result summary
    logger.info(f"Workflow completed. Result: {result}")
