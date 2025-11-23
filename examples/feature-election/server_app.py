"""
Feature Election Server for Flower

Configures the Feature Election strategy with proper parameters.

Default configuration:
- freedom_degree: 0.5 (balance between intersection and union)
- aggregation_mode: 'weighted' (weight by sample count)
- num_rounds: 1 (single round for feature selection)
"""

import json
import logging
from pathlib import Path

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from strategy import FeatureElectionStrategy

logger = logging.getLogger(__name__)


def server_fn(context: Context) -> ServerAppComponents:
    """
    Constructs the server components for Feature Election.
    
    Configuration is loaded from context.run_config or defaults are used.
    """
    
    # =======================================================
    # Feature Election Configuration
    # =======================================================
    
    run_config = context.run_config
    
    # Feature Election parameters
    freedom_degree = run_config.get("freedom-degree", 0.5)
    aggregation_mode = run_config.get("aggregation-mode", "weighted")
    
    # Federated learning parameters
    num_rounds = run_config.get("num-rounds", 1)
    num_clients = run_config.get("num-clients", 10)
    fraction_fit = run_config.get("fraction-fit", 1.0)
    fraction_evaluate = run_config.get("fraction-evaluate", 1.0)
    min_fit_clients = run_config.get("min-fit-clients", 2)
    min_evaluate_clients = run_config.get("min-evaluate-clients", 2)
    
    # Feature selection method (for logging)
    fs_method = run_config.get("fs-method", "lasso")
    
    # =======================================================
    # Create Feature Election Strategy
    # =======================================================
    
    strategy = FeatureElectionStrategy(
        freedom_degree=freedom_degree,
        aggregation_mode=aggregation_mode,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_fit_clients,
        accept_failures=True,
    )
    
    # =======================================================
    # Create Server Config
    # =======================================================
    
    config = ServerConfig(num_rounds=num_rounds)
    
    # =======================================================
    # Print Configuration
    # =======================================================
    
    print("=" * 70)
    print("Feature Election Configuration")
    print("=" * 70)
    print(f"  Freedom degree: {freedom_degree}")
    print(f"  Aggregation mode: {aggregation_mode}")
    print(f"  Feature selection method: {fs_method}")
    print(f"  Number of rounds: {num_rounds}")
    print(f"  Number of clients: {num_clients}")
    print(f"  Fraction fit: {fraction_fit}")
    print(f"  Fraction evaluate: {fraction_evaluate}")
    print("=" * 70)
    
    # =======================================================
    # Define evaluation callback (optional)
    # =======================================================
    
    def evaluate_fn(server_round: int, feature_mask: any) -> tuple:
        """
        Optional server-side evaluation of the global feature mask.
        Can be used to validate the selected features.
        """
        if feature_mask is not None:
            n_selected = int(feature_mask.sum()) if hasattr(feature_mask, 'sum') else 0
            logger.info(f"Round {server_round}: {n_selected} features selected")
        return 0.0, {}
    
    # Uncomment to enable server-side evaluation
    # strategy.evaluate_fn = evaluate_fn
    
    # =======================================================
    # Save results callback (after completion)
    # =======================================================
    
    def on_shutdown():
        """Save results when server shuts down."""
        try:
            results = strategy.get_results()
            output_path = Path("feature_election_results.json")
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"✓ Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    # Note: Flower doesn't have a built-in shutdown callback in ServerAppComponents
    # Results should be saved manually after flwr run completes
    # Or use a custom wrapper to save after the run
    
    return ServerAppComponents(strategy=strategy, config=config)


# Create the ServerApp
app = ServerApp(server_fn=server_fn)
