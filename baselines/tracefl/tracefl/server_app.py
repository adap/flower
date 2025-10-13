"""tracefl-baseline: A Flower Baseline."""

import logging
import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from tracefl.model import initialize_model
from tracefl.dataset import get_clients_server_data
from tracefl.config import create_tracefl_config
from tracefl.strategy import TraceFL_Strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create ServerApp
app = ServerApp()

# Global variable to store server data
_server_data = None
_client2class = None


def _load_server_data(context):
    """Load server test data and client label mappings using TraceFL preparation."""
    global _server_data, _client2class
    if _server_data is None or _client2class is None:
        # Create TraceFL config from Flower context
        cfg = create_tracefl_config(context)

        # Load dataset
        ds_dict = get_clients_server_data(cfg)

        # Store server test data
        _server_data = ds_dict["server_testdata"]
        _client2class = ds_dict.get("client2class", {})

        print(f"📊 Server loaded {len(_server_data)} test samples")

    return _server_data, _client2class


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]

    # Load TraceFL server data
    server_data, client2class = _load_server_data(context)
    
    # Get TraceFL config
    cfg = create_tracefl_config(context)
    
    # Log architecture detection results
    print(f"🔧 Architecture Detection:")
    print(f"   📊 Dataset: {cfg.data_dist.dname} ({cfg.data_dist.architecture})")
    print(f"   🤖 Model: {cfg.data_dist.model_name} ({cfg.data_dist.model_architecture})")
    print(f"   📐 Classes: {cfg.data_dist.num_classes}, Channels: {cfg.data_dist.channels}")
    print(f"   ✅ Compatibility: {cfg.data_dist.model_architecture} + {cfg.data_dist.architecture}")

    # Load global model
    # Initialize model based on configuration
    model_dict = initialize_model(cfg.data_dist.model_name, cfg.data_dist)
    global_model = model_dict["model"]
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize TraceFL strategy
    strategy = TraceFL_Strategy(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_available_nodes=2,
        provenance_rounds=cfg.provenance_rounds,
        enable_beta=cfg.enable_beta,
        client_weights_normalization=cfg.client_weights_normalization,
        cfg=cfg,
    )
    
    # Set server test data for provenance analysis
    strategy.set_server_test_data(server_data)
    strategy.set_client2class(client2class)

    # Start strategy, run TraceFL for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    
    # Print TraceFL configuration info
    print(f"\n🎯 TraceFL Configuration:")
    print(f"📊 Dataset: {cfg.data_dist.dname}")
    print(f"👥 Clients: {cfg.data_dist.num_clients}")
    print(f"📈 Distribution: {cfg.data_dist.dist_type}")
    print(f"🔢 Dirichlet Alpha: {cfg.data_dist.dirichlet_alpha}")
    print(f"💾 Max per client: {cfg.data_dist.max_per_client_data_size}")
    print(f"🖥️  Server test data: {len(server_data)} samples")
    print(f"🔍 Provenance rounds: {cfg.provenance_rounds}")
    print(f"⚡ Enable beta (layer importance): {cfg.enable_beta}")
    print(f"📏 Client weights normalization: {cfg.client_weights_normalization}")
    print(f"🛡️  Noise multiplier: {cfg.noise_multiplier}")
    print(f"✂️  Clipping norm: {cfg.clipping_norm}")
    print(f"⚠️ Faulty clients: {cfg.faulty_clients_ids}")
    print(f"🔁 Label flip map: {cfg.label2flip}")
