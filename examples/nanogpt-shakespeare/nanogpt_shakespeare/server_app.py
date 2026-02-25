"""Flower ServerApp for federated NanoGPT."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from nanogpt_shakespeare.task import build_model, load_centralized_data, test, _get_meta

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    num_rounds = int(cfg["num-server-rounds"])
    lr = float(cfg["learning-rate"])
    fraction_evaluate = float(cfg["fraction-evaluate"])

    model = build_model(cfg)
    arrays = ArrayRecord(model.state_dict())

    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=make_evaluate_fn(cfg),
    )

    # Generate a sample from the final model
    final_model = build_model(cfg)
    final_model.load_state_dict(result.arrays.to_torch_state_dict())
    _generate_sample(final_model)


def make_evaluate_fn(cfg: dict):
    """Create a global evaluation function."""
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = build_model(cfg)
        model.load_state_dict(arrays.to_torch_state_dict())
        valloader = load_centralized_data(
            batch_size=int(cfg["batch-size"]),
            block_size=int(cfg["block-size"]),
        )
        loss, ppl = test(model, valloader, "cpu")
        print(f"  [Round {server_round}] val_loss={loss:.4f}  perplexity={ppl:.2f}")
        return MetricRecord({"val_loss": loss, "perplexity": ppl})
    return global_evaluate


def _generate_sample(model, max_tokens: int = 200):
    """Generate Shakespeare-style text from the trained model."""
    meta = _get_meta()
    itos = meta["itos"]
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long)  # start with newline char (id 0)
    output = model.generate(idx, max_new_tokens=max_tokens, temperature=0.8, top_k=40)
    text = "".join(itos[i] for i in output[0].tolist())
    print("\n--- Generated Shakespeare sample ---")
    print(text)
    print("--- End of sample ---\n")
