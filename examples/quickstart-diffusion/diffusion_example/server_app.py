"""Memory-optimized Flower server for federated LoRA fine-tuning."""
from collections import OrderedDict

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from diffusion_example.task import get_lora_model

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Load base model and extract initial LoRA parameters only
    base_model = context.run_config.get("base-model", "runwayml/stable-diffusion-v1-5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model just to get LoRA structure
    pipe = get_lora_model(base_model, device)

    # Extract only LoRA parameters for federation
    lora_state_dict = OrderedDict()
    for key, param in pipe.unet.state_dict().items():
        if "lora" in key and param.requires_grad:
            lora_state_dict[key] = param.cpu().numpy()

    initial_arrays = ArrayRecord(lora_state_dict)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Configure training strategy
    fraction_train = float(context.run_config.get("fraction-train", 1.0))
    fraction_evaluate = float(context.run_config.get("fraction-evaluate", 0.0))
    num_rounds = int(context.run_config.get("num-server-rounds", 2))

    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # Start the federated training
    print(f"\n Starting federated diffusion training for {num_rounds} rounds...")
    print(f" Using base model: {base_model}")
    print(f" Training LoRA parameters only ({len(lora_state_dict)} layers)")

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )

    # Save final LoRA weights
    print("\n Saving final federated LoRA weights...")
    final_state = result.arrays.to_torch_state_dict()
    torch.save(final_state, "final_lora_model.pt")
    # print(" Federated LoRA model saved successfully!")
