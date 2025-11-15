import torch
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusionPipeline
from flwr_datasets import FederatedDataset
from peft import LoraConfig, get_peft_model
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms
from random import sample

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
fds = None


def collate_fn(batch):
    pixel_values = torch.stack([torch.as_tensor(item["pixel_values"]) for item in batch])
    return {"pixel_values": pixel_values}

def load_data(
        partition_id: int,
        num_partitions: int,
        image_size: int = 64,
        batch_size: int = 16
) -> tuple[DataLoader, DataLoader]:
    """Load image data for diffusion model training"""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner}
        )
    partition = fds.load_partition(partition_id)

    def transform_function(examples):
        # For MNIST, convert grayscale → RGB
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # always 3 channels
            transforms.Resize((64, 64)),                  # match UNet input size
            transforms.ToTensor(),                        # convert to tensor [0,1]
        ])

        # handle both CIFAR-like and MNIST-like keys
        key = "img" if "img" in examples else "image"

        images = []
        for img in examples[key]:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            images.append(preprocess(img))
        return {"pixel_values": images}

    partition = partition.map(
        transform_function,
        batched=True,
        remove_columns=["image", "label"]  # Remove original columns
    )
    partition_train_test = partition.train_test_split(test_size=0.20, seed=42)

    def limit_dataset(dataset, n_samples):
        n = min(n_samples, len(dataset))
        indices = sample(range(len(dataset)), n)
        return dataset.select(indices)

    partition_train_test["train"] = limit_dataset(partition_train_test["train"], 1000)
    partition_train_test["test"] = limit_dataset(partition_train_test["test"], 200)


    trainload = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )

    testload = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )
    print(f"Partition {partition_id}: {len(partition_train_test['train'])} training samples, "f"{len(partition_train_test['test'])} test samples")

    return trainload, testload


def get_lora_model(base_model: str, device: torch.device):
    """Load Stable Diffusion model with memory-optimized LoRA adapters."""
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )

    pipe = enable_memory_efficient_attention(pipe)
    pipe.to(device)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Only attention layers
        lora_dropout=0.0,
        bias="none",
    )

    pipe.unet = get_peft_model(pipe.unet, lora_config)
    print(f"Trainable parameters: {sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)}")
    return pipe


def enable_memory_efficient_attention(pipe):
    """Enable memory-efficient attention mechanisms."""
    try:
        pipe.unet.set_use_memory_efficient_attention_xformers(True)
    except:
        try:
            pipe.unet.enable_attention_slicing()
        except:
            pass

    # Enable CPU offloading for VAE and text encoder
    try:
        pipe.vae.enable_slicing()
        pipe.enable_attention_slicing()
    except:
        pass

    return pipe

def train_lora_step(pipe, dataloader, device):
    """Perform a single memory-efficient LoRA update."""
    pipe.unet.train()
    pipe.unet.requires_grad_(True)

    # Only optimize LoRA parameters
    lora_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(lora_params, lr=1e-4)

    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        images = batch["pixel_values"]
        if isinstance(images, list):
            images = torch.stack(images)  # Convert list to tensor

        images = images.to(device)
        optimizer.zero_grad()
        print("Inside training loop.")

        # Encode images to latents (diffusion model training)
        with torch.no_grad():
            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215

        # Sample random timestep and noise
        timestep = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (images.size(0),), device=device).long()
        noise = torch.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)

        # UNet forward pass (use empty text embeddings for unconditional training)
        text_embeddings = torch.zeros(images.size(0), 77, 768, device=device)
        noise_pred = pipe.unet(noisy_latents, timestep, encoder_hidden_states=text_embeddings).sample

        loss = nn.functional.mse_loss(noise_pred, noise)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_lora_step(pipe, dataloader, device):
    """Evaluate LoRA adapters with minimal memory usage."""
    pipe.unet.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"]
            print("Inside evaluation loop.")

            # 🔧 Handle both Tensor and list cases
            if isinstance(pixel_values, torch.Tensor):
                images = pixel_values.to(device)
            elif isinstance(pixel_values, list):
                images = torch.stack(pixel_values).to(device)
            else:
                raise TypeError(f"Unexpected batch type: {type(pixel_values)}")

            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
            timestep = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (images.size(0),), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)

            text_emb = torch.zeros(images.size(0), 77, 768, device=device)
            noise_pred = pipe.unet(noisy_latents, timestep, encoder_hidden_states=text_emb).sample

            loss = nn.functional.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


