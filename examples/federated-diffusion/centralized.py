import time

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from utils import IPR, make_grid, prepare_tensors


def get_model():
    model = UNet2DModel(
        sample_size=32,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        freq_shift=1,
        flip_sin_to_cos=False,
        block_out_channels=(
            128,
            256,
            256,
            256,
        ),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
        downsample_padding=0,
        attention_head_dim=None,
        norm_eps=1e-06,
    )
    return model


def train(args, model, train_dataloader, cid, server_round, cpu):
    learning_rate = 1e-4
    lr_warmup_steps = int(args.num_inference_steps / 2)
    mixed_precision = "fp16"
    gradient_accumulation_steps = 1
    save_image_epochs = 50

    # Set group and job_type to see auto-grouping in the UI
    wandb.init(
        project="Diffusion-Cifar10",
        group="Rnd-" + str(server_round),
        # track hyperparameters and run metadata
        config={
            "N_Clients": args.num_clients,
            "N_ServerRnds": args.num_rounds,
            "T": args.num_inference_steps,
            "Scheduler": "cosine",
            "AggregationStrategy": "FedAvg",
            "Dataset": "CIFAR-10",
            "Epochs": args.num_epochs,
        },
    )

    wandb.run.name = "Client-" + cid

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_inference_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        cpu=cpu,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    # Now you train the model

    for epoch in range(args.num_epochs):
        # log metrics to wandb
        wandb.log({"Epoch": epoch, "cid": int(cid), "Server round": server_round})

        for _, batch in enumerate(train_dataloader):
            clean_images = batch[0]  # 0 index is images, 1 index is label
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # log metrics to wandb
            wandb.log(
                {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
            )

            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )
            pipeline.set_progress_bar_config(disable=True)
            if (epoch + 1) % save_image_epochs == 0 or epoch == args.num_epochs - 1:
                eval_batch_size = 16
                seed = 0

                # Sample some images from random noise (this is the backward diffusion process).
                # The default pipeline output type is `List[PIL.Image]`
                images = pipeline(
                    batch_size=eval_batch_size,
                    generator=torch.manual_seed(seed),
                    num_inference_steps=args.num_inference_steps,
                ).images

                # Make a grid out of the images
                image_grid = make_grid(images, rows=4, cols=4)

                try:
                    images = wandb.Image(image_grid)
                    wandb.log({"image_grid": images})
                except:
                    print("Could not save images in wandb")


def validate(args, model, cid, device):
    mixed_precision = "fp16"
    gradient_accumulation_steps = 1

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_inference_steps)
    pipeline = DDPMPipeline(
        unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    val_dataset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    subset_size = 1000
    indices = torch.arange(len(val_dataset))
    torch.manual_seed(0)
    indices = indices[torch.randperm(len(indices))]
    subset_indices = indices[:subset_size]

    val_dataset.data = val_dataset.data[subset_indices]
    val_dataset.targets = [val_dataset.targets[i] for i in subset_indices]
    cifar10_1k_val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size_test, shuffle=True
    )

    eval_batch_size = 100  # 100
    all_images = []
    for _ in tqdm(range(10)):  # 10
        images = pipeline(
            batch_size=eval_batch_size,
            generator=torch.manual_seed(int(time.time())),
            num_inference_steps=args.num_inference_steps,
        ).images
        all_images.append(images)
    gen_images = [item for sublist in all_images for item in sublist]
    # Make a grid out of the images
    orig_tensor, gen_tensor, _ = prepare_tensors(
        cifar10_1k_val_dataloader, gen_images, num=subset_size
    )

    ipr = IPR(
        args.batch_size_test, 3, subset_size, device=device
    )  # args.batch_size, args.k, args.num_samples
    if device == "cuda":
        ipr.compute_manifold_ref(
            orig_tensor.float().cuda()
        )  # args.path_real can be either directory or pre-computed manifold file
        metric = ipr.precision_and_recall(gen_tensor.float().cuda())
    else:
        ipr.compute_manifold_ref(
            orig_tensor.float()
        )  # args.path_real can be either directory or pre-computed manifold file
        metric = ipr.precision_and_recall(gen_tensor.float())
    print("precision =", metric.precision, " Cid: ", cid)
    print("recall =", metric.recall, " Cid: ", cid)

    return metric.precision, metric.recall, subset_size
