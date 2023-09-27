# Flower x Diffusers, Federated Diffusion model training

This example demonstrate how to implement federated learning for training diffusion models using `Flower` and `Diffusers`.

## Dependencies

This example requires the following dependencies:

    * accelerate

    * diffusers

    * flwr

    * matplotlib

    * numpy

    * torch

    * torchvision

    * tqdm

    * wandb

Which can be installed using `poetry install` if you are using `poetry` or `pip install -r requirements.txt`.

## Centralized Diffusion model

First, we will consider the centralized setting for training a diffusion model.

### Data

In the centralized setting, we don't have to worry about the partioning, so we will just use the `CIFAR10` function from `torchvision.datasets` and return the `DataLoader`:

```python
def load_data():
    train_batch_size = 4

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, shuffle=True
    )
    return dataloader
```

### Model

Note that most of the code is inspired by this [tutorial](https://huggingface.co/docs/diffusers/tutorials/basic_training).

We first need to define the architecture of our model, to do that we use the `UNet2DModel` class from `diffusers`:

```python
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
```

Then, we can write the training function:

```python
def train(model, train_dataloader, cid, server_round, epochs, timesteps, cpu):
    learning_rate = 1e-4
    lr_warmup_steps = int(PARAMS.num_inference_steps / 2)
    mixed_precision = "fp16"
    gradient_accumulation_steps = 1
    save_image_epochs = 50

    # Weights and biases initialization is skipped...

    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * epochs),
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

    for epoch in range(epochs):
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

            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )
            if (epoch + 1) % save_image_epochs == 0 or epoch == epochs - 1:
                eval_batch_size = 16
                seed = 0

                # Sample some images from random noise (this is the backward diffusion process).
                # The default pipeline output type is `List[PIL.Image]`
                images = pipeline(
                    batch_size=eval_batch_size,
                    generator=torch.manual_seed(seed),
                    num_inference_steps=PARAMS.num_inference_steps,
                ).images

                image_grid = make_grid(images, rows=4, cols=4)

                # Then we can either save the images or log them to Weights and Biases as done in
                # the code.
```

Where we first create our scheduler with `DDPMScheduler` and then instantiate our `AdamW` optimizer. In this training function, we will use the HuggingFace `accelerate` library, in order to make the training simple and efficient on any configuration.

During the training loop itself, we add noise to the images, pass the noisy images through the model, and compute the loss between the predicted noise residual and the actual noise.

Finally we use `acclerate`'s `is_main_process` to run the generation of images with the trained model only once over the clients.

The `validate` function will only focus on the generation of new images and comparing the generated images with the ground truth from the same label:

```python
def validate(model, cid, timesteps, device):
    mixed_precision = "fp16"
    gradient_accumulation_steps = 1

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)
    pipeline = DDPMPipeline(
        unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
    )
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
        val_dataset, batch_size=4, shuffle=True
    )

    eval_batch_size = 100  # 100
    all_images = []
    for _ in tqdm(range(10)):  # 10
        images = pipeline(
            batch_size=eval_batch_size,
            generator=torch.manual_seed(int(time.time())),
            num_inference_steps=PARAMS.num_inference_steps,
        ).images
        all_images.append(images)
    gen_images = [item for sublist in all_images for item in sublist]
    # Make a grid out of the images
    orig_tensor, gen_tensor, _ = prepare_tensors(
        cifar10_1k_val_dataloader, gen_images, num=1000
    )

    ipr = IPR(4, 3, 1000)  # args.batch_size, args.k, args.num_samples
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
```

Note that we use functions defined in `utils.py` to compute the precision and recall from the images. The `IPR` class is from this [GitHub repo](https://github.com/youngjung/improved-precision-and-recall-metric-pytorch).

With those elements we could run a centralized ML pipeline to train our model as such:

```python
trainloader = get_data()
model = get_model()
train(model, trainloader, 1, 1, False)
precision, recall = validate(model, 1, 0, 1000, "cpu")
```

But instead, let's federate it!

## Federating the example

### Data partioning

In order to federate our example, we first need to write a way to paration our dataset, so each client in our simulated environment can have different subset of the data.

We will write 2 functions, one for an IID partitioning, where all clients get a completely random subset of the data, and the other for a non-IID partitioning, where clients get subsets of data containing only a few distinct classes (so for instance, the first client will only have images of planes and horses while the second client will only have images of dogs and cats).

For IID partitioning:

```python
def load_iid_data():
    train_batch_size = 4

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10("./dataset", train=True, download=True, transform=transform)

    # Split training set into N partitions to simulate the individual dataset
    partition_size = len(dataset) // PARAMS.num_clients
    lengths = [partition_size] * PARAMS.num_clients
    datasets = random_split(dataset, lengths)
    trainloaders = []

    for ds in datasets:
        trainloaders.append(
            torch.utils.data.DataLoader(ds, batch_size=train_batch_size, shuffle=True)
        )

    return trainloaders
```

For non-IID partitioning, it is a bit more involved, we will first write a function that generates a dictionnary that assignes subset of classes to clients:

```python
def cifar_extr_noniid(train_dataset, num_users, n_class, num_samples, rate_unbalance):
    num_shards_train, num_imgs_train = int(50000 / num_samples), num_samples
    num_classes = 10

    assert n_class * num_users <= num_shards_train
    assert n_class <= num_classes

    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train * num_imgs_train)
    labels = np.array(train_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (
                        dict_users_train[i],
                        idxs[rand * num_imgs_train : (rand + 1) * num_imgs_train],
                    ),
                    axis=0,
                )
                user_labels = np.concatenate(
                    (
                        user_labels,
                        labels[rand * num_imgs_train : (rand + 1) * num_imgs_train],
                    ),
                    axis=0,
                )
            else:
                dict_users_train[i] = np.concatenate(
                    (
                        dict_users_train[i],
                        idxs[
                            rand
                            * num_imgs_train : int(
                                (rand + rate_unbalance) * num_imgs_train
                            )
                        ],
                    ),
                    axis=0,
                )
                user_labels = np.concatenate(
                    (
                        user_labels,
                        labels[
                            rand
                            * num_imgs_train : int(
                                (rand + rate_unbalance) * num_imgs_train
                            )
                        ],
                    ),
                    axis=0,
                )
            unbalance_flag = 1

    return dict_users_train
```

Then we can just use this function to create the data loaders based on the generated indices:

```python
def load_noniid_data(train_dataset_cifar, user_groups_train_cifar):
    for client_no, array in user_groups_train_cifar.items():
        class_no = []

        for idx in array:
            class_no.append(train_dataset_cifar[int(idx)][1])

    # combine all index list into one nested list
    indices = [val for d in [user_groups_train_cifar] for val in d.values()]
    indices = [list(a) for a in indices]
    indices = [[int(val) for val in sublist] for sublist in indices]

    trainloaders = []

    for index_list in indices:
        subset = Subset(train_dataset_cifar, index_list)
        trainloaders.append(
            torch.utils.data.DataLoader(subset, batch_size=4, shuffle=True)
        )

    return trainloaders
```

Putting everything together, we write a function that provides a simple interface:

```python
def load_datasets(iid=True):
    if iid:
        return load_iid_data()
    else:
        train_dataset_cifar, user_groups_train_cifar = get_dataset_cifar10_noniid(
            PARAMS.num_clients,
            PARAMS.nclass_cifar,
            PARAMS.nsamples_cifar,
            PARAMS.rate_unbalance_cifar,
        )
        return load_noniid_data(train_dataset_cifar, user_groups_train_cifar)
```

### Writing the client

We can now start writing our `FlowerClient`, which is the actual part of the code necessary for Federated Learning.

It will be quite straight forward, as we will only need to write 3 main methods: `get_parameters`, `fit`, and `evaluate`. We will also write a few helper functions to make the code more readable.

The helper functions will be:

```python
def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
```

Those functions will respectively allow us to get the parameters as a list of NumPy arrays from our model, and to update the parameters of our model by providing a new list of NumPy arrays.

The `FlowerClient` itself will be:

```python
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, model, trainloader, cid, timesteps, epochs, device, personalization_layers
    ):
        self.model = model
        self.trainloader = trainloader
        self.cid = cid
        self.timesteps = timesteps
        self.epochs = epochs
        self.device = device
        self.personalization_layers = personalization_layers

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # Update local model parameters
        set_parameters(self.model, parameters)

        # Read values from config
        server_round = config["server_round"]

        cpu = False
        if PARAMS.device == "cpu":
            cpu = True

        # Update local model parameters
        if int(server_round) > 1 and PARAMS.personalized:
            load_personalization_weight(
                self.cid, self.model, self.personalization_layers
            )

        train(
            self.model,
            self.trainloader,
            self.cid,
            server_round,
            self.epochs,
            self.timesteps,
            cpu,
        )
        if PARAMS.personalized:
            save_personalization_weight(self.cid, self.model, self.personalization_layers)

        return get_parameters(self.model), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # Update local model parameters
        set_parameters(self.model, parameters)

        server_round = config["server_round"]

        precision, recall, num_examples = validate(
            self.model, self.cid, self.timesteps, self.device
        )
        results = {
            "precision": precision,
            "recall": recall,
            "cid": self.cid,
            "server_round": server_round,
        }
        json.dump(results, open("logs.json", "a"))

        return (
            1.0,
            num_examples,
            {"precision": precision, "recall": recall, "cid": self.cid},
        )
```

Note that in the above `FlowerClient` we also added personalization. This will allow each client's model to keep a locally trained subset of parameters that won't be affected by the global aggregation. This can be easily disabled by setting `personalized` to `False` inside `conf.py`. For this personalization to work we also added 2 utility functions:

```python
def save_personalization_weight(cid, model, personalization_layers):
    weights = get_parameters(model)
    # save weight
    personalized_weight = weights[len(weights) - personalization_layers :]
    with open(f"Per_{cid}.pickle", "wb") as file_weight:
        pickle.dump(personalized_weight, file_weight)
    file_weight.close()


def load_personalization_weight(cid, model, personalization_layers):
    weights = get_parameters(model)
    with open(f"Per_{cid}.pickle", "rb") as file_weight:
        personalized_weight = pickle.load(file_weight)
        file_weight.close()
    weights[len(weights) - personalization_layers :] = personalized_weight

    # set new weight to the model
    set_parameters(model, weights)
```

The first one saves given weights from a subset of the layers of a model to a file while the second one update given layers of a model with the weights from the same file.

Finally, in order to instantiate our clients, we write our `client_fn`:

```python
def client_fn(cid):
    """Create a Flower client representing a single organization."""

    timesteps = PARAMS.num_inference_steps  # diffusion model decay steps
    epochs = PARAMS.num_epochs  # training epochs

    # Load model
    model = get_model().to(DEVICE)
    personalization_layers = 4

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = TRAINLOADERS[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(
        model,
        trainloader,
        cid,
        timesteps,
        epochs,
        PARAMS.device,
        personalization_layers,
    )
```

Now, we should be all set on the client-side.

### Writing the strategy

On the server-side, we need to define our strategy. This strategy will be a subclass of `flwr.server.strategy.FedAvg` where we add a new `client_manager` parameter and redefine `aggregate_fit` and `aggregate_evaluate`. The only change made to `aggregate_fit` is to save the model every round. `aggregate_evaluate`, on the other hand, is quite different: first of all, we don't compute any loss, so the aggregated_loss will always be 1.0, then, after the 5th round, if a `client_manager` has been supplied to the strategy, we disconnect clients with the lowest precision, we also dump the results inside a log file. This is how everything looks like together:

```python
class SaveModelAndMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        client_manager=None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.client_manager = client_manager
        self.history = History()

    def aggregate_fit(
        self,
        server_round,
        results,  # FitRes is like EvaluateRes and has a metrics key
        failures,
    ):
        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            model = get_model()
            params_dict = zip(model.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            output_dir = "ddpm-cifar10-32-iid-default-flwr/clients4/model"
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                f"ddpm-cifar10-32-iid-default-flwr/clients4/model/model_round_{server_round}.pth",
            )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        aggregated_loss = 1.0
        
        if server_round > 5:
            cid_list = [r.metrics["cid"] for _, r in results]
            precision_list = [r.metrics["precision"] for _, r in results]
            recall_list = [r.metrics["recall"] for _, r in results]
            print(cid_list, precision_list, recall_list, server_round)

            lowest_precision = float('inf')
            lowest_precision_cid = None
            lowest_precision_count = 0
            for i, precision in enumerate(precision_list):
                if float(precision) < float(lowest_precision):
                    lowest_precision = precision
                    lowest_precision_cid = cid_list[i]
                    lowest_precision_count = 1
                elif precision == lowest_precision:
                    lowest_precision_count += 1

            if lowest_precision_count > 1:
                lowest_precision_cid = None
    
            lowest_precision_cid = str(lowest_precision_cid)
            
            client_to_disconnect = None
            for client, evaluate_res in results:
                if evaluate_res.metrics.get('cid') == lowest_precision_cid:
                    client_to_disconnect = client
            
            if lowest_precision_cid == None:
                loss_aggregated, metrics_aggregated = aggregated_loss, {"server_round": server_round}
            else:
                print("client_to_disconnect:", lowest_precision_cid, client_to_disconnect)
                print("====done with agg evaluate======")
                
                data = {"cid_list": cid_list, "precision_list": precision_list, 
                                         "recall_list": recall_list , "lowest_precision_cid": lowest_precision_cid, 
                                         "server_round": server_round, "client_to_disconnect": client_to_disconnect,
                                         "warning_client": 0}
                print(data)
                # Serialize data into file:
                json.dump(data, open("logs.json", 'a' ))

                loss_aggregated, metrics_aggregated = aggregated_loss, {"cid_list": cid_list, "precision_list": precision_list, 
                                         "recall_list": recall_list , "lowest_precision_cid": lowest_precision_cid, 
                                         "server_round": server_round, "client_to_disconnect": client_to_disconnect,
                                         "warning_client": 0}
        else:
            loss_aggregated, metrics_aggregated = aggregated_loss, {"client_to_disconnect": None, "server_round": server_round}

        if self.client_manager:
            print("For Personalization and Threshold Filtering Strategy")
            print("History=====")
            warning_clients = [
                metric
                for _, metric in self.history.metrics_distributed.get(
                    "warning_client", []
                )
            ]
            if len(warning_clients) > 1:
                if int(metrics_aggregated["lowest_precision_cid"]) in warning_clients:
                    print("Disconnecting after 2 attempts ...")
                    if (
                        int(metrics_aggregated["server_round"]) >= 1
                        and metrics_aggregated["client_to_disconnect"] is not None
                    ):
                        client_to_disconnect = metrics_aggregated[
                            "client_to_disconnect"
                        ]
                        lowest_precision_cid = metrics_aggregated[
                            "lowest_precision_cid"
                        ]
                        print(
                            "client_to_disconnect:",
                            lowest_precision_cid,
                            client_to_disconnect,
                        )
                        self.client_manager.unregister(client_to_disconnect)
                        print("=====disconnected=====")
                        all_clients = self.client_manager.all()
                        clients = list(all_clients.keys())
                        print(
                            f"Clients still connected after Server round {metrics_aggregated['server_round']}:{clients}"
                        )
            if (
                int(metrics_aggregated["server_round"]) >= 1
                and metrics_aggregated["client_to_disconnect"] is None
            ):
                print("Nothing to disconnect")

        self.history.add_metrics_distributed(server_round, metrics_aggregated)

        return loss_aggregated, metrics_aggregate
```

Note that the Federated Learning for diffusion model can be implemented with a standard `FedAvg` strategy, this custom strategy is only here to bring extra features (like threshold filtering).

### Starting the simulation

In order to start the simulation, we first instantiate our strategy:

```python
strategy = SaveModelAndMetricsStrategy(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=PARAMS.num_clients,  # Never sample less than 10 clients for training
    min_evaluate_clients=PARAMS.num_clients,  # Never sample less than 5 clients for evaluation
    min_available_clients=PARAMS.num_clients,  # Wait until all 10 clients are available
    on_fit_config_fn=trainconfig,
    on_evaluate_config_fn=trainconfig,
    client_manager=client_manager,
)
```

Note that we are using a custom `client_manager` and a `trainconfig` function defined here:

```python
def trainconfig(server_round):
    """Return training configuration dict for each round."""
    config = {"server_round": server_round}  # The current round of federated learning
    return config


class ClientManager(fl.server.SimpleClientManager):
    def sample(
        self,
        num_clients,
        server_round,
        min_num_clients,
        criterion,
    ):
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)

        # First round and odd rounds
        if server_round >= 1 and num_clients <= 20:
            available_cids_sorted = sorted(available_cids, key=int)
            section = []
            for j in range(4):
                index = ((server_round - 1) * 4 + j) % len(available_cids_sorted)
                section.append(available_cids_sorted[index])
            available_cids = section.copy()
            print("Available cids: ", available_cids)
            return [self.clients[cid] for cid in available_cids]

        if num_clients > 20:
            print(
                "Support for distributed training of UNet in specified hardware for larger clients is unavailable"
            )
```

Then, once all of this is defined, we can finally call our `start_simulation` function:

```python
# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=PARAMS.num_clients,
    # client_resources={"num_cpus": 10, "num_gpus":1},
    config=fl.server.ServerConfig(num_rounds=PARAMS.num_rounds),
    strategy=strategy,
)
```

We commented out the `client_resources` parameters as it will depend on each person's setup.

## Usage

In order to run the training, you can just run (with the dependencies installed):

```bash
python main.py
```

Or, if you are using `poetry`:

```bash
poetry run python main.py
```

You can change the default parameters by modifing the values in `conf.py`.
