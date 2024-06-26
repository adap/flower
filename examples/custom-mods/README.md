---
title: Example Flower App with Custom Mods
labels: [mods, monitoring, app]
dataset: [CIFAR-10]
framework: [torch, wandb, tensorboard, torchvision]
---

# Using custom mods 🧪

> 🧪 = This example covers experimental features that might change in future versions of Flower
> Please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)) to learn how to use Flower with PyTorch.

The following steps describe how to write custom Flower Mods and use them in a simple example.

## Writing custom Flower Mods

### Flower Mods basics

As described [here](https://flower.ai/docs/framework/how-to-use-built-in-mods.html#what-are-mods), Flower Mods in their simplest form can be described as:

```python
def basic_mod(msg: Message, context: Context, app: ClientApp) -> Message:
    # Do something with incoming Message (or Context)
    # before passing to the inner ``ClientApp``
    reply = app(msg, context)
    # Do something with outgoing Message (or Context)
    # before returning
    return reply
```

and used when defining the `ClientApp`:

```python
app = fl.client.ClientApp(
    client_fn=client_fn,
    mods=[basic_mod],
)
```

Note that in this specific case, this mod won't modify anything, and perform FL as usual.

### WandB Flower Mod

If we want to write a mod to monitor our client-side training using [Weights & Biases](https://github.com/wandb/wandb), we can follow the steps below.

First, we need to initialize our W&B project with the correct parameters:

```python
wandb.init(
    project=...,
    group=...,
    name=...,
    id=...,
    resume="allow",
    reinit=True,
)
```

In our case, the group should be the `run_id`, specific to a `ServerApp` run, and the `name` should be the `node_id`. This will make it easy to navigate our W&B project, as for each run we will be able to see the computed results as a whole or for each individual client.

The `id` needs to be unique, so it will be a combination of `run_id` and `node_id`.

In the end we have:

```python
def wandb_mod(msg: Message, context: Context, app: ClientAppCallable) -> Message:
    run_id = msg.metadata.run_id
    group_name = f"Run ID: {run_id}"

    node_id = str(msg.metadata.dst_node_id)
    run_name = f"Node ID: {node_id}"

    wandb.init(
        project="Mod Name",
        group=group_name,
        name=run_name,
        id=f"{run_id}_{node_id}",
        resume="allow",
        reinit=True,
    )
```

Now, before the message is processed by the server, we will store the starting time and the round number, in order to compute the time it took the client to perform its fit step.

```python
server_round = int(msg.metadata.group_id)
start_time = time.time()
```

And then, we can send the message to the client:

```python
reply = app(msg, context)
```

And now, with the message we got back, we can gather our metrics:

```python
if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():

    time_diff = time.time() - start_time

    metrics = reply.content.configs_records

    results_to_log = dict(metrics.get("fitres.metrics", ConfigsRecord()))
    results_to_log["fit_time"] = time_diff
```

Note that we store our metrics in the `results_to_log` variable and that we only initialize this variable when our client is sending back fit results (with content in it).

Finally, we can send our results to W&B using:

```python
wandb.log(results_to_log, step=int(server_round), commit=True)
```

The complete mod becomes:

```python
def wandb_mod(msg: Message, context: Context, app: ClientAppCallable) -> Message:
    server_round = int(msg.metadata.group_id)

    if reply.metadata.message_type == MessageType.TRAIN and server_round == 1:
        run_id = msg.metadata.run_id
        group_name = f"Run ID: {run_id}"

        node_id = str(msg.metadata.dst_node_id)
        run_name = f"Node ID: {node_id}"

        wandb.init(
            project="Mod Name",
            group=group_name,
            name=run_name,
            id=f"{run_id}_{node_id}",
            resume="allow",
            reinit=True,
        )

    start_time = time.time()

    reply = app(msg, context)

    if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():

        time_diff = time.time() - start_time

        metrics = reply.content.configs_records

        results_to_log = dict(metrics.get("fitres.metrics", ConfigsRecord()))

        results_to_log["fit_time"] = time_diff

        wandb.log(results_to_log, step=int(server_round), commit=True)

    return reply
```

And it can be used like:

```python
app = fl.client.ClientApp(
    client_fn=client_fn,
    mods=[wandb_mod],
)
```

If we want to pass an argument to our mod, we can use a wrapper function:

```python
def get_wandb_mod(name: str) -> Mod:
    def wandb_mod(msg: Message, context: Context, app: ClientAppCallable) -> Message:
        server_round = int(msg.metadata.group_id)

        run_id = msg.metadata.run_id
        group_name = f"Run ID: {run_id}"

        node_id = str(msg.metadata.dst_node_id)
        run_name = f"Node ID: {node_id}"

        wandb.init(
            project=name,
            group=group_name,
            name=run_name,
            id=f"{run_id}_{node_id}",
            resume="allow",
            reinit=True,
        )

        start_time = time.time()

        reply = app(msg, context)

        if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():

            time_diff = time.time() - start_time

            metrics = reply.content.configs_records

            results_to_log = dict(metrics.get("fitres.metrics", ConfigsRecord()))

            results_to_log["fit_time"] = time_diff

            wandb.log(results_to_log, step=int(server_round), commit=True)

        return reply

    return wandb_mod
```

And use it like:

```python
app = fl.client.ClientApp(
    client_fn=client_fn,
    mods=[
        get_wandb_mod("Custom mods example"),
     ],
)
```

### TensorBoard Flower Mod

The [TensorBoard](https://www.tensorflow.org/tensorboard) Mod will only differ in the initialization and how the data is sent to TensorBoard:

```python
def get_tensorboard_mod(logdir) -> Mod:
    os.makedirs(logdir, exist_ok=True)

    def tensorboard_mod(
        msg: Message, context: Context, app: ClientAppCallable
    ) -> Message:
        logdir_run = os.path.join(logdir, str(msg.metadata.run_id))

        node_id = str(msg.metadata.dst_node_id)

        server_round = int(msg.metadata.group_id)

        start_time = time.time()

        reply = app(msg, context)

        time_diff = time.time() - start_time

        if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():
            writer = tf.summary.create_file_writer(os.path.join(logdir_run, node_id))

            metrics = dict(
                reply.content.configs_records.get("fitres.metrics", ConfigsRecord())
            )

            with writer.as_default(step=server_round):
                tf.summary.scalar(f"fit_time", time_diff, step=server_round)
                for metric in metrics:
                    tf.summary.scalar(
                        f"{metric}",
                        metrics[metric],
                        step=server_round,
                    )
                writer.flush()

        return reply

    return tensorboard_mod
```

For the initialization, TensorBoard uses a custom directory path, which can, in this case, be passed as an argument to the wrapper function.

It can be used in the following way:

```python
app = fl.client.ClientApp(
    client_fn=client_fn,
    mods=[get_tensorboard_mod(".runs_history/")],
)
```

## Running the example

### Preconditions

Let's assume the following project structure:

```bash
$ tree .
.
├── client.py           # <-- contains `ClientApp`
├── server.py           # <-- contains `ServerApp`
├── task.py             # <-- task-specific code (model, data)
└── requirements.txt    # <-- dependencies
```

### Install dependencies

```bash
pip install -r requirements.txt
```

For [W&B](https://wandb.ai) you will also need a valid account.

### Start the long-running Flower server (SuperLink)

```bash
flower-superlink --insecure
```

### Start the long-running Flower client (SuperNode)

In a new terminal window, start the first long-running Flower client using:

```bash
flower-client-app client:wandb_app --insecure
```

for W&B monitoring, or:

```bash
flower-client-app client:tb_app --insecure
```

for TensorBoard.

In yet another new terminal window, start the second long-running Flower client (with the mod of your choice):

```bash
flower-client-app client:{wandb,tb}_app --insecure
```

### Run the Flower App

With both the long-running server (SuperLink) and two clients (SuperNode) up and running, we can now run the actual Flower App:

```bash
flower-server-app server:app --insecure
```

### Check the results

For W&B, you will need to login to the [website](https://wandb.ai).

For TensorBoard, you will need to run the following command in your terminal:

```sh
tensorboard --logdir <LOG_DIR>
```

Where `<LOG_DIR>` needs to be replaced by the directory passed as an argument to the wrapper function (`.runs_history/` by default).
