---
tags: [finetuning, speech, transformers]
dataset: [SpeechCommands]
framework: [transformers, whisper]
---

# On-device Federated Finetuning for Speech Classification

This example demonstrates how to, from a pre-trained [Whisper](https://openai.com/research/whisper) model, finetune it for the downstream task of keyword spotting. We'll be implementing a federated downstream finetuning pipeline using Flower involving a total of 100 clients. As for the downstream dataset, we'll be using the [Google Speech Commands](https://huggingface.co/datasets/speech_commands) dataset for keyword spotting. We'll take the encoder part of the [Whisper-tiny](https://huggingface.co/openai/whisper-tiny) model, freeze its parameters, and learn a lightweight classification (\<800K parameters !!) head to correctly classify a spoken word.

![Keyword Spotting with Whisper overview](_static/keyword_spotting_overview.png)

This example can be run in three modes:

- **Centralized training**: the standard way of training ML models, where all the data is available to the node doing the finetuning.
- **Federated Learning**: the better way of doing ML, where a model is finetuned collaboratively by nodes (i.e. clients), each using their own data. These clients can run:
  - in _simulation_ mode: a client is an ephemeral Python process with a portion of the system resources assigned to it.
  - in _on-device_ mode: clients are detached entities and each can run on a different device.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/whisper-federated-finetuning  . \
        && rm -rf _tmp \
        && cd whisper-federated-finetuning 
```

This will create a new directory called `whisper-federated-finetuning` containing the following files:

```shell
whisper-federated-finetuning
├── whisper_example
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   ├── model.py        # Defines the model and training functions
│   └── dataset.py      # Defines your dataset and its processing
├── centralized.py      # Centralized version of this example
├── preprocess.py       # A utility script to preprocess all partitions
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

> \[!NOTE\]
> This example can be run in different ways, please refer to the corresponding section for further instructions. This example was tested with `PyTorch 2.4.1` for all the different ways of running this example except when running on the Raspberry Pi, which seemed to only work with `PyTorch 1.13.1`. Please note the requirement files do not specify a version of PyTorch, therefore you need to choose one that works for you and your system.

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `whisper_example` package.

```bash
pip install -e .
```

## Centralized Training

> \[!TIP\]
> This section describes how to finetune `Whisper-tiny` for keyword spotting without making use of Federated Learning. This means that the whole training set is available at any point and therefore it is in its entirety to finetune the model each epoch. Skip to the next section if you want to jump straight into how to run `Whisper-tiny` with Flower!

Then run centralized training as follows. Please note that the first time you run the code, the `SpeechCommnads` dataset will be downloaded and pre-processed using 🤗 API (which takes a little while -- approx 40min -- and is cached in `~/.cache/huggingface/datasets/speechcommands` wiht a footprint of ~83GB). Subsequent runs shouldn't require this preprocessing.

```bash
python centralized.py --compile # don't use `--compile` flag if you are using pytorch < 2.0

# The script will save a checkpoint of the classifier head after each epoch
# These checkpoints followo the naming style: `classifier_<val_accuracy>.pt`

# You can load a checkpoint by passing it like this:
python centralized.py --checkpoint <my_checkpoint>.pt
```

Within 2 epochs you should see a validation accuracy of over 95%. On an RTX 3090Ti each epoch takes ~3min30sec. The final test set consistently reaches 97%+. Below is the log you should expect to see:

```bash
...
classifier_head_params = 781964
Initial (loss, acc): loss = 0.04124763025785586, accuracy = 0.03215788419154478
Epoch: 0
100%|████████████████████████| 84928/84928 [03:05<00:00, 456.93it/s, avg_loss=0.7269, avg_acc=0.8282]
VALIDATION ---> loss = 0.0051703976778501234, accuracy = 0.9319775596072931
Epoch: 1
100%|████████████████████████| 84928/84928 [03:07<00:00, 454.06it/s, avg_loss=0.1588, avg_acc=0.9629]
VALIDATION ---> loss = 0.003613288299632327, accuracy = 0.943097575636145
Epoch: 2
100%|████████████████████████| 84928/84928 [03:06<00:00, 454.16it/s, avg_loss=0.1208, avg_acc=0.9675]
VALIDATION ---> loss = 0.0022978041400064466, accuracy = 0.9610298537367261
Training done...
Evaluating test set. Loading best model
TEST ---> loss = 0.001703281509680464, accuracy = 0.9740286298568507
```

## Run the project

Centralized training is ok but in many settings it cannot be realised. Primarily because the training data must remain distributed (i.e. on the client side) and cannot be aggregated into a single node (e.g. your server). With [Flower](https://flower.ai/) we can easily design a federated finetuning pipeline by which clients locally train the classification head on their data, before communicating it to a central server. There, the updates sent by the clients get aggregated and re-distributed among clients for another round of FL. This process is repeated until convergence. Note that, unlike the encoder part of the Whisper model, the classification head is incredibly lightweight (just 780K parameters), adding little communication costs as a result.

There are a total of 2112 speakers in the `train` partition, which is the one we'll use in FL.

> \[!CAUTION\]
> Replace with FDS group partitioner code

```python
from datasets import load_dataset
sc_train = load_dataset("speech_commands", "v0.02", split="train", token=False)
print(sc_train)
# Dataset({
#     features: ['file', 'audio', 'label', 'is_unknown', 'speaker_id', 'utterance_id'],
#     num_rows: 84848
# })

# The training set is comprised of ~85K 1-second audio clips from 2112 individual speakers
ids = set(sc_train['speaker_id'])
print(len(ids))
# 2113  # <--- +1 since a "None" speaker is included (for clips to construct the _silence_ training examples)
```

In this example, we use the [GroupedNaturalIdPartitioner](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.GroupedNaturalIdPartitioner.html) from [Flower Datasets](https://flower.ai/docs/datasets/index.html) to partition the SpeepCommands dataset based on `speaker_id`. We will create 100 groups, each containing a varying number of speakers. Each `speaker_id` is only present in a single group. You can think of each group as an individual Federated Learning _node_ that contains several users/speakers. One way to think about this is to view each client as an office with several people working there, each interacting with the Keyword spotting system.

![Federated Whisper Finetuning pipeline](_static/federated_finetuning_flower_pipeline.png)

The resulting data partitions are not equal-sized (which is what you'd often find in practice in the real world) because not all `speaker_id` contributed the same amount of audio clips when the [Speech Commands Dataset](https://arxiv.org/abs/1804.03209) was created. If we make a bar plot showing the amount of data each client/node has this is the result.

![Amount of data per client](_static/whisper_flower_data.png)

> \[!NOTE\]
> You can make create this plot or adjust it by running the [visualize_labels](visualize_labels.ipynb) notebook. It makes use of Flower Dataset's [visualization tools](https://flower.ai/docs/datasets/tutorial-visualize-label-distribution.html).

An overview of the FL pipeline built with Flower for this example is illustrated above.

1. At the start of a round, the `ServerApp` communicates the weights of classification head to a fraction of the nodes.
2. The `ClientApp` in each node, using a frozen pre-trained Whisper encoder, trains the classification head using its own data samples.
3. Once on-site training is completed, each node sends back the (now updated) classification head to the `ServerApp`.
4. The Flower `ServerApp` aggregates (via [FedAvg](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html) -- but you can [choose any other strategy](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html), or implement your own!) the classification heads in order to obtain a new _global_ classification head. This head will be shared with nodes in the next round.

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

The run is defined in the `pyproject.toml` which: specifies the paths to `ClientApp` and `ServerApp` as well as their parameterization with configs in the `[tool.flwr.app.config]` block.

> \[!NOTE\]
> By default, it will run on CPU only. On a MacBook Pro M2, running 5 rounds of Flower FL should take ~10 min. Assuming the dataset has already been downloaded. Running on GPU is recommended.

```shell
# Run with default settings
flwr run .
```

To run your `ClientApps` on GPU, you'll need to run it in another federation (see `local-sim-gpu` in `pyprojec.toml`). To adjust the degree of parallelism, consider updating the `option.backend` settings. `ClientApp` instances consume only 800MB of VRAM, which enables you to run several in parallel in the same GPU. By default, the command below will run `5xClientApp` in parallel for each GPU available.

```
# Run with GPU
flwr run . local-sim-gpu
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
# Runs for 10 rounds and sampling 20% of the clients in each round
flwr run . --run-config "num-server-rounds=10 fraction-fit=0.2"
```

With just 5 FL rounds, the global model should be reaching ~97% validation accuracy. A test accuracy of 96% can be reached with 10 rounds of FL training using the default hyperparameters. On an RTX 3090Ti, each round takes ~40-50s depending on the amount of data the clients selected in a round have.

> \[!CAUTION\]
> Replace after group partitioner is introduced

Run on GPU with central evaluation activated and for 10 rounds.

```shell
flwr run . local-sim-gpu --run-config "central-eval=true num-server-rounds=10"
```

![Global validation accuracy FL with Whisper model](_static/whisper_flower_acc.png)

> \[!TIP\]
> If you find this federated setup not that challenging, try reducing the sizes of the groups created by the `GroupedNaturalIdPartitioner`. That will increase the number of individual clients/nodes in the federation.

### Run with the Deployment Engine

> \[!CAUTION\]
> Replace after deployment enginge consensus on instructions

Running the exact same FL pipeline as in the simulation setting can be done wihtout requiring any change to the `ClientApp` and `ServerApp` design. To achieve this, you need to ...

```bash
# python client.py --server_address='localhost' --cid=50
# This client runs on a NVIDIA RTX 3090Ti
INFO flwr 2023-11-08 14:12:50,135 | grpc.py:49 | Opened insecure gRPC connection (no certificates were passed)
DEBUG flwr 2023-11-08 14:12:50,136 | connection.py:42 | ChannelConnectivity.IDLE
DEBUG flwr 2023-11-08 14:12:50,136 | connection.py:42 | ChannelConnectivity.CONNECTING
DEBUG flwr 2023-11-08 14:12:50,140 | connection.py:42 | ChannelConnectivity.READY
99%|████████████████████████| 920/925 [00:09<00:00, 93.39it/s, avg_loss=2.4414, avg_acc=0.1837]
99%|████████████████████████| 920/925 [00:04<00:00, 216.93it/s, avg_loss=2.0191, avg_acc=0.3315]
99%|████████████████████████| 920/925 [00:04<00:00, 214.29it/s, avg_loss=1.5950, avg_acc=0.5500]
99%|████████████████████████| 920/925 [00:04<00:00, 212.70it/s, avg_loss=1.1883, avg_acc=0.7348]
99%|████████████████████████| 920/925 [00:04<00:00, 208.69it/s, avg_loss=0.8466, avg_acc=0.8228]
99%|████████████████████████| 920/925 [00:04<00:00, 206.31it/s, avg_loss=0.6353, avg_acc=0.8837]
99%|████████████████████████| 920/925 [00:03<00:00, 266.73it/s, avg_loss=0.4842, avg_acc=0.9207]
99%|████████████████████████| 920/925 [00:04<00:00, 212.13it/s, avg_loss=0.3519, avg_acc=0.9391]
99%|████████████████████████| 920/925 [00:04<00:00, 213.17it/s, avg_loss=0.3233, avg_acc=0.9359]
99%|████████████████████████| 920/925 [00:04<00:00, 205.12it/s, avg_loss=0.2646, avg_acc=0.9543]
DEBUG flwr 2023-11-08 14:20:01,065 | connection.py:139 | gRPC channel closed
INFO flwr 2023-11-08 14:20:01,065 | app.py:215 | Disconnect and shut down
```

### Federated Finetuning on Raspberry Pi

> \[!CAUTION\]
> Replace after deployment enginge consensus on instructions

Setting up the environment for the Raspberry Pi is not that different from the steps you'd follow on any other Ubuntu machine (this example assumes your Raspberry Pi -- either 5 or 4 -- runs Ubuntu server 22.04/23.10 64bits). Using the code as-is, RAM usage on the Raspberry Pi does not exceed 1.5GB. Note that unlike in the previous sections of this example, clients for Raspberry Pi work better when using PyTorch 1.13.1 (or earlier versions to PyTorch 2.0 in general).

> Please follow the steps [here](rpi_setup.md) if you are looking for a step-by-step guide on how to setup your Raspberry Pi to run this example.

In order to run this example on a Raspberry Pi, you'll need to follow the same steps as outlined above in the `non-simulated` section. First, launch the server on your development machine.

```bash
# The server will wait until at least two clients are connected
python server.py --server_addres=<YOUR_SERVER_IP>
```

Then, on each of your Raspberry Pi do the following. If you only have one RPi, you can still run the example! But you will need two clients. In addition to the one on the Raspberry Pi, you could launch a client in a separate terminal on your development machine (as shown above in the `non-simulated` section).

```bash
# use a difference `--cid` (client id) to make this device load a particular dataset partition
# we pass the `--no-compile` option since for RPi we are not using PyTorch 2.0+
python client.py --server_address=<YOUR_SERVER_IP> --cid=0 --no-compile
```

The first time you run a client on the RPi, the dataset of a client needs to be extracted from the full train set and then pre-processed. The Raspberry Pi 5 is also faster in this pre-processing stage using `.filter()` and `.map()` of 🤗 HuggingFace Dataset. `map()` used `num_proc=4`:

|                **Stage**                |                      Notes                       | **RPi 4** | **RPi 5** |
| :-------------------------------------: | :----------------------------------------------: | --------- | --------- |
| Filter through training set (~85k rows) |     doing `.filter()` in `client.client_fn`      | 1:58      | 0.37      |
| Encode 845 rows with `WhisperProcessor` | doing `.map()` passing `utils.prepare_dataset()` | 1:55      | 1:06      |

Some clients have more data than others, but on average, the RPi5 is 1.9x faster than an RPi4 when training the classification head given a frozen encoder. A client with 925 training examples needs ~20min on an RPi to complete an epoch of on-device finetuning.

```bash
# Running the 50-th client on a RPi 5 showed the following log (a RPi4 ran client 83)
python client.py --cid=50 --server_address=<YOUR_SERVER_IP> --no-compile
INFO flwr 2023-11-08 16:20:33,331 | grpc.py:49 | Opened insecure gRPC connection (no certificates were passed)
DEBUG flwr 2023-11-08 16:20:33,333 | connection.py:42 | ChannelConnectivity.IDLE
DEBUG flwr 2023-11-08 16:20:33,334 | connection.py:42 | ChannelConnectivity.CONNECTING
DEBUG flwr 2023-11-08 16:20:33,349 | connection.py:42 | ChannelConnectivity.READY
99%|████████████████████████████████████████| 920/925 [20:09<00:06,  1.31s/it, avg_loss=2.4392, avg_acc=0.1902]
99%|████████████████████████████████████████| 920/925 [20:06<00:06,  1.31s/it, avg_loss=1.9830, avg_acc=0.3533]
99%|████████████████████████████████████████| 920/925 [20:06<00:06,  1.31s/it, avg_loss=1.6069, avg_acc=0.5641]
99%|████████████████████████████████████████| 920/925 [20:07<00:06,  1.31s/it, avg_loss=1.1933, avg_acc=0.7402]
99%|████████████████████████████████████████| 920/925 [20:07<00:06,  1.31s/it, avg_loss=0.8749, avg_acc=0.8478]
99%|████████████████████████████████████████| 920/925 [20:06<00:06,  1.31s/it, avg_loss=0.5933, avg_acc=0.9109]
99%|████████████████████████████████████████| 920/925 [20:08<00:06,  1.31s/it, avg_loss=0.4882, avg_acc=0.9359]
99%|████████████████████████████████████████| 920/925 [20:01<00:06,  1.31s/it, avg_loss=0.4022, avg_acc=0.9304]
99%|████████████████████████████████████████| 920/925 [20:10<00:06,  1.32s/it, avg_loss=0.3219, avg_acc=0.9533]
99%|████████████████████████████████████████| 920/925 [20:13<00:06,  1.32s/it, avg_loss=0.2729, avg_acc=0.9641]
DEBUG flwr 2023-11-08 19:47:56,544 | connection.py:139 | gRPC channel closed
INFO flwr 2023-11-08 19:47:56,544 | app.py:215 | Disconnect and shut down
```
