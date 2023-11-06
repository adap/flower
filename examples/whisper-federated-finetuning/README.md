# On-device Federated Downstreaming for Speech Classification

This example demonstrates how to, from a pre-trained [Whisper](https://openai.com/research/whisper) model, finetune it for the downstream task of keyword spotting. We'll be implementing a federated downstream finetuning pipeline using Flower involving a total of 100 clients. As for the downstream dataset, we'll be using the [Google Speech Commands](https://huggingface.co/datasets/speech_commands) dataset for keyword spotting. We'll take the encoder part of the [Whisper-tiny](https://huggingface.co/openai/whisper-tiny) model, freeze its parameters, and learn a lightweight classification (\<800K parameters !!) head to correctly classify a spoken word.

This example can be run in three modes:

- **Centralized training**: the standard way of training ML models, where all the data is available to the node doing the finetuning.
- **Federated Learning**: the better way of doing ML, where a model is finetuned collaboratively by nodes (i.e. clients), each using their own data. These clients can run:
  - in _simulation_ mode: a client is an ephemeral Python process with a portion of the system resources assigned to it.
  - in _on-device_ mode: clients are detached entities and each can run on a different device.

## Running the example

Start by cloning the code example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/whisper-federated-finetuning . && rm -rf flower && cd whisper-federated-finetuning
```

This will create a new directory called `whisper-federated-finetuning` containing the following files:

```
-- README.md         <- Your're reading this right now
-- rpi_setup.md      <- A guide that illustrates how to setup your RPi from scratch
-- sim.py            <- Runs the example with Flower simulation
-- server.py         <- Defines the server-side logic for the on-device setting
-- client.py         <- Defines the client-side logic for the on-device setting
-- utils.py          <- auxiliary functions for this example
-- centralised.py    <- Runs the example in centralized mode
-- pyproject.toml    <- Example dependencies (if you use Poetry)
-- requirements.txt  <- Example dependencies
```

This example can be run in different ways, please refer to the corresponding section for further instructions. This example was tested with `PyTorch 2.1.0` for all the different ways of running this example except when running on the Raspberry Pi, which seemed to only work with `PyTorch 1.13.1`. Please not the requirement files do not specify a version of PyTorch, therefore you need to choose one that works for you and your system.

## Centralized Training

This section describes how to finetune `Whisper-tiny` for keyword spotting without making use of Federated Learning. This means that the whole training set is available at any point and therefore it is in its entirety to finetune the model each epoch.

On your favorite Python environment manager, install a recent version of [PyTorch](https://pytorch.org/get-started/locally/) (PyTorch 2.0+ is recommended for faster training times). Then install the rest of the requirements. For instance:

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Then run centralized training as follows. Please note that the first time you run the code, the `SpeechCommnads` dataset will be downloaded and pre-processed using 🤗 API (which takes a little while -- approx 30min). Subsequent runs shouldn't require this preprocessing.

```bash
python centralised.py --compile # don't use `--compile` flag if you are using pytorch < 2.0

# The script will save a checkpoint of the classifier head after each epoch
# These checkpoints followo the naming style: `classifier_<val_accuracy>.pt`

# You can load a checkpoint by passing it like this:
python centralised.py --checkpoint <my_checkpoint>.pt
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

> You made it work better ? Let us know how you did it by opening an GitHub issue or a PR and we'll gladly incorporate your suggestions!

## Federated Learning

Centralized training is ok but in many settings it cannot be realised. Primarily because the training data must remain distributed (i.e. on the client side) and cannot be aggregated into a single node (e.g. your server). With Flower we can easily design a federated finetuning pipeline by which clients locally train the classification head on their data, before communicating it to a central server. There, the updates sent by the clients get aggregated and re-distributed among clients for another round of FL. This process is repeated until convergence. Note that, unlike the encoder part of the Whisper model, the classification head is incredibly lightweight (just 780K parameters), adding little communication costs as a result.

Flower supports two ways of doing Federated Learning: simulated and non-simulated FL. The former, managed by the [`VirtualClientEngine`](https://flower.dev/docs/framework/how-to-run-simulations.html), allows you to run large-scale workloads in a system-aware manner, that scales with the resources available on your system (whether it is a laptop, a desktop with a single GPU, or a cluster of GPU servers). The latter is better suited for settings where clients are unique devices (e.g. a server, a smart device, etc). This example shows you how to use both.

### Preparing the dataset

If you have run the centralized version of this example first, you probably realized that it takes some time to get a fully pre-processed SpeechCommands dataset using the 🤗 HuggingFace API. This pre-processing is ideal so nothing slowdowns our training once we launch the experiment. For the federated part of this example, we also need to pre-process the data however in a different way since first the training set needs to be split into N different buckets, one for each client.

To launch a Flower client we write a `client_fn` callable that will: (1) Load the dataset of the client; then (2) return the Client object itself. In `client.py` we have included a few lines of code that preprocess the training partition of a given client and saves it to disk (so this doesn't have to be repeated each time you run the experiment). You can run the experiment right away and the data will be pre-processed on-demand (i.e. when the `i`-th client is spawned for the first time), or you can pre-process all client partitions first. In order to do so, please run:

```bash
# will write to disk all pre-processed data partitions
# by default these will go to a new directory named `client_datasets`
# Similarly to the centralised setting, this preprocessing will take a while (30mins approx)
python sim.py --preprocess
```

The resulting data partitions are not equal-sized (which is what you'd often find in practice in the real world). If we make a bar plot showing the amount of data each client has this is the result.

![Amount of data per client](_static/whisper_flower_data.png)

### Federated Downstreaming (Simulation)

The setup instructions for simulations are the same as those described for the centralized setting above. Then, you can launch your simulation as shown below.

```bash
# By default it will run 2 clients in parallel on a single GPU (which should be fine if your GPU has at least 16GB )
# If that's too much, consider reduing either the batch size or raise `num_gpus` passed to `start_simulation`
python sim.py --num-rounds 10 # append --num_gpus=0 if you don't have GPUs on your system
```

![Global validation accuracy FL with Whisper model](_static/whisper_flower_acc.png)

With just 5 FL rounds, the global model should be reaching ~95% validation accuracy. A test accuracy of 97% can be reached with 10 rounds of FL training using the default hyperparameters. On an RTX 3090Ti, each round takes ~20-30s depending on the amount of data the clients selected in a round have.

### Federated Downstreaming (non-simulated)

Running the exact same FL pipeline as in the simulation setting can be done without using Flower's simulation engine. To achieve this, you need to launch first a server and then two or more clients. You can do this on your development machine assuming you have set up your environment already.

First, launch the server, which will orchestrate the FL process:

```bash
# The server will wait until at least two clients are connected
python server.py --server_addres=<YOUR_SERVER_IP>
```

Then on different (new) terminals run:

```bash
# use a difference `--cid` (client id) to make this device load a particular dataset partition
python client.py --server_address=<YOUR_SERVER_IP> --cid=0

# and on a new terminal
python client.py --server_address=<YOUR_SERVER_IP> --cid=1
```

Once the second client connects to the server, the FL process will begin.

### Federated Downstreaming on Raspberry Pi

> Please follow the steps [here](rpi_setup.md) if you are looking for a step-by-step guide on how to setup your Raspberry Pi to run this example.

Setting up the environment for the Raspberry Pi is not that different from the steps you'd follow on any other Ubuntu machine (this example assumes your Raspberry Pi -- either 5 or 4 -- runs Ubuntu server 22.04/23.10 64bits).

In order to run this example on a Raspberry Pi, you'll need to follow the same steps as outlined above in the `non-simulated` section. First, launch the server on your development machine.

```bash
# The server will wait until at least two clients are connected
python server.py --server_addres=<YOUR_SERVER_IP>
```

Then, on each of your Raspberry Pi do the following. If you only have one RPi, you can still run the example! But you will need two clients. In addition to the one on the Raspberry Pi, you could launch a client in a separate terminal on your development machine.

```bash
# use a difference `--cid` (client id) to make this device load a particular dataset partition
python client.py --server_address=<YOUR_SERVER_IP> --cid=0
```

Some clients have more data than others, but on average, the RPi5 is 1.9x faster than an RPi4. A client with 850 training examples needs ~19min on an RPi to complete an epoch of on-device finetuning.
