# Whisper + Flower

This example demonstrates how to, from a pre-trained Whisper model, finetune it for the downstream task of keyword spotting. We'll be using the [Google Speech Commands](https://huggingface.co/datasets/speech_commands) dataset for keyword spotting. We'll take the encoder part of the [Whisper-tiny](https://huggingface.co/openai/whisper-tiny) model, freeze its parameters, and learn a lightweight classification head.

This example can be run in three modes: centralised training, federated (simulation), and federated on-device with Raspberry Pi. Note that most of the code is re-used across both centralised and federated modes.

## Centralised training

This section describes how to finetune `Whisper-tiny` for keyword spotting without making use of Federated Learning. This means that the whole training set will be used at once to finetune the model.

On your favorite Python environment manager, install a recent version of PyTorch (PyTorch 2.0+ is recommended for faster training times). Then install the rest of the requirements. For instance:

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Then run centralised training as follows:

```bash
python centralised.py --compile # don't use `--compile` flag if you are using pytorch < 2.0

# The script will save a checkpoint of the classifier head after each epoch
# These checkpoints followo the naming style: `classifier_<val_accuracy>.pt`

# You can load a checkpoint by passing it like this:
python centralised.py --checkpoint <my_checkpoint>.pt
```

Within a 2 or 3 epochs you should see a test accuracy of ~97%. On an RTX 3090Ti each epoch takes ~3min30sec.

## Federated Learning

Centralised training is ok but in many settings it cannot be realised. Primarily due to the fact that the training remains distributed (i.e. on the client side) and cannot be aggregated into a single node (e.g. server). With Flower we can easily design a federated finetuning pipeline by which clients locally train the classification head on their own data, before communicating the updated part of the model to a central server. There, the updates sent by the clients get aggregated and re-distributed among clients for another round of FL.

Flower supports two ways of doing Federated Learning: simulated and non-simulated FL. The former, managed by the [`VirtualClientEngine`](https://flower.dev/docs/framework/how-to-run-simulations.html), allows you to run FL . The latter is better suited for settings where clients are unique devices (e.g. a server, a smart device, etc). This example shows you how to use both.

### Federated Finetuning (Simulation)

The setup instructions for simulations are the same as those described for the centralised setting above. Then, you can launch your simulation as follows:

```bash
# By default it will run 2 clients in parallel on a single GPU (which should be fine if your GPU has at least 16GB )
# If that's too much, consider reduing either the batch size or raise `num_gpus` passed to `start_simulation`
python sim.py --num-rounds 5
```

With just 5 FL rounds, the global model should be reaching ~95% test accuracy. On an RTX 3090Ti, running 5 rounds took ~5minutues.

### Federated Finetuning (on-device / RaspberryPi)

Setting up the environment for the RaspberryPi is not that different from the steps you'd follow on any other Ubuntu machine (yes, this example assumes your Raspberry Pi -- either 5 or 4 -- runs Ubuntu server 22.04/23.10 64bits). If you have a RaspberryPi already up and running you can probably skip most of the steps below.

#### Setting up your RaspberryPi for Python development

We are going to use [`pyenv`](https://github.com/pyenv/pyenv) to manage different Python versions and to create an environment. First, we need to install some system dependencies

```bash
sudo apt update
# the last package is needed for whisper
sudo apt install build-essential zlib1g-dev libssl-dev libsqlite3-dev libreadline-dev libbz2-dev libffi-dev liblzma-dev libsndfile1
```

Create Python environment with `pyenv`:

```bash

# Ensure you have installed pyenv, else do the below:
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Install python 3.9+
pyenv install 3.9.17

# Install pyenv virtual env plugin
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
# Restart your shell
exec "$SHELL"

# Create the environment
pyenv virtualenv 3.9.17 flower-whisperer
```

With our environmnet ready, let's install the dependencies. Please note that at the time of writing, PyTorch 2.0+ won't work properly on `aarm64`. Because of this, we'll be using an earlier version of this package.

```bash
# activate your environment
pyenv activate flower-whisperer

# install pytorch (RPi aren't ready for PyTorch 2.0+ apparently...)
pip install torch==1.13.1
# install rest of requirerments
pip install -r requirements.txt
```

#### Running the experiment

Since you won't be running in simulation mode, you need to run first the Server (which will orchestrate the entire experiment) and then the Clients (one per device).

First launch the server on your development machine.

```bash
# The server will wait until at least two clients are connected
python server.py --server_addres=<YOUR_SERVER_IP>
```

Then, on each of your RaspberryPi do the following. If you only have one RPi, you can still run the example! You can launch a client in a separate terminal on your development machine.

```bash
# use a difference `--cid` (client id) to make this device load a particular dataset partition
python client.py --server_address=<YOUR_SERVER_IP> --cid=0
```

Some clients have more data than others, but on average the RPi5 is 1.9x faster than a RPi4. A client with 850 training examples needs ~19min on a RPi to complete an epoch of on-device finetuning.
