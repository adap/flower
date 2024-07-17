![](_static/flower_llm.jpg)

# FlowerTune LLM Leaderboard

This repository guides you through the process of federated LLM instruction tuning with a
pre-trained [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) model across 4 domains --- general NLP, finance, medical and code.

Please follow the instructions to run and evaluate the federated LLMs.

## Create a new project

As the first step, please register a Flower account on [Flower website](https://flower.ai/login).
Assuming `flwr` package is already installed on your system (check [here](https://flower.ai/docs/framework/how-to-install-flower.html) for `flwr` installation).
We provide a single-line command to create a new project directory based on your selected challenge:

```shell
flwr new --framework=FlowerTune --username=your_flower_account
```

Then you will see a prompt to ask your project name and the choice of LLM challenges from the set of general NLP, finance, medical and code.
Type your project name and select your preferred challenge,
and then a new project directory will be generated automatically.

### Structure

After running `flwr new`, you will see a new directory generated with the following structure:

```bash
<project-name>
       ├── README.md           # <- Instructions
       ├── pyproject.toml      # <- Environment dependencies
       └── <project_name>
                  ├── app.py          # <- Flower ClientApp/ServerApp build
                  ├── client.py       # <- Flower client constructor
                  ├── server.py       # <- Sever-related functions
                  ├── models.py       # <- Model build
                  ├── dataset.py      # <- Dataset and tokenizer build
                  ├── conf/config.yaml         # <- User configuration
                  └── conf/static_config.yaml  # <- Static configuration
```

This can serve as the starting point for you to build up your own federated LLM fine-tuning methods.
Please note that any modification to the content of `conf/static_config.yaml` is strictly prohibited for those who wish to participate in the [LLM Leaderboard](https://flower.ai/benchmarks/llm-leaderboard).
Otherwise, the submission will not be considered.

## Run FlowerTune LLM challenges

With a new project directory created, running a baseline challenge can be done by:

1. Navigate inside the directory that you just created.


2. Follow the `Environments setup` section of `README.md` in the project directory to install project dependencies.


3. Run the challenge as indicated in the `Running the challenge` section in the `README.md`.

## Evaluate fine-tuned LLMs

After the LLM fine-tuning finished, evaluate the performance of your fine-tuned LLMs
following the `README.md` in [`evaluation`](https://github.com/adap/flower/tree/main/benchmarks/flowertune-llm/evaluation) directory.


If you have any questions about running FlowerTune LLM challenges or evaluation, 
please feel free to make posts at [Flower Discuss](https://discuss.flower.ai) forum, 
or join our [Slack channel](https://flower.ai/join-slack/) to ask questions in the `#flowertune-llm-leaderboard` channel.
