![](_static/flower_llm.jpg)

# FlowerTune LLM Leaderboard

This repository guides you through the process of federated LLM instruction tuning with a 
pre-trained [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) models across different domains.
Please follow the instructions to run and evaluate the LLM baselines.


## Structure
Each LLM challenge within this directory is completely self-contained, with its source code neatly organized in its respective subdirectory.
The challenge subdirectories contain the following structure:

```bash
flowertune-llm/<challenge-name>
                      ├── README.md
                      ├── pyproject.toml   # <- Environment dependencies
                      └── <challenge_name>
                                ├── app.py          # <- Flower ClientApp/ServerApp build
                                ├── client.py       # <- Flower client constructor
                                ├── server.py       # <- Sever-related functions
                                ├── models.py       # <- Model build
                                ├── dataset.py      # <- Dataset and tokenizer build
                                ├── conf/config.yaml         # <- User configuration
                                └── conf/static_config.yaml  # <- Static configuration   
```

The `evaluation` subdirectory contains the code and instruction to evaluate trained LLMs for different domains.


## Create a new project
Assuming `flwr` package is already installed on your system (check [here](https://flower.ai/docs/framework/how-to-install-flower.html) for `flwr` installation).
We provide a single-line command to create a new project directory based on your selected challenge:

```shell
flwr new --framework=flwrtune --username=your_flower_account
```

Then you will see a prompt to ask your project name and the choice of LLM challenges. 
Type your project name and select your preferred challenge, 
and then a new project directory with same structure as above will be generated automatically.

## Running FlowerTune LLM challenges

With a new project directory created, running a challenge task can be done by:

1. Navigate inside the directory that you just created.

2. Follow the `Environments setup` section in the `README.md` to install project dependencies.

4. Run the challenge task as indicated in the `Running the task` section in the `README.md`.

5. Test your trained LLM following the `README.md` in the `evaluation/challenge-name` directory.
