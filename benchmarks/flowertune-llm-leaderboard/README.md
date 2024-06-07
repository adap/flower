![](_static/flower_llm.jpg)

# FlowerTune LLM Leaderboard

This repository guides you through the process of federated LLM instruction tuning with a pre-trained [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) models across various tasks.
Please follow the instructions to run baseline tasks and create your own projects.


## Structure
Each LLM task within this directory is completely self-contained, with its source code neatly organized in its respective subdirectory.
The task subdirectories contain the following structure:

```bash
flowertune-llm-leaderboard/<task-name>/
                 ├── README.md
                 ├── main.py         # <- Flower ClientApp/ServerApp build
                 ├── client.py       # <- Flower client constructor
                 ├── server.py       # <- Sever-related functions
                 ├── model.py        # <- Model build
                 ├── dataset.py      # <- Dataset and tokenizer build
                 ├── pyproject.toml           # <- Environment dependencies
                 ├── conf/config.yaml         # <- User configuration
                 ├── static_conf/config.yaml  # <- Static configuration
                 └── <evaluation>             # <- Evaluation metrics
```

## Running FlowerTune-LLM tasks

Each LLM-FlowerTune task is self-contained in its own directory.
Running a task can be done by:

1. Follow the `Environments setup` section in the `README.md` of each task subdirectory to create a new project directory.


2. Navigate inside the directory of the task you'd like to run.


3. Install project dependencies are defined in `pyproject.toml`. 


4. Run the task as indicated in the `Running the task` section in the `README.md`.


5. Test your trained model following the `README.md` in the `evaluation` subdirectory.
