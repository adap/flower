![](_static/flower_llm.jpg)

# FlowerTune LLM Leaderboard

This repository guides you through the process of federated LLM instruction tuning with pre-trained [LLama2](https://huggingface.co/mistralai/Mistral-7B-v0.3) models across various tasks.
Please follow the instructions to run baseline tasks and create your own projects.


## Structure
Each LLM task within this directory is completely self-contained, with its source code neatly organized in its respective subdirectory.
The task subdirectories contain the following structure:

```bash
llm-flowertune/<task-name>/
                 ├── README.md
                 ├── main.py         # <- Flower ClientApp/ServerApp build
                 ├── client.py       # <- Flower client constructor
                 ├── server.py       # <- Sever-related functions
                 ├── model.py        # <- Model build
                 ├── dataset.py      # <- Dataset and tokenizer build
                 ├── requirements.txt         # <- Environment dependencies
                 ├── conf/config.yaml         # <- User configuration
                 ├── static_conf/config.yaml  # <- Static configuration
                 └── <evaluation>             # <- Evaluation metrics
```
> More tasks will come soon.

## Running FlowerTune-LLM tasks

Each LLM-FlowerTune task is self-contained in its own directory.
Running a task can be done by:

1. Follow the `Environments setup` section in the `README.md` of each task subdirectory to create a new project directory.


2. Navigate inside the directory of the task you'd like to run.


3. Install project dependencies are defined in `requirements.txt`. We recommend [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to install those dependencies and manage your virtual environment, but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

    ```shell
    conda create -n llm-flowertune python==3.10.0
    conda activate llm-flowertune
    pip install -r requirements.txt
    ```
4. Run the task as indicated in the `Running the task` section in the `README.md`.


5. Test your trained model following the `README.md` in the `evaluation` subdirectory.


## Create a new project (replace by `flwr new`)

We provide a single-line command to create a new project directory based on your selected tasks:

```shell
# Replace `task-name` based on your choice
git clone --depth=1 https://github.com/adap/flower.git && mv flower/benchmark/llm-flowertune/task-name ./task-name && rm -rf flower && cd task-name
```

## Contribute to new tasks (TBC)

If you have new LLM fine-tuning tasks on different domains, we really appreciate your contribution to the LLM task board !!

The steps to follow are:

1. Fork the Flower repo and clone it into your machine.


2. Create your own project based on the given template.


3. Contribute your code following the steps detailed in README.md.


4. Once your code is ready, you just need to create a Pull Request (PR). Then, the process to merge your code into the FlowerTune LLM leaderboard will begin!

Further resources:
* [GitHub docs: About forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks)
* [GitHub docs: Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
* [GitHub docs: Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)






