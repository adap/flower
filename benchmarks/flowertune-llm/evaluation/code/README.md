## Evaluation for Code challenge

We leverage the code generation evaluation metrics provided by [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main) to evaluate our fine-tuned LLMs.
Three datasets have been selected for this evaluation: [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) (Python), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) (Python), and [MultiPL-E](https://github.com/nuprl/MultiPL-E) (JavaScript, C++). 


### Step 0. Set up Environment

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/code ./flowertune-eval-code && rm -rf flower && cd flowertune-eval-code
```

Then, install dependencies with:

```shell
# From a new python environment, run:
pip install -e .

# Log in HuggingFace account
huggingface-cli login
```

After that, install `Node.js` for the evaluation of JavaScript, C++:

```shell
# Install nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Download and install Node.js (you may need to restart the terminal)
nvm install 20
```


### Step 1. Generate model answers & calculate pass@1 score

```bash
python main.py \
--peft_model=/path/to/fine-tuned-peft-model-dir/  # e.g., ./peft_1
--max_length_generation=1024 # change to 2048 when running mbpp
--batch_size=4 
--save_generations 
--save_references
--tasks=humaneval # chosen from [mbpp, humaneval, multiple-js, multiple-cpp]
--metric_output_path=./evaluation_results_humaneval.json # change dataset name based on your choice
```
The model answers and pass@1 scores will be saved to `generations_{dataset_name}.json` and `evaluation_results_{dataset_name}.json`, respectively.

> [!NOTE]
> Please ensure that you provide all **four pass@1 scores** for the evaluation datasets when submitting to the LLM Leaderboard (see the [`Make Submission`](https://github.com/adap/flower/tree/main/benchmarks/flowertune-llm/evaluation#make-submission-on-flowertune-llm-leaderboard) section).
