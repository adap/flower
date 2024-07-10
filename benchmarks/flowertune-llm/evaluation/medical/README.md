## Evaluation for Medical challenge

We leverage the medical question answering (QA) metric provided by [Meditron](https://github.com/epfLLM/meditron/tree/main/evaluation) to evaluate our trained LLMs.
Three datasets have been selected for this evaluation: [PubMedQA](https://huggingface.co/datasets/bigbio/pubmed_qa), [MedMCQA](https://huggingface.co/datasets/medmcqa), and [MedQA](https://huggingface.co/datasets/bigbio/med_qa). 


### Step 0. Set up Environment

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/medical . && rm -rf flower && cd medical
```

Then, install dependencies with:

```shell
# From a new python environment, run:
pip install -e .

# Log in HuggingFace account
huggingface-cli login
```

### Step 1. Generate model answers to medical questions

```bash
python inference.py \
--peft-path=/path/to/pre-trained-model-dir/  # e.g., ./peft_1
--dataset-name=pubmedqa  # chosen from [pubmedqa, medmcqa, medqa]
--run-name=fl  # arbitrary name for this run 
```
The answers will be saved to `benchmarks/generations/[dataset_name]-[run_name].jsonl` in default.


### Step 2. Calculate accuracy

```bash
python evaluate.py \
--dataset-name=pubmedqa  # chosen from [pubmedqa, medmcqa, medqa]
--run-name=fl  # run_name used in Step 1
```
The accuracy value will be printed on the screen.

> [!NOTE]
> Please ensure that you provide all **three accuracy values** for three evaluation datasets when submitting to the LLM Leaderboard.
