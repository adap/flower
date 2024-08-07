# Evaluation for Medical challenge

We leverage the medical question answering (QA) metric provided by [Meditron](https://github.com/epfLLM/meditron/tree/main/evaluation) to evaluate our fined-tuned LLMs.
Three datasets have been selected for this evaluation: [PubMedQA](https://huggingface.co/datasets/bigbio/pubmed_qa), [MedMCQA](https://huggingface.co/datasets/medmcqa), and [MedQA](https://huggingface.co/datasets/bigbio/med_qa). 


## Environment Setup

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/medical ./flowertune-eval-medical && rm -rf flower && cd flowertune-eval-medical
```

Create a new Python environment (we recommend Python 3.10), activate it, then install dependencies with:

```shell
# From a new python environment, run:
pip install -e .

# Log in HuggingFace account
huggingface-cli login
```

## Generate model answers to medical questions

```bash
python inference.py \
--peft-path=/path/to/fine-tuned-peft-model-dir/  # e.g., ./peft_1
--dataset-name=pubmedqa  # chosen from [pubmedqa, medmcqa, medqa]
--run-name="fl-pubmedqa"  # an identifier for this run (up to you to choose) 
```
The answers will be saved to `benchmarks/generations/[dataset_name]-[run_name].jsonl` in default.


## Calculate accuracy

```bash
python evaluate.py \
--dataset-name=pubmedqa  # chosen from [pubmedqa, medmcqa, medqa]
--run-name=fl  # run_name used in Step 1
```
The accuracy value will be printed on the screen.

> [!NOTE]
> Please ensure that you provide all **three accuracy values** for three evaluation datasets when submitting to the LLM Leaderboard (see the [`Make Submission`](https://github.com/adap/flower/tree/main/benchmarks/flowertune-llm/evaluation#make-submission-on-flowertune-llm-leaderboard) section).
