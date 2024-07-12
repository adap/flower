## Evaluation for Finance challenge

We leverage the sentiment classification pipeline on finance-related text provided by [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT/tree/master) to evaluate our trained LLMs.
Three datasets have been selected for this evaluation: [FPB](https://huggingface.co/datasets/takala/financial_phrasebank), [FIQA](https://huggingface.co/datasets/pauri32/fiqa-2018), and [TFNS](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment). 


### Step 0. Set up Environment

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/finance ./flowertune-eval-finance && rm -rf flower && cd flowertune-eval-finance
```

Then, install dependencies with:

```shell
# From a new python environment, run:
pip install -e .

# Log in HuggingFace account
huggingface-cli login
```

### Step 1. Generate model decision & calculate accuracy

```bash
python eval.py \
--peft-path=/path/to/pre-trained-model-dir/ # e.g., ./peft_1
--run-name=fl  # arbitrary name for this run  
--batch-size=32 
--quantization=4 
--datasets=fpb,fiqa,tfns
```
The model answers and accuracy values will be saved to `benchmarks/generation_{dataset_name}_{run_name}.jsonl` and `benchmarks/acc_{dataset_name}_{run_name}.txt`, respectively.

> [!NOTE]
> Please ensure that you provide all **three accuracy values** for three evaluation datasets when submitting to the LLM Leaderboard (see the [`Make Submission`](https://github.com/adap/flower/tree/main/benchmarks/flowertune-llm/evaluation#make-submission-on-flowertune-llm-leaderboard) section).
