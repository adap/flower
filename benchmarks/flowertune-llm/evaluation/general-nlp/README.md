## Evaluation for General NLP challenge

We leverage MT-bench metric provided by [FastChat](https://github.com/lm-sys/FastChat) to evaluate our trained LLMs.
MT-bench represents a comprehensive suite of multi-turn, open-ended questions designed to evaluate chat assistants.
Strong LLMs, such as GPT-4, serve as judges to assess the quality of responses provided by the chat assistants under examination.

### Step 0. Set up Environment

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/general-nlp . && rm -rf flower && cd general-nlp
```

Create a new Python environment (we recommend Python 3.10), activate it, then install dependencies with:

```shell
# From a new python environment, run:
pip install -e .

# Log in HuggingFace account
huggingface-cli login
```


### Step 1. Generate model answers to MT-bench questions

```bash
python gen_model_answer.py --peft-path=/path/to/pre-trained-model-dir/ # e.g., ./peft_1
```
The answers will be saved to `data/mt_bench/model_answer/[base_model_name].jsonl` in default.


### Step 2. Generate judgments using GPT-4 
```bash
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgement.py --model-list Mistral-7B-v0.3
```

You can specify the base model name via `--model-list`.
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl` in default.

### Step 3. Show MT-bench scores

```bash
python show_result.py --model-list Mistral-7B-v0.3
```
GPT-4 will give a score on a scale of 10 to the first-turn (MT-1) and second-turn (MT-2) of the conversations, along with an average value as the third score.

> [!NOTE]
> Please ensure that you provide all **three scores** when submitting to the LLM Leaderboard.
