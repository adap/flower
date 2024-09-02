# Evaluation for General NLP challenge

We leverage MT-Bench metric provided by [FastChat](https://github.com/lm-sys/FastChat) to evaluate fine-tuned LLMs.
[MT-Bench](https://arxiv.org/abs/2306.05685) represents a comprehensive suite of multi-turn, open-ended questions designed to evaluate chat assistants.
Strong LLMs, such as GPT-4, serve as judges to assess the quality of responses provided by the chat assistants under examination.

## Environment Setup

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/general-nlp ./flowertune-eval-general-nlp && rm -rf flower && cd flowertune-eval-general-nlp
```

Create a new Python environment (we recommend Python 3.10), activate it, then install dependencies with:

```shell
# From a new python environment, run:
pip install -r requirements.txt

# Log in HuggingFace account
huggingface-cli login
```

Download data from [FastChat](https://github.com/lm-sys/FastChat):

```shell
git clone --depth=1 https://github.com/lm-sys/FastChat.git && cd FastChat && git checkout d561f87b24de197e25e3ddf7e09af93ced8dfe36 && mv fastchat/llm_judge/data ../data && cd .. && rm -rf FastChat
```


## Generate model answers from MT-bench questions

```bash
python gen_model_answer.py --peft-path=/path/to/fine-tuned-peft-model-dir/ # e.g., ./peft_1
```
The answers will be saved to `data/mt_bench/model_answer/[base_model_name].jsonl` in default.


## Generate judgments using GPT-4

Please follow these [instructions](https://platform.openai.com/docs/quickstart/developer-quickstart) to create a OpenAI API key.
The estimated costs of running this evaluation is approximately USD10.

> [!NOTE]
> If you changed the base model of your LLM project specify it to the command below via `--model-list`.

```bash
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgement.py --model-list Mistral-7B-v0.3
```

The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl` in default.


## Show MT-bench scores

```bash
python show_result.py --model-list Mistral-7B-v0.3
```
GPT-4 will give a score on a scale of 10 to the first-turn (MT-1) and second-turn (MT-2) of the conversations, along with an average value as the third score.

> [!NOTE]
> Please ensure that you provide all **three scores** when submitting to the LLM Leaderboard (see the [`Make Submission`](https://github.com/adap/flower/tree/main/benchmarks/flowertune-llm/evaluation#make-submission-on-flowertune-llm-leaderboard) section).

