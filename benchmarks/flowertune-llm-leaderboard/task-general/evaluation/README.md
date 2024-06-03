## Evaluation using MT-Bench

We leverage MT-bench metric provided by [FastChat](https://github.com/lm-sys/FastChat) to evaluate our trained LLMs.
MT-bench represents a comprehensive suite of multi-turn, open-ended questions designed to evaluate chat assistants.
Strong LLMs, such as GPT-4, serve as judges to assess the quality of responses provided by the chat assistants under examination.

### Step 1. Generate model answers to MT-bench questions

```bash
python gen_model_answer.py --peft-path=/path/to/pre-trained-model-dir/ # e.g., ./peft_1
```
The answers will be saved to `data/mt_bench/model_answer/[base_model_name].jsonl` in default.


### Step 2. Generate judgments using GPT-4 
```bash
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgement.py --model_list open_llama_7b_v2
```

You can specify the base model name via `--model_list`.
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl` in default.

### Step 3. Show MT-bench scores

```bash
python show_result.py --model_list open_llama_7b_v2
```
GPT-4 will give a score on a scale of 10 to the first-turn (MT-1) and second-turn (MT-2) of the conversations, along with an average value as the third score.
