from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM


def formatting_prompts_func(example):
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    # From: https://huggingface.co/docs/trl/en/sft_trainer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func
