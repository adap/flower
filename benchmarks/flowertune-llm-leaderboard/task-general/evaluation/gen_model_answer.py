# This python file is adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py
# The model/tokenizer loader modified to adapt our trained models.

import json
import argparse
import os
import random
from tqdm import tqdm
import torch
import time

from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastchat.conversation import get_conv_template
from fastchat.llm_judge.common import load_questions, temperature_config


parser = argparse.ArgumentParser()
parser.add_argument("--peft-path", type=str, default=None)
parser.add_argument("--template", type=str, default="vicuna_v1.1")
parser.add_argument("--max-new-token", type=int, default=1024)
parser.add_argument("--num-choices", type=int, default=1)
args = parser.parse_args()

# Load model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    args.peft_path, torch_dtype=torch.float16
).to("cuda")
base_model = model.peft_config["default"].base_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(base_model)

model_name = base_model.split('/')[1]
question_file = f"./data/mt_bench/question.jsonl"
answer_file = f"./data/mt_bench/model_answer/{model_name}.jsonl"

# Load questions
questions = load_questions(question_file, None, None)
# Random shuffle the questions to balance the loading
random.shuffle(questions)

# Generate answers
for question in tqdm(questions):
    # Set temperature value
    temperature = temperature_config[question["category"]] if question["category"] in temperature_config else 0.7
    choices = []
    for i in range(args.num_choices):
        torch.manual_seed(i)
        conv = get_conv_template(args.template)
        turns = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            do_sample = False if temperature < 1e-4 else True

            # Some models may error out when generating long outputs
            try:
                output_ids = model.generate(
                    input_ids=torch.as_tensor(input_ids).cuda(),
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=args.max_new_token,
                    pad_token_id=tokenizer.eos_token_id,
                )
                output_ids = output_ids[0] if model.config.is_encoder_decoder else output_ids[0][len(input_ids[0]) :]

                # Be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            conv.update_last_message(output)
            turns.append(output)
        choices.append({"index": i, "turns": turns})

    # Dump answers
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(os.path.expanduser(answer_file), "a") as fout:
        ans_json = {
            "question_id": question["question_id"],
            "model_id": model_name,
            "choices": choices,
            "tstamp": time.time(),
        }
        fout.write(json.dumps(ans_json) + "\n")
