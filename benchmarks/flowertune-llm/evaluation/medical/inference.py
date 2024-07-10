# This python file is adapted from https://github.com/epfLLM/meditron/blob/main/evaluation/inference.py

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmarks import benchmark_factory, load_instruction

# Fixed seed
torch.manual_seed(2024)

INSTRUCTIONS = {
    "medmcqa": {"task": "mcq", "partition": "validation", "instructions": "medmcqa"},
    "pubmedqa": {"task": "mcq", "partition": "test", "instructions": "pubmedqa"},
    "medqa": {"task": "mcq", "partition": "test", "instructions": "medqa"},
}


def tokenizer_param(tokenizer, target, task_type="mcq"):
    """Determines the maximum number of tokens to generate for a given prompt and
    target. Also determines the stop sequence to use for generation.

    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for
        inference
    :param target: str, the target to generate
    :param task_type: str, the type of answer to generate (mcq or open)
    """
    max_new_tokens = len(tokenizer(target, add_special_tokens=True)["input_ids"])
    stop_seq = ["###"]
    if tokenizer.eos_token is not None:
        stop_seq.append(tokenizer.eos_token)
    if tokenizer.pad_token is not None:
        stop_seq.append(tokenizer.pad_token)

    if task_type == "mcq":
        max_new_tokens = len(
            tokenizer(target[0], add_special_tokens=False)["input_ids"]
        )

    return max_new_tokens, stop_seq


def benchmark_infer(model, tokenizer, data, device):
    """Runs inference on a benchmark and stores generations in a pd.DataFrame.

    :param model: HuggingFace model, the model to use for inference
    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param data: HuggingFace Dataset, the dataset to run inference on
    :param device: str, the device to run inference on, cpu or cuda

    return: pd.DataFrame, a DataFrame containing the scores for each answer
    """
    columns_to_save = ["prompt", "gold"]
    if "subset" in data.features:
        columns_to_save.append("subset")
    predictions = pd.DataFrame(data, columns=data.features)[columns_to_save]
    predictions = predictions.assign(output="Null")
    temperature = 1.0

    inference_data = json.loads(predictions.to_json(orient="records"))
    data_loader = DataLoader(inference_data, batch_size=16, shuffle=False)

    batch_counter = 0
    for batch in tqdm(data_loader, total=len(data_loader), position=0, leave=True):
        prompts = [
            f"<|im_start|>question\n{prompt}<|im_end|>\n<|im_start|>answer\n"
            for prompt in batch["prompt"]
        ]
        if batch_counter == 0:
            print(prompts[0])

        max_new_tokens, stop_seq = tokenizer_param(tokenizer, batch["gold"])

        outputs = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            output_ids = model.generate(
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=1.0,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            output_ids = output_ids[0][len(input_ids[0]) :]
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            outputs.append(output)

        for prompt, out in zip(batch["prompt"], outputs):
            predictions.loc[predictions["prompt"] == prompt, "output"] = out
        batch_counter += 1

    return predictions


def benchmark_preparation(data_obj, dataset_name, partition):
    """Runs the benchmark preparation pipeline on a given dataset.

    :param data_obj: benchmark.Benchmark, the benchmark to run the preparation pipeline
        on
    :param dataset_name: str, the dataset used for benchmark preparation
    :param partition: str, the partition to run the preparation pipeline on
    """
    data_obj.load_data(partition=partition)
    data_obj.preprocessing(partition=partition)
    prompt_name = INSTRUCTIONS[dataset_name]["instructions"]

    instruction = load_instruction(prompt_name)
    print(
        f'Instruction used for evaluation: \n\t{instruction["system"]}\n\t{instruction["user"]}\n'
    )

    data_obj.add_instruction(instruction=instruction, partition=partition)
    return prompt_name


def main(
    base_model_name_path="mistralai/Mistral-7B-v0.3",
    peft_path=None,
    dataset_name="pubmedqa",
    run_name="fl",
    quantization=4,
):
    # Load model and tokenizer
    if quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        torch_dtype = torch.float32
    elif quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = torch.float16
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {quantization}/"
        )

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        quantization_config = None
        print(
            f"{quantization} bit quantization is chosen, but a GPU is not found, running on CPU without quantization."
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_path,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )
    if peft_path is not None:
        model = PeftModel.from_pretrained(model, peft_path, torch_dtype=torch_dtype).to(
            device
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_path)

    # Prepare data
    partition = INSTRUCTIONS[dataset_name]["partition"]
    data_obj = benchmark_factory(dataset_name)
    benchmark_preparation(data_obj, dataset_name, partition)

    # Prediction
    predictions = benchmark_infer(model, tokenizer, data_obj.test_data, device)

    # Save results
    data_obj.add_generations(data=predictions)
    data_obj.save_generations(dataset_name=dataset_name, run_name=run_name)
    print(f"{len(predictions)} generations store.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-name-path",
        type=str,
        default="mistralai/Mistral-7B-v0.3",
        help="Base model name used for fine-tuning",
    )
    parser.add_argument("--peft-path", type=str, help="Path for saved PEFT model")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="pubmedqa",
        help="The dataset to infer on: [pubmedqa, medqa, medmcqa]",
    )
    parser.add_argument(
        "--run-name", type=str, default="fl", help="The experiment name"
    )
    parser.add_argument(
        "--quantization",
        type=int,
        default=4,
        help="Quantization value for inference chosen from [4, 8]",
    )
    args = parser.parse_args()

    main(
        base_model_name_path=args.base_model_name_path,
        peft_path=args.peft_path,
        dataset_name=args.dataset_name,
        run_name=args.run_name,
        quantization=args.quantization,
    )
