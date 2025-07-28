import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmarks import infer_careqa, infer_medmcqa, infer_medqa, infer_pubmedqa

# Fixed seed
torch.manual_seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-model-name-path", type=str, default="mistralai/Mistral-7B-v0.3"
)
parser.add_argument("--run-name", type=str, default="fl")
parser.add_argument("--peft-path", type=str, default=None)
parser.add_argument(
    "--datasets",
    type=str,
    default="pubmedqa",
    help="The dataset to infer on: [pubmedqa, medqa, medmcqa, careqa]",
)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--quantization", type=int, default=4)
args = parser.parse_args()


# Load model and tokenizer
if args.quantization == 4:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    torch_dtype = torch.float32
elif args.quantization == 8:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    torch_dtype = torch.float16
else:
    raise ValueError(
        f"Use 4-bit or 8-bit quantization. You passed: {args.quantization}/"
    )

model = AutoModelForCausalLM.from_pretrained(
    args.base_model_name_path,
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
)
if args.peft_path is not None:
    model = PeftModel.from_pretrained(
        model, args.peft_path, torch_dtype=torch_dtype
    ).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_path)

# Evaluate
for dataset in args.datasets.split(","):
    if dataset == "pubmedqa":
        infer_pubmedqa(model, tokenizer, args.batch_size, args.run_name)
    elif dataset == "medqa":
        infer_medqa(model, tokenizer, args.batch_size, args.run_name)
    elif dataset == "medmcqa":
        infer_medmcqa(model, tokenizer, args.batch_size, args.run_name)
    elif dataset == "careqa":
        infer_careqa(model, tokenizer, args.batch_size, args.run_name)
    else:
        raise ValueError("Undefined Dataset.")
