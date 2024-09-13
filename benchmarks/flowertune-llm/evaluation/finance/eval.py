import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmarks import infer_fiqa, infer_fpb, infer_tfns

# Fixed seed
torch.manual_seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-model-name-path", type=str, default="mistralai/Mistral-7B-v0.3"
)
parser.add_argument("--run-name", type=str, default="fl")
parser.add_argument("--peft-path", type=str, default=None)
parser.add_argument("--datasets", type=str, default="fpb")
parser.add_argument("--batch-size", type=int, default=32)
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

if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))


# Evaluate
model = model.eval()
with torch.no_grad():
    for dataset in args.datasets.split(","):
        if dataset == "fpb":
            infer_fpb(model, tokenizer, args.batch_size, args.run_name)
        elif dataset == "fiqa":
            infer_fiqa(model, tokenizer, args.batch_size, args.run_name)
        elif dataset == "tfns":
            infer_tfns(model, tokenizer, args.batch_size, args.run_name)
        else:
            raise ValueError("Undefined Dataset.")
