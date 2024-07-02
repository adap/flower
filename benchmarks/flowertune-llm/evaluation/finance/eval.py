import argparse
import torch

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from fpb import test_fpb
from fiqa import test_fiqa
from tfns import test_tfns
from nwgi import test_nwgi
from headline import test_headline


# Fixed seed
torch.manual_seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument("--peft-path", type=str, default=None)
parser.add_argument("--datasets", type=str, default='fpb')
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--instruct-template", default='default')
args = parser.parse_args()


# Load model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    args.peft_path, torch_dtype=torch.float16
).to("cuda")
base_model = model.peft_config["default"].base_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(base_model)

if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


# Evaluate
model = model.eval()
with torch.no_grad():
    for dataset in args.datasets.split(','):
        if dataset == "fpb":
            test_fpb(args, model, tokenizer)
        elif dataset == 'fiqa':
            test_fiqa(args, model, tokenizer)
        elif dataset == 'tfns':
            test_tfns(args, model, tokenizer)
        elif dataset == 'nwgi':
            test_nwgi(args, model, tokenizer)
        elif dataset == 'headline':
            test_headline(args, model, tokenizer)
        else:
            raise ValueError('Undefined Dataset.')
