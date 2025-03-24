# Copyright © 2023-2024 Apple Inc.

import argparse
import sys
import mlx.core as mx

from mlx_lm import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = None
DEFAULT_MAX_TOKENS = 512
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to guide the assistant's behavior."
    )
    return parser

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    begin = True

    if args.seed is not None:
        mx.random.seed(args.seed)

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )

    prompt_cache = make_prompt_cache(model, args.max_kv_size)

    for line in sys.stdin:
        query = line.strip()
        if query == "":
            continue
        if query.startswith("r/"):
            new_prompt = query[2:].strip()
            if new_prompt:
                args.system_prompt = new_prompt
            prompt_cache = make_prompt_cache(model, args.max_kv_size)
            begin = True
        if query == "q":
            break

        if begin:
            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": query},
            ]
            begin = False
        else:
            messages = [
                {"role": "user", "content": query},
            ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        print("[START]", flush=True)
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(args.temp, args.top_p),
            prompt_cache=prompt_cache,
        ):
            print(response.text, flush=True, end="")

        print("[END]", flush=True)


if __name__ == "__main__":
    main()
