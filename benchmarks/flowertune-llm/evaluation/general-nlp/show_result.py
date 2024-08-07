# This python file is adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/show_result.py

import argparse

from fastchat.llm_judge.show_result import display_result_single

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--bench_name", type=str, default="mt_bench")
    parser.add_argument("--judge_model", type=str, default="gpt-4")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    args = parser.parse_args()
    display_result_single(args)
