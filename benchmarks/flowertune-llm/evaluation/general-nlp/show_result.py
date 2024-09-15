"""
This python file is adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/show_result.py

FastChat (https://github.com/lm-sys/FastChat) is licensed under the Apache License, Version 2.0.

Citation:
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu
      and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang
      and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

import argparse

from fastchat.llm_judge.show_result import display_result_single

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    args = parser.parse_args()
    display_result_single(args)
