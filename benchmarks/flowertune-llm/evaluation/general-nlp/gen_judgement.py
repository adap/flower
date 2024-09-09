"""
This python file is adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_judgment.py

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
import json

from fastchat.llm_judge.common import (
    NEED_REF_CATS,
    check_data,
    get_model_list,
    load_judge_prompts,
    load_model_answers,
    load_questions,
    play_a_match_single,
)
from fastchat.llm_judge.gen_judgment import make_judge_single, make_match_single
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    args = parser.parse_args()

    question_file = "data/mt_bench/question.jsonl"
    answer_dir = "data/mt_bench/model_answer"
    ref_answer_dir = "data/mt_bench/reference_answer"

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    judges = make_judge_single(args.judge_model, judge_prompts)
    play_a_match_func = play_a_match_single
    output_file = f"data/mt_bench/model_judgment/{args.judge_model}_single.jsonl"
    make_match_func = make_match_single
    baseline_model = None

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = "mt_bench"
    match_stat["mode"] = "single"
    match_stat["judge"] = args.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    input("Press Enter to confirm...")

    # Play matches
    for match in tqdm(matches):
        play_a_match_func(match, output_file=output_file)
