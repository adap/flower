# This python file is adapted from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/custom_metrics/multiple_metrics/evaluation.py

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

from . import eval_cpp, eval_javascript

# Get working directory
WORKING_DIR = Path(__file__).parent.parent

# program: str => Result
CACHE = dict()
CACHE_LOCK = Lock()

EVALUATORS = {
    "javascript": (eval_javascript.eval_script, ".js"),
    "js": (eval_javascript.eval_script, ".js"),
    "cpp": (eval_cpp.eval_script, ".cpp"),
}


def cache_get(program: str) -> Optional[dict]:
    if program in CACHE:
        result = CACHE[program]
        return result
    else:
        return None


def cache_set(program: str, result: dict):
    if program in CACHE:
        print("Setting already-existing cache")
    CACHE[program] = result


def cached_eval_script(problem, index) -> dict:
    # here prompt is already included in completions
    program = problem["completions"][index] + "\n" + problem["tests"]
    CACHE_LOCK.acquire(True)
    cached = cache_get(program)
    if cached is not None:
        CACHE_LOCK.release()
        return cached
    else:
        result_yaml = dict()
        cache_set(program, result_yaml)
        CACHE_LOCK.release()
        result_dict = eval_string_script(problem["language"], program)
        for k in result_dict.keys():
            result_yaml[k] = result_dict[k]
            result_yaml["timestamp"] = int(time.time())
        return result_yaml


def get_test_results_json_path(
    output_dir: str, problem_json_path: str, input_dir: Path
) -> Path:
    suffixes = ".results.json"
    problem_name = problem_json_path[: -len(".json")]
    if input_dir:
        raise ValueError("input dir given")
        return Path(output_dir) / (
            problem_json_path.relative_to(input_dir).parent / (problem_name + suffixes)
        )
    return Path(output_dir) / (problem_name + suffixes)


def evaluate_problem(
    output_dir: str, problem_json_path: str, max_workers: int, input_dir: Path = None
):
    with open(problem_json_path, "r") as f:
        problem = json.load(f)
    test_results_path = get_test_results_json_path(
        output_dir, problem_json_path, input_dir
    )
    test_results_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)

    test_results = problem.copy()
    del test_results["completions"]
    test_results["results"] = []

    num_problems = len(problem["completions"])
    min_problem = len(test_results["results"])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for j in executor.map(
            lambda index: cached_eval_script(problem, index),
            range(min_problem, num_problems),
        ):
            test_results["results"].append(j)
            with open(test_results_path, "w") as f:
                f.write(json.dumps(test_results, indent=2))


def eval_string_script(language, program):
    if language in EVALUATORS:
        (eval_script, file_ext) = EVALUATORS[language]
    else:
        eval_module = __import__(
            f"eval_{language}" if language != "go_test.go" else "eval_go"
        )
        eval_script = eval_module.eval_script
        file_ext = f".{language}" if language != "go_test.go" else "_test.go"
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=True) as f:
        f.write(program.encode("utf-8"))
        f.flush()
        result = eval_script(Path(f.name))
        # Only save the first 2K of output from the running program. Any futher
        # output is very likely an exceptionally long stack trace or a long
        # series of prints.
        if type(result["stdout"]) == bytes:
            result["stdout"] = result["stdout"].decode("utf-8", errors="ignore")
        if result["stdout"] is None:
            result["stdout"] = ""
        if result["stderr"] is None:
            result["stderr"] = ""
        if type(result["stderr"]) == bytes:
            result["stderr"] = result["stderr"].decode("utf-8", errors="ignore")
        assert type(result["stdout"]) == str
        assert type(result["stderr"]) == str
        return {
            "program": program,
            "stdout": result["stdout"].replace("!!int", "")[:2048],
            "stderr": result["stderr"][:2048],
            "exit_code": result["exit_code"],
            "status": result["status"],
        }
