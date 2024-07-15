# This python file is adapted from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/custom_metrics/multiple_metrics/eval_cpp.py

import argparse
import sys
from pathlib import Path
from sys import exit as sysexit

from .safe_subprocess import run

LANG_NAME = "C++"
LANG_EXT = ".cpp"


def eval_script(path: Path):
    basename = ".".join(str(path).split(".")[:-1])
    build_result = run(["g++", path, "-o", basename, "-std=c++17"])
    if build_result.exit_code != 0:
        return {
            "status": "SyntaxError",
            "exit_code": build_result.exit_code,
            "stdout": build_result.stdout,
            "stderr": build_result.stderr,
        }

    run_result = run([basename])
    if "In file included from /shared/centos7/gcc/9.2.0-skylake/" in run_result.stderr:
        raise Exception("Skylake bug encountered")
    if "/4.8.2" in run_result.stderr:
        raise Exception("Ancient compiler encountered")
    if run_result.timeout:
        status = "Timeout"
    elif run_result.exit_code != 0:
        status = "Exception"
    else:
        status = "OK"
    return {
        "status": status,
        "exit_code": run_result.exit_code,
        "stdout": run_result.stdout,
        "stderr": run_result.stderr,
    }


def list_files(directory, ext):
    files_unsorted = directory.glob(f"HumanEval_*{ext}")

    # assumption: base filenames are in the format of HumanEval_X_*
    # Where X is a valid number
    def key(s):
        return int(str(s.name).split("_")[1])

    files_sorted = sorted(files_unsorted, key=(lambda s: key(s)))

    # assumption: there may be missing files, but no extra files
    # so we build files_array where the index corresponds to the file's number,
    # and a missing file is represented by None
    size = key(files_sorted[-1]) + 1
    files_array = [None] * size
    for f in files_sorted:
        k = key(f)
        files_array[k] = f

    return files_array


def main(eval_script, language, extension):
    args = argparse.ArgumentParser()

    args.add_argument(
        "--directory", type=str, required=True, help="Directory to read benchmarks from"
    )
    args.add_argument(
        "--files",
        type=int,
        nargs="*",
        default=[],
        help="Specify the benchmarks to evaluate by their number, e.g. --files 0 1 2",
    )
    args = args.parse_args()

    directory = Path(args.directory).resolve()

    files_sorted = list_files(directory, extension)

    # the directory you specified does not contain the right language
    if len(files_sorted) == 0:
        print(f"The specified directory does not contain files of type {extension}")
        sysexit(1)

    files_index = []
    if len(args.files) > 0:
        files_index = args.files
    else:
        files_index = range(len(files_sorted))

    total = 0
    passed = 0
    syntax_error = 0

    results_file = Path(
        Path(__file__).parent, "..", "results", language.lower() + ".csv"
    ).resolve()

    with open(results_file, "w") as f:
        for i in files_index:
            filepath = files_sorted[i]
            if filepath is None:
                print("File {} does not exist!".format(i))
                continue
            res = eval_script(filepath)
            output = f"{language},{filepath.stem},{res['status']}\n"
            f.write(output)
            print(output, end="")
            total += 1
            if res["status"] == "OK":
                passed += 1
            elif res["status"] == "SyntaxError":
                syntax_error += 1
    print(f"Total {total}, Syntax Error {syntax_error}, Passed {passed}")


if __name__ == "__main__":
    main(eval_script, LANG_NAME, LANG_EXT)
