import json
import re
import os
import argparse

from sklearn.metrics import accuracy_score


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

benchmark_output_type = {
    'pubmedqa': 'boolean',
    'medmcqa': 'mcq',
    'medqa': 'mcq',
}


def load_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding JSON for line: {line}")
    return data


def clean_mcq_answer(output):
    output = clean_answer(output)
    try:
        output = output[0]
    except Exception:
        return output
    return output


def clean_double_answer(output):
    if "yesyes" in output:
        output = output.replace('yesyes', 'yes')
    elif "nono" in output:
        output = output.replace('nono', 'no')
    elif "yesno" in output:
        output = output.replace('yesno', 'yes')
    elif "noyes" in output:
        output = output.replace('noyes', 'no')
    output = clean_answer(output)
    return output


def clean_answer(output):
    output_clean = output.encode('ascii', 'ignore').decode('ascii')
    return output_clean


def verbose_metric_report(metric_dict):
    print(f'# Accuracy: {metric_dict["accuracy"]}')
    print(f'# Accuracy (calibrated): {metric_dict["accuracy_calibrate"]}')
    print(f'# Precision: {metric_dict["precision"]}')
    print(f'# Recall: {metric_dict["recall"]}')
    print(f'# F1: {metric_dict["f1"]}')

    print(f'# Correct: {metric_dict["correct"]}')
    print(f'# Counted: {metric_dict["counted"]}')
    print(f'# Total: {metric_dict["total"]}')
    print(f'# Unable to find answer: {metric_dict["unable_to_find_answer"]}')
    print(f'# Ignored prompts: {len(metric_dict["ignored"])}')


def eval(output_full, answer, answer_type="mcq"):
    output = output_full
    default = (2, output_full, answer)

    if "\n##" in output:
        try:
            output = output.split("\n##")[1].split("\n")[0].strip().lower()
        except Exception:
            return default
    if "###" in answer:
        try:
            answer = answer.split("answer is:")[1].split("###")[0].strip()
        except Exception:
            return default

    output = re.sub(r"[^a-zA-Z0-9]", " ", output).strip()
    output = re.sub(" +", " ", output)

    if answer_type == 'boolean':
        output = clean_double_answer(output)
    elif answer_type == 'mcq':
        output = clean_mcq_answer(output)

    if output in ['a', 'b', 'c', 'd', 'e', 'yes', 'no']:
        return output == answer, output, answer
    else:
        return default


def accuracy_metric(data, **kwargs):
    acc, counter, error = 0, 0, 0
    preds, golds = [], []
    ignored_prompts = []
    for row in data:
        answer = row['gold'].lower()
        output = row['output'].lower()
        correct, pred, gold = eval(
            output, answer,
            answer_type=kwargs["answer_type"])

        preds.append(pred)
        golds.append(gold)

        if correct == 2:
            error += 1
            correct = 0
            ignored_prompts.append(row)
        else:
            acc += correct
            counter += 1

    accuracy =  accuracy_score(preds, golds)
    assert accuracy == acc / len(data)

    return {
        "accuracy": accuracy_score(preds, golds),
        "accuracy_calibrate": acc / counter,
        "correct": acc,
        "counted": counter,
        "ignored": ignored_prompts,
        "unable_to_find_answer": error,
        "total": len(data)
    }


def display(metric_dict, run_name, benchmark, verbose=False):
    print("====================================")
    print(f'Report accuracy for {run_name} on {benchmark}:')
    print(f'# Accuracy: {metric_dict["accuracy"]}')

    if verbose:
        print(f'# Accuracy (calibrated): {metric_dict["accuracy_calibrate"]}')
        print("------------------------------------")
        print(f'# Correct: {metric_dict["correct"]}')
        print(f'# Counted: {metric_dict["counted"]}')
        print(f'# Total: {metric_dict["total"]}')
        print(f'# Unable to find answer: {metric_dict["unable_to_find_answer"]}')
        print(f'# Ignored prompts: {len(metric_dict["ignored"])}')
    print("====================================")


def main(gen_dir=f'{ROOT_DIR}/benchmarks/generations', dataset_name='pubmedqa', run_name='fl', verbose=True):
    # Load data
    path = f'{gen_dir}/{dataset_name}-{run_name}.jsonl'
    run_name_ = path.split('/')[-1].split('.')[0]
    answer_type = benchmark_output_type[dataset_name]
    data = load_jsonl(path)

    accuracy_kwargs = {'answer_type': answer_type}
    metrics = accuracy_metric(data, **accuracy_kwargs)
    display(
        metrics, run_name_, dataset_name, verbose=verbose
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-dir', type=str, default=f'{ROOT_DIR}/benchmarks/generations',
                        help="The directory to save the generations")
    parser.add_argument('--dataset-name', type=str, default='pubmedqa',
                        help="The benchmark to evaluate on: [pubmedqa, medqa, medmcqa]")
    parser.add_argument('--run-name', type=str, default='fl',
                        help="The experiment name used for inference")
    parser.add_argument('--verbose', action='store_true',
                        help="Whether to print detailed results")
    args = parser.parse_args()

    main(gen_dir=args.gen_dir, dataset_name=args.dataset_name, run_name=args.run_name, verbose=args.verbose)
