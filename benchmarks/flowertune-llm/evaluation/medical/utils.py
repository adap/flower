import os
import re


def format_example(question, choices):
    if not question.endswith("?") and not question.endswith("."):
        question += "?"
    options_str = "\n".join([f"{chr(65+i)}. {choices[i]}" for i in range(len(choices))])
    prompt = "Question: " + question + "\n\nOptions:\n" + options_str
    return prompt


def save_results(dataset_name, run_name, dataset, acc):
    path = "./benchmarks/"
    if not os.path.exists(path):
        os.makedirs(path)

    # Save results
    results_path = os.path.join(path, f"acc_{dataset_name}_{run_name}.txt")
    with open(results_path, "w") as f:
        f.write(f"Accuracy: {acc}. ")
    print(f"Accuracy: {acc}. ")

    # Save generations
    generation_path = os.path.join(path, f"generation_{dataset_name}_{run_name}.jsonl")
    dataset.to_json(generation_path, orient="records")


def format_answer(output_full, answer, answer_type="mcq"):
    output = output_full
    default = (output_full, answer)
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

    if answer_type == "boolean":
        output = clean_boolean_answer(output)
    elif answer_type == "mcq":
        output = clean_mcq_answer(output)

    if output in ["a", "b", "c", "d", "e", "yes", "no"]:
        return output, answer
    else:
        return default


def clean_mcq_answer(output):
    output = clean_answer(output)
    try:
        output = output[0]
    except Exception:
        return output
    return output


def clean_boolean_answer(output):
    if "yesyes" in output:
        output = output.replace("yesyes", "yes")
    elif "nono" in output:
        output = output.replace("nono", "no")
    elif "yesno" in output:
        output = output.replace("yesno", "yes")
    elif "noyes" in output:
        output = output.replace("noyes", "no")
    output = clean_answer(output)
    return output


def clean_answer(output):
    output_clean = output.encode("ascii", "ignore").decode("ascii")
    return output_clean
