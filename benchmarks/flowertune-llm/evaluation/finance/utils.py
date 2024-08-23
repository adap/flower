import os

import datasets
from datasets import Dataset


def load_data(dataset_path, name=None, concat=False, valid_set=None):
    dataset = datasets.load_dataset(dataset_path, name, trust_remote_code=True)

    if concat:
        dataset = datasets.concatenate_datasets(
            [dataset["train"], dataset["validation"], dataset["test"]]
        )

    if valid_set:
        dataset = dataset[valid_set]
    else:
        dataset = dataset if concat else dataset["train"]
        dataset = dataset.train_test_split(0.25, seed=42)["test"]

    dataset = dataset.to_pandas()

    return dataset


def format_example(example: dict):
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def generate_label(value):
    return "negative" if value < -0.1 else "neutral" if value < 0.1 else "positive"


def add_instruct(content):
    tag = "tweet" if content.format == "post" else "news"
    return f"What is the sentiment of this {tag}? Please choose an answer from {{negative/neutral/positive}}."


def change_target(x):
    if "positive" in x or "Positive" in x:
        return "positive"
    elif "negative" in x or "Negative" in x:
        return "negative"
    else:
        return "neutral"


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
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.remove_columns(
        ["input", "output", "instruction", "target", "out_text"]
    )
    dataset.to_json(generation_path, orient="records")
