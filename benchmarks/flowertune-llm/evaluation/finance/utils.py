import os
from datasets import Dataset


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'


def save_results(dataset_name, run_name, dataset, acc):
    path = './benchmarks/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save results    
    results_path = os.path.join(path, f"results_{dataset_name}_{run_name}.txt")
    with open(results_path, "w") as f:
        f.write(f"Accuracy: {acc}. ")
    print(f"Accuracy: {acc}. ")
    
    # Save generations 
    generation_path = os.path.join(path, f"generation_{dataset_name}_{run_name}.jsonl")
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.remove_columns(["input", "output", "instruction", "target", "out_text"])
    dataset.to_json(generation_path, orient="records")
