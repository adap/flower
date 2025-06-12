import json

import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import format_answer, format_example, save_results

from datasets import Dataset, load_dataset

INSTRUCTIONS = {
    "mmlu": "Answer the following multiple choice question.",
}

MMLU_CATEGORY = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "social_sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ],
}


def infer_mmlu(model, tokenizer, batch_size, category, run_name):
    name = "mmlu"
    answer_type = "mcq"

    # Download dataset
    dataframes = []
    for subset in MMLU_CATEGORY[category]:
        subset_data = load_dataset(
            "lukaemon/mmlu",
            subset,
            split="test",
            trust_remote_code=True,
        )
        subset_df = pd.DataFrame(subset_data.map(lambda x: {"subset": subset, **x}))
        dataframes.append(subset_df)

    dataset_df = pd.concat(dataframes, axis=0)
    dataset = Dataset.from_pandas(dataset_df)
    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns("__index_level_0__")

    # Post process
    instruction = INSTRUCTIONS[name]

    def post_process(row):
        options = [row["A"], row["B"], row["C"], row["D"]]
        row["prompt"] = format_example(row["input"], options)
        row["gold"] = row["target"]
        row["subset"] = row["subset"]
        row["prompt"] = f"{instruction}\n{row['prompt']}\nThe answer is:\n"
        return row

    dataset = dataset.map(post_process)

    # Generate results
    generate_results(
        name, run_name, dataset, model, tokenizer, batch_size, answer_type, category
    )


def generate_results(
    name, run_name, dataset, model, tokenizer, batch_size, answer_type, category
):
    # Run inference
    prediction = inference(dataset, model, tokenizer, batch_size)

    # Calculate accuracy
    acc = accuracy_compute(prediction, answer_type)

    # Save results and generations
    save_results(name, category, run_name, prediction, acc)


def inference(dataset, model, tokenizer, batch_size):
    columns_process = ["prompt", "gold"]
    if "subset" in dataset.features:
        columns_process.append("subset")
    dataset_process = pd.DataFrame(dataset, columns=dataset.features)[columns_process]
    dataset_process = dataset_process.assign(output="Null")
    temperature = 1.0

    inference_data = json.loads(dataset_process.to_json(orient="records"))
    data_loader = DataLoader(inference_data, batch_size=batch_size, shuffle=False)

    batch_counter = 0
    for batch in tqdm(data_loader, total=len(data_loader), position=0, leave=True):
        prompts = [
            f"<|im_start|>question\n{prompt}<|im_end|>\n<|im_start|>answer\n"
            for prompt in batch["prompt"]
        ]
        if batch_counter == 0:
            print(prompts[0])

        # Process tokenizer
        stop_seq = ["###"]
        if tokenizer.eos_token is not None:
            stop_seq.append(tokenizer.eos_token)
        if tokenizer.pad_token is not None:
            stop_seq.append(tokenizer.pad_token)
        max_new_tokens = len(
            tokenizer(batch["gold"][0], add_special_tokens=False)["input_ids"]
        )

        outputs = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            output_ids = model.generate(
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=1.0,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            output_ids = output_ids[0][len(input_ids[0]) :]
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            outputs.append(output)

        for prompt, out in zip(batch["prompt"], outputs):
            dataset_process.loc[dataset_process["prompt"] == prompt, "output"] = out
        batch_counter += 1

    return dataset_process


def accuracy_compute(dataset, answer_type):
    dataset = json.loads(dataset.to_json(orient="records"))
    preds, golds = [], []
    for row in dataset:
        answer = row["gold"].lower()
        output = row["output"].lower()
        pred, gold = format_answer(output, answer, answer_type=answer_type)
        preds.append(pred)
        golds.append(gold)

    accuracy = accuracy_score(preds, golds)

    return accuracy
