import json

import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import format_answer, format_example, save_results

import datasets

# The instructions refer to Meditron evaluation:
# https://github.com/epfLLM/meditron/blob/main/evaluation/instructions.json
INSTRUCTIONS = {
    "pubmedqa": "As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? Answer yes, no, or maybe.",
    "medqa": "You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. For the following multiple-choice question, select one correct answer from A to E. Base your answer on the current and standard practices referenced in medical guidelines.",
    "medmcqa": "You are a medical doctor answering realworld medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following multiple-choice question. Select one correct answer from A to D. Base your answer on the current and standard practices referenced in medical guidelines.",
    "careqa": "You are a medical doctor answering realworld medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following multiple-choice question. Select one correct answer from A to D. Base your answer on the current and standard practices referenced in medical guidelines.",
}


def infer_pubmedqa(model, tokenizer, batch_size, run_name):
    name = "pubmedqa"
    answer_type = "boolean"
    dataset = datasets.load_dataset(
        "bigbio/pubmed_qa",
        "pubmed_qa_labeled_fold0_source",
        split="test",
        trust_remote_code=True,
    )
    # Post process
    instruction = INSTRUCTIONS[name]

    def post_process(row):
        context = "\n".join(row["CONTEXTS"])
        row["prompt"] = f"{context}\n{row['QUESTION']}"
        row["gold"] = row["final_decision"]
        row["long_answer"] = row["LONG_ANSWER"]
        row["prompt"] = f"{instruction}\n{row['prompt']}\nThe answer is:\n"
        return row

    dataset = dataset.map(post_process)

    # Generate results
    generate_results(name, run_name, dataset, model, tokenizer, batch_size, answer_type)


def infer_medqa(model, tokenizer, batch_size, run_name):
    name = "medqa"
    answer_type = "mcq"
    dataset = datasets.load_dataset(
        "bigbio/med_qa",
        "med_qa_en_4options_source",
        split="test",
        trust_remote_code=True,
    )

    # Post process
    instruction = INSTRUCTIONS[name]

    def post_process(row):
        choices = [opt["value"] for opt in row["options"]]
        row["prompt"] = format_example(row["question"], choices)
        for opt in row["options"]:
            if opt["value"] == row["answer"]:
                row["gold"] = opt["key"]
                break
        row["prompt"] = f"{instruction}\n{row['prompt']}\nThe answer is:\n"
        return row

    dataset = dataset.map(post_process)

    # Generate results
    generate_results(name, run_name, dataset, model, tokenizer, batch_size, answer_type)


def infer_medmcqa(model, tokenizer, batch_size, run_name):
    name = "medmcqa"
    answer_type = "mcq"
    dataset = datasets.load_dataset(
        "medmcqa", split="validation", trust_remote_code=True
    )

    # Post process
    instruction = INSTRUCTIONS[name]

    def post_process(row):
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]
        answer = int(row["cop"])
        row["prompt"] = format_example(row["question"], options)
        row["gold"] = chr(ord("A") + answer) if answer in [0, 1, 2, 3] else None
        row["prompt"] = f"{instruction}\n{row['prompt']}\nThe answer is:\n"
        return row

    dataset = dataset.map(post_process)

    # Generate results
    generate_results(name, run_name, dataset, model, tokenizer, batch_size, answer_type)


def infer_careqa(model, tokenizer, batch_size, run_name):
    name = "careqa"
    answer_type = "mcq"
    dataset = datasets.load_dataset(
        "HPAI-BSC/CareQA",
        "CareQA_en",
        split="test",
        trust_remote_code=True,
    )

    # Post process
    instruction = INSTRUCTIONS[name]

    def post_process(row):
        options = [row["op1"], row["op2"], row["op3"], row["op4"]]
        answer = int(row["cop"]) - 1
        row["prompt"] = format_example(row["question"], options)
        row["gold"] = chr(ord("A") + answer) if answer in [0, 1, 2, 3] else None
        row["prompt"] = f"{instruction}\n{row['prompt']}\nThe answer is:\n"
        return row

    dataset = dataset.map(post_process)

    # Generate results
    generate_results(name, run_name, dataset, model, tokenizer, batch_size, answer_type)


def generate_results(
    name, run_name, dataset, model, tokenizer, batch_size, answer_type
):
    # Run inference
    prediction = inference(dataset, model, tokenizer, batch_size)

    # Calculate accuracy
    acc = accuracy_compute(prediction, answer_type)

    # Save results and generations
    save_results(name, run_name, prediction, acc)


def inference(dataset, model, tokenizer, batch_size):
    columns_process = ["prompt", "gold"]
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
