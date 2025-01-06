import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import (
    add_instruct,
    change_target,
    format_example,
    generate_label,
    load_data,
    save_results,
)


def infer_fiqa(model, tokenizer, batch_size, run_name):
    name = "fiqa"
    dataset = load_data("pauri32/fiqa-2018", concat=True)

    # Post process
    dataset["output"] = dataset.sentiment_score.apply(generate_label)
    dataset["instruction"] = dataset.apply(add_instruct, axis=1)
    dataset = dataset[["sentence", "output", "instruction"]]
    dataset.columns = ["input", "output", "instruction"]

    dataset[["context", "target"]] = dataset.apply(
        format_example, axis=1, result_type="expand"
    )

    # Print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    # Run inference
    dataset, acc = inference(dataset, model, tokenizer, batch_size)

    # Save results and generations
    save_results(name, run_name, dataset, acc)


def infer_fpb(model, tokenizer, batch_size, run_name):
    name = "fpb"
    dataset = load_data("takala/financial_phrasebank", "sentences_50agree")

    # Post process
    dataset.columns = ["input", "output"]
    dic = {0: "negative", 1: "neutral", 2: "positive"}
    dataset["output"] = dataset["output"].apply(lambda x: dic[x])

    dataset["instruction"] = (
        "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    )
    dataset[["context", "target"]] = dataset.apply(
        format_example, axis=1, result_type="expand"
    )

    # Print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    # Run inference
    dataset, acc = inference(dataset, model, tokenizer, batch_size)

    # Save results and generations
    save_results(name, run_name, dataset, acc)


def infer_tfns(model, tokenizer, batch_size, run_name):
    name = "tfns"
    dataset = load_data(
        "zeroshot/twitter-financial-news-sentiment", valid_set="validation"
    )

    # Post process
    dic = {0: "negative", 1: "positive", 2: "neutral"}
    dataset["label"] = dataset["label"].apply(lambda x: dic[x])

    dataset["instruction"] = (
        "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
    )

    dataset.columns = ["input", "output", "instruction"]
    dataset[["context", "target"]] = dataset.apply(
        format_example, axis=1, result_type="expand"
    )

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    # Run inference
    dataset, acc = inference(dataset, model, tokenizer, batch_size)

    # Save results and generations
    save_results(name, run_name, dataset, acc)


def inference(dataset, model, tokenizer, batch_size):
    context = dataset["context"].tolist()

    last_batch = dataset.shape[0] % batch_size
    total_steps = dataset.shape[0] // batch_size + 1
    print(
        f"Total len: {len(context)}. Batch size: {batch_size}. Total steps: {total_steps}"
    )

    out_text_list = []
    for i in tqdm(range(total_steps)):
        idx_s = i * batch_size
        tmp_context = (
            context[idx_s : idx_s + last_batch]
            if i == total_steps - 1
            else context[idx_s : idx_s + batch_size]
        )

        if tmp_context:
            tokens = tokenizer(
                tmp_context,
                return_tensors="pt",
                padding=True,
                max_length=512,
                return_token_type_ids=False,
            )
            for k in tokens.keys():
                tokens[k] = tokens[k].cuda()
            res = model.generate(
                **tokens, max_length=512, eos_token_id=tokenizer.eos_token_id
            )
            res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
            out_text = [
                o.split("Answer: ")[1] if len(o.split("Answer: ")) > 1 else "None"
                for o in res_sentences
            ]
            out_text_list += out_text
            torch.cuda.empty_cache()

    dataset["out_text"] = out_text_list
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["new_target"], dataset["new_out"])

    return dataset, acc
