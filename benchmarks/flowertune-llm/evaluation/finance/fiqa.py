import warnings
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import datasets
import torch

from utils import format_example, change_target, save_results


warnings.filterwarnings("ignore")


def add_instructions(x):
    if x.format == "post":
        return "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
    else:
        return "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."


def make_label(x):
    if x < - 0.1:
        return "negative"
    elif x >= -0.1 and x < 0.1:
        return "neutral"
    elif x >= 0.1:
        return "positive"


def test_fiqa(args, model, tokenizer, prompt_fun=add_instructions):
    name = "fiqa"
    batch_size = args.batch_size
    
    dataset = datasets.load_dataset('pauri32/fiqa-2018', trust_remote_code=True)
    dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    dataset = dataset.train_test_split(0.226, seed=42)['test']
    dataset = dataset.to_pandas()
    dataset["output"] = dataset.sentiment_score.apply(make_label)
    if prompt_fun is None:
        dataset[
            "instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis=1)

    dataset = dataset[['sentence', 'output', "instruction"]]
    dataset.columns = ["input", "output", "instruction"]
    dataset[["context", "target"]] = dataset.apply(format_example, axis=1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()

    last_batch = dataset.shape[0] % batch_size
    total_steps = dataset.shape[0] // batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text_list = []

    for i in tqdm(range(total_steps)):
        idx_s = i * batch_size
        tmp_context = context[idx_s:idx_s + last_batch] if i == total_steps - 1 else context[idx_s:idx_s + batch_size]
        
        if tmp_context:
            tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512, return_token_type_ids=False)
            for k in tokens.keys():
                tokens[k] = tokens[k].cuda()
            res = model.generate(**tokens, max_length=512, eos_token_id=tokenizer.eos_token_id)
            res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
            out_text = [o.split("Answer: ")[1] for o in res_sentences]
            out_text_list += out_text
            torch.cuda.empty_cache()

    dataset["out_text"] = out_text_list
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["new_target"], dataset["new_out"])

    # Save results and generations
    save_results(name, args.run_name, dataset, acc)
