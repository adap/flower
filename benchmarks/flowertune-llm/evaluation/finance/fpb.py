import os
import warnings
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import datasets
import torch

from utils import format_example, change_target, save_results


warnings.filterwarnings("ignore")

dic = {
    0: "negative",
    1: 'neutral',
    2: 'positive',
}


def test_fpb(args, model, tokenizer, prompt_fun=None):
    name = "fpb"
    batch_size = args.batch_size

    dataset = datasets.load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    dataset = dataset["train"]
    dataset = dataset.train_test_split(seed=42)['test']
    dataset = dataset.to_pandas()
    dataset.columns = ["input", "output"]
    dataset["output"] = dataset["output"].apply(lambda x: dic[x])

    if prompt_fun is None:
        dataset[
            "instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis=1)

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
