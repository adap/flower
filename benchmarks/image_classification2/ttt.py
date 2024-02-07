from datasets import DatasetDict


def resplit(dataset: DatasetDict) -> DatasetDict:
    train_test = dataset["train"].train_test_split(test_size=0.1, seed=1234)

    return DatasetDict({
        "train": train_test["train"],
        "test": train_test["test"],

    })
