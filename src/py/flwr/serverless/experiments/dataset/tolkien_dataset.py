import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class TolkienDataset(Dataset):
    def __init__(self, df, model_name, max_len=256):
        self.model_name = model_name
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
        )

        self.df = df  # pandas dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = self.df["sentences"][idx]
        encodings_dict = self.encode_text(sentence)

        # no labels because input_ids will be used as "labels" for CausalLM
        return {
            "sentence": sentence,
            "input_ids": torch.tensor(encodings_dict["input_ids"]),
            "attention_mask": torch.tensor(encodings_dict["attention_mask"]),
        }

    def encode_text(self, text):
        encodings_dict = self.tokenizer(
            self.bos_token + text + self.eos_token,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        return encodings_dict
