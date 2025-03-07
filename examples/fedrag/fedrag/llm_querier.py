"""fedrag: A Flower Federated RAG app."""

import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMQuerier:

    def __init__(self, model_name, use_gpu=False):
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def answer(self, question, documents, options, dataset_name, max_new_tokens=10):
        # Format options as A) ... B) ... etc.
        formatted_options = "\n".join([f"{k}) {v}" for k, v in options.items()])

        prompt = self.__format_prompt(
            question, documents, formatted_options, dataset_name
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(
            self.device
        )

        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            early_stopping=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return full_response, self.__parse_response(full_response, prompt)

    @classmethod
    def __format_prompt(cls, question, documents, options, dataset_name):
        instruction = None
        if dataset_name == "pubmedqa":
            instruction = "As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? Answer yes, no, or maybe."
        elif dataset_name == "bioasq":
            "You are an advanced biomedical AI assistant trained to understand and process medical and scientific texts. Given a biomedical question, your goal is to provide a concise and accurate answer based on relevant scientific literature."

        ctx_documents = "\n".join(
            [f"Document {i + 1}: {doc}" for i, doc in enumerate(documents)]
        )
        prompt = f"""{instruction}

            Here are the relevant documents:
            {ctx_documents}

            Question: 
            {question}

            Options:
            {options}

            Please answer with only the correct option: """
        return prompt

    @classmethod
    def __parse_response(cls, full_response, original_prompt):
        # Extract only the new generated text
        response = full_response[len(original_prompt) :].strip()

        # Find first occurrence of A-D (case-insensitive)
        match = re.search(r"\b([A-Da-d])\b", response)
        if match:
            return match.group(1).upper()
        return None
