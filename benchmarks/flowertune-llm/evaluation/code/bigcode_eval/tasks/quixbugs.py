"""QuixBugs"""

import re

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = """
@inproceedings{lin2017quixbugs,
  title={QuixBugs: A multi-lingual program repair benchmark set based on the Quixey Challenge},
  author={Lin, Derrick and Koppel, James and Chen, Angela and Solar-Lezama, Armando},
  booktitle={Proceedings Companion of the 2017 ACM SIGPLAN international conference on systems, programming, languages, and applications: software for humanity},
  pages={55--56},
  year={2017}
}
"""


class QuixBugs(Task):

    DATASET_PATH = "Muennighoff/quixbugs"

    def __init__(self, prompt="prompt"):
        self.prompt = prompt
        if self.prompt == "edit":
            self.stop_words = [
                "<commit_before>",
                "<commit_msg>", 
                "<commit_after>", 
                "<|endoftext|>",
            ]
        elif self.prompt.startswith("prompt"):
            self.stop_words = [
                "\ndef",
                "\nclass",
                "\n#",
                "\n@",
                "\nprint",
                "\nif",
                "###",
                "///",
                "<|endoftext|>",
            ]
        elif self.prompt.startswith("prompt_codex"):
            # https://arxiv.org/pdf/2111.03922.pdf
            self.stop_words = [
                "\nclass", "###", "///", "<|endoftext|>",
            ]
        else:
            raise ValueError(f"Unknown prompt: {self.prompt}")

        super().__init__(
            stop_words=self.stop_words,
            requires_execution=True,
        )
        self.max_length_multiplier = 3 # Allow 3 times the length of the prompt

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["train"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.prompt == "edit":
            prompt = "<commit_before>" + doc["buggy_program"]
            prompt += "<commit_msg>" + "Fix bug in " + doc["name"]
            prompt += "<commit_after>"
        elif self.prompt == "edit-openai":
            return doc["buggy_program"], "Fix bug in " + doc["name"]
        elif self.prompt == "prompt":
            prompt = "# Buggy function"
            prompt += "\n" + doc["buggy_program"] + "\n"
            prompt += "# Fixed function\ndef"            
        elif self.prompt == "prompt_codex":
            # https://arxiv.org/pdf/2111.03922.pdf, Prenner et al.
            prompt = "### fix the bug in the following function"
            prompt += "\n" + doc["buggy_program"] + "\n"
            prompt += "### fixed function"
        else:
            raise ValueError(f"Unknown prompt: {prompt}")

        return prompt.strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return (doc["name"], doc["tests"].strip())

    @staticmethod
    def remove_last_block(string, stop_words):
        stop_words = [re.escape(word) for word in stop_words] # Escape e.g. | in <|endoftext|>
        # Remove the last block of the code containing stop_words for HumanEval
        string_list = re.split("(%s)" % "|".join(stop_words), string)
        # last string should be ""
        return "".join(string_list[:-2])

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        generation = generation[len(prompt):]
        if self.prompt == "prompt":
            generation = "def" + generation # Add def which is in the prompt back to the output        
        return self.remove_last_block(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results = {}
        for i, (gen, (name, ref)) in enumerate(zip(generations, references)):
            sub_results, _ = compute_code_eval(
                references=[ref],
                predictions=[gen],
                timeout=10, # Levenshtein distance is slow
            )
            results[name] = sub_results
        # Provide average of all metrics computed
        if results:
            results["all"] = {
                k: sum(v[k] for v in results.values()) / len(results) for k in results[list(results.keys())[0]]
            }
            results["num_correct"] = results["all"]["pass@1"] * (len(results) - 1) # -1 for the all metric
        return results
