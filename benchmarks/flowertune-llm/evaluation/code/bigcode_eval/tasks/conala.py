"""Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow
https://arxiv.org/pdf/1805.08949.pdf

Python Code generation with CoNaLa. It is a benchmark of code and natural language pairs, for the evaluation of code generation tasks. 
The dataset was crawled from Stack Overflow, automatically filtered, then curated by annotators,
split into 2,379 training and 500 test examples.

Homepage: https://conala-corpus.github.io/
Here we use two-shot evaluation (the original paper evaluates finetuned models)
"""

import json

from evaluate import load

from bigcode_eval.base import Task

_CITATION = """
@inproceedings{yin2018learning,
  title={Learning to mine aligned code and natural language pairs from stack overflow},
  author={Yin, Pengcheng and Deng, Bowen and Chen, Edgar and Vasilescu, Bogdan and Neubig, Graham},
  booktitle={2018 IEEE/ACM 15th international conference on mining software repositories (MSR)},
  pages={476--486},
  year={2018},
  organization={IEEE}
}
"""


class Conala(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "neulab/conala"

    def __init__(self, max_order=4, smooth=True):
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )
        self.max_order = max_order
        self.smooth = smooth

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        with open(
            "bigcode_eval/tasks/few_shot_examples/conala_few_shot_prompts.json", "r"
        ) as file:
            examples = json.load(file)
        return examples

    @staticmethod
    def two_shot_prompt(entry, text, examples):
        """Two shot prompt format as instructions & solutions"""
        prompt = f"\nInstruction:\n{examples['instruction1']}\
                   \nSolution:\n{examples['solution1']}\
                   \nInstruction:\n{examples['instruction2']}\
                   \nSolution:\n{examples['solution2']}\
                   \nInstruction:\n{text}\
                   \nSolution:\n"
        assert (
            prompt.count("Solution:\n") == 3
        ), "Splitting operation in postprocess_generation is invalid"
        return entry + prompt

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        examples = self.fewshot_examples()
        text_column = "rewritten_intent" if doc["rewritten_intent"] else "intent"
        text = doc[text_column].strip()
        entry = "Answer the following instructions in one line of Python code:\n"
        prompt = self.two_shot_prompt(entry, text, examples)
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["snippet"]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        output = generation.split("Solution:\n", 3)[-1].strip()
        return output

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        """
        bleu = load("bleu")
        gens = [gen[0] for gen in generations]
        results = bleu.compute(
            references=references, predictions=gens, max_order=self.max_order, smooth=self.smooth
        )
        return results
