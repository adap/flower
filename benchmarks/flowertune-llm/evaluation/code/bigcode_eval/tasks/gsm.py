"""PAL: Program-aided Language Models
https://arxiv.org/abs/2211.10435

GSM-8k: Training Verifiers to Solve Math Word Problems
https://arxiv.org/abs/2110.14168

In PaL, Large Language Model solves reasoning problems that involve complex arithmetic and procedural tasks by generating 
reasoning chains of text and code.This offloads the execution of the code to a program runtime, in our case, a Python interpreter.

This task implements PAL methodology to evaluate GSM-8k and GSM-Hard benchmarks.
"""

import json
import os
import re
from enum import Enum
from typing import Union

from evaluate import load

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.pal_metric.pal_code_exec import compute

_CITATION = """
@article{gao2022pal,
  title={PAL: Program-aided Language Models},
  author={Gao, Luyu and Madaan, Aman and Zhou, Shuyan and Alon, Uri and Liu, Pengfei and Yang, Yiming and Callan, Jamie and Neubig, Graham},
  journal={arXiv preprint arXiv:2211.10435},
  year={2022}
}

@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
"""
# Number of few shot examples to consider
NUM_SHOTS = 8


class EvaluationType(str, Enum):
    """Possible values for evaluation type argument"""

    GREEDY = "greedy"
    MAJORITY_VOTING = "majority_voting"


def create_all_tasks():
    """Creates a dictionary of tasks for all evalution type
    :return: {task_name: task}
        e.g. {pal-gsm8k-greedy: Task, pal-gsm8k-majority_voting: Task}
    """

    tasks = [Gsm8k, GsmHard]
    eval_types = [et.value for et in EvaluationType]

    return {
        f"pal-{task.__name__.lower()}-{eval_type}": create_task(task, eval_type)
        for eval_type in eval_types
        for task in tasks
    }


def create_task(cls, evaluation_type):
    class Gsm(cls):
        def __init__(self, **kwargs):
            super().__init__(evaluation_type, **kwargs)

    return Gsm


class Gsm8k(Task):

    DATASET_PATH = "gsm8k"
    DATASET_NAME = "main"
    POST_SCRIPT = "print(solution())"
    SPLIT = "test"

    def __init__(
        self, evaluation_type: Union[str, EvaluationType] = EvaluationType.GREEDY
    ):
        """
        :param evaluation_type: Union[str,EvaluationType]
            Type of evaluation to perform. Authors of PAL had originally evaluated the generations on greedy and majority voting methods.
            Values can be `greedy` or `majority_voting`
            greedy: One Generation is sampled using greedy decoding and evaluated against references
            majority_voting: Predicted answer is selected from multiple generations based on majority voting and evaluated.
        """
        stop_words = ["\n\n\n"]
        requires_execution = True
        if evaluation_type == EvaluationType.MAJORITY_VOTING:
            self.majority_voting = True
        else:
            self.majority_voting = False
        super().__init__(stop_words, requires_execution)

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        if self.SPLIT:
            return self.dataset[self.SPLIT]
        return self.dataset

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        with open(
            "bigcode_eval/tasks/few_shot_examples/gsm8k_few_shot_prompts.json",
            "r",
        ) as file:
            examples = json.load(file)
        return examples

    @staticmethod
    def few_shot_prompt(entry, text, examples):
        """Two shot prompt format as source & target language documentation"""
        prompt = ""
        for question, solution in zip(
            examples["questions"][:NUM_SHOTS], examples["solutions"][:NUM_SHOTS]
        ):
            prompt += f'''Q: {question}\n\n# solution in Python:\n\n\ndef solution():\n    """{question}"""\n{solution}\n\n\n\n\n\n'''
        prompt += f"""Q: {text}\n\n# solution in Python:\n\n\n"""
        return entry + prompt

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        text = doc["question"]
        entry = f""
        examples = self.fewshot_examples()
        prompt = self.few_shot_prompt(entry, text, examples)
        return prompt

    @staticmethod
    def parse_target(txt):
        def _is_num(txt):
            try:
                txt = txt.replace(",", "")
                float(txt)
            except ValueError:
                return False
            return True

        txt = txt.strip()
        if _is_num(txt):
            txt = txt.replace(",", "")
            try:
                num = int(txt)
            except ValueError:
                num = float(txt)
            return num
        return txt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        _answer_delim = "#### "
        target = doc["answer"].split(_answer_delim)[-1]
        return self.parse_target(target)

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        output = generation.split("# solution in Python:", NUM_SHOTS + 1)[-1].strip()
        if "Q:" in output:
            output = output.split("Q:")[0]
        output += "\n" + self.POST_SCRIPT
        return output

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(float)
            list of references
        """
        results = compute(
            references=references,
            predictions=generations,
            majority_voting=self.majority_voting,
        )
        return results


class GsmHard(Gsm8k):
    DATASET_PATH = "reasoning-machines/gsm-hard"
    DATASET_NAME = None
    # the default split of GSMHARD - actually taken from test split of GSM dataset
    SPLIT = "train"

    def __init__(self, evaluation_type: str = EvaluationType.GREEDY):
        """
        :param evaluation_type: str
            Type of evaluation to perform. Authors of PAL had originally evaluated the generations on greedy and majority voting methods.
            Values can be `greedy` or `majority_voting`
            greedy: One Generation is sampled using greedy decoding and evaluated against references
            majority_voting: Predicted answer is selected from multiple generations based on majority voting and evaluated.
        """
        super().__init__(evaluation_type)

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        text = doc["input"]
        entry = ""
        examples = self.fewshot_examples()
        prompt = self.few_shot_prompt(entry, text, examples)
        return prompt

    def get_reference(self, doc):
        return doc["target"]
