"""
CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Text to text task from CodeXGlue (documentation translation)
"""

import json
import os
import re

from evaluate import load

from bigcode_eval.base import Task

_CITATION = """
@article{CodeXGLUE,
         title={CodeXGLUE: A Benchmark Dataset and Open Challenge for Code Intelligence},
         year={2020},}
"""

SOURCE_LANG = {
    "da_en": "danish",
    "zh_en": "chinese",
    "no_en": "norwegian",
    "lv_en": "latvian",
}


def create_all_tasks():
    """Creates a dictionary of tasks from a list of languages
    :return: {task_name: task}
        e.g. {codexglue_text_to_text-da_en: Task, codexglue_text_to_text-zh_en: Task}
    """
    return {
        f"codexglue_text_to_text-{translation_task}": create_task(translation_task)
        for translation_task in SOURCE_LANG
    }


def create_task(translation_task):
    class CodexglueTextToTextTask(CodexglueTextToText):
        def __init__(self, **kwargs):
            super().__init__(translation_task, **kwargs)

    return CodexglueTextToTextTask


class CodexglueTextToText(Task):

    DATASET_PATH = "code_x_glue_tt_text_to_text"
    DATASET_NAME = None

    def __init__(self, translation_task, max_order=4, smooth=True):
        self.DATASET_NAME = translation_task
        stop_words = ["\n"]
        requires_execution = False
        super().__init__(stop_words, requires_execution)
        self.max_order = max_order
        self.smooth = smooth

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        with open(
            "bigcode_eval/tasks/few_shot_examples/codexglue_text_to_text_few_shot_prompts.json",
            "r",
        ) as file:
            examples = json.load(file)
        return examples

    @staticmethod
    def two_shot_prompt(entry, text, examples, language):
        """Two shot prompt format as source & target language documentation"""
        prompt = f"\n{language.title()}:\n{examples['source1']}\
                   \nEnglish:\n{examples['target1']}\
                   \n{language.title()}:\n{examples['source2']}\
                   \nEnglish:\n{examples['target2']}\
                   \n{language.title()}:\n{text}\
                   \nEnglish:\n"
        return entry + prompt

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        language = SOURCE_LANG[self.DATASET_NAME]
        text = doc["source"]
        entry = f"Translate the following documentation from {language.title()} to English:\n"
        examples = self.fewshot_examples()
        examples = examples[language]
        prompt = self.two_shot_prompt(entry, text, examples, language)
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["target"].strip()

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        output = generation.split("\nEnglish:\n", 3)[-1].strip()
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
