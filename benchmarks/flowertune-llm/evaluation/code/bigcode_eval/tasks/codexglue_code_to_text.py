"""CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Code to text task from CodeXGlue (documentation generation):
* for all subsets ("python", "java", "javascript", "ruby", "php", "go") where the whole function body (without docstring) is given as a prompt
* for Python subset where only function signature is used as a prompt (this setting can give better results).
"""

import os
import re
import typing

from bigcode_eval.base import Task

_CITATION = """
@article{husain2019codesearchnet,
  title={Codesearchnet challenge: Evaluating the state of semantic code search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}
"""


LANGUAGES = ["python", "java", "javascript", "ruby", "php", "go"]
TRIPLE_QUOTE = '"""'
SINGLE_TRIPLE_QUOTE = "'''"
SPACES4 = " " * 4
SUFFIX_PROMPT = {
    "python": '\n""" The goal of this function is to:\n',
    "ruby": "\n=begin The goal of this function is to:\n",
    "other": "\n/* The goal of this function is to:\n",
}


def create_all_tasks():
    """Creates a dictionary of tasks from a list of languages
    :return: {task_name: task}
        e.g. {codexglue_code_to_text-python: Task, codexglue_code_to_text-java: Task}
    """
    return {
        f"codexglue_code_to_text-{language}": create_task(language)
        for language in LANGUAGES
    }


def create_task(language):
    class CodeToText(GeneralCodeToText):
        def __init__(self, **kwargs):
            super().__init__(language, **kwargs)

    return CodeToText


def compute_codexglue_code_to_text_bleu(
    gold_and_predicted_items: typing.List[typing.Tuple[str, str]]
):

    """
    Compute BLEU scores using codexglue_code_to_text_bleu.computeMaps (codexglue_summarization_evaluator)
    This uses a specific BLEU tokenization and preprocessing necessary for this task by
    the original authors of the dataset.

    Taken from: https://github.com/dpfried/lm-evaluation-harness/blob/5d9a6aaaaa929bcad95bb73d85e78fe75eb64b4e/lm_eval/tasks/codexglue_summarization.py#L102
    """
    from bigcode_eval.tasks.custom_metrics import codexglue_code_to_text_bleu

    predicted_map = {}
    gold_map = {}

    for ix, (gold_str, predicted_str) in enumerate(gold_and_predicted_items):
        gold, *rest = gold_str.strip().split("\t")
        if len(rest) > 0:
            print(f"warning: gold instance {ix} contains a tab; ignoring text after")
        gold_map[ix] = [codexglue_code_to_text_bleu.splitPuncts(gold.strip().lower())]

        pred, *rest = predicted_str.strip().split("\t")
        if len(rest) > 0:
            print(f"warning: gold instance {ix} contains a tab; ignoring text after")
        predicted_map[ix] = [
            codexglue_code_to_text_bleu.splitPuncts(pred.strip().lower())
        ]

    return codexglue_code_to_text_bleu.bleuFromMaps(gold_map, predicted_map)[0]


class GeneralCodeToText(Task):
    """Code to text task from CodeXGlue for all subsets where the whole
    function body (without docstring) is given as a prompt
    """

    DATASET_PATH = "code_x_glue_ct_code_to_text"
    DATASET_NAME = None

    def __init__(self, language):
        self.DATASET_NAME = language
        stop_words = ["'''", '"""'] if language == "python" else ["\n"]
        super().__init__(
            stop_words=stop_words,
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    @staticmethod
    def standardize_docstring_prompt(prefix):
        """Strips any existing docstring delimiters from the prompt prefix
        and adds our own delimiter (triple quote) and whitespace.
        Note an edge cases being handled here:
        - codexglue docstring text sometimes contains the docstring delimiters, inconsistently

        source: InCoder evaluation code https://github.com/dpfried/lm-evaluation-harness/
        """

        for delim in [TRIPLE_QUOTE, SINGLE_TRIPLE_QUOTE]:
            if delim in prefix:
                prefix = prefix[: prefix.index(delim)]
                break

        single_single_quote_with_trailing_spaces = re.compile(r'[^\'"][\']\s*$')
        if single_single_quote_with_trailing_spaces.search(prefix):
            prefix = prefix[
                : single_single_quote_with_trailing_spaces.search(prefix).start()
            ]

        single_double_quote_with_trailing_spaces = re.compile(r'[^\'"]["]\s*$')
        if single_double_quote_with_trailing_spaces.search(prefix):
            prefix = prefix[
                : single_double_quote_with_trailing_spaces.search(prefix).start()
            ]

        prefix += TRIPLE_QUOTE
        return prefix

    def get_prompt(self, doc):
        """Generate prompts for Code to text benchmark (documentation generation)
        Prompt = full function body (withoout the docstring) + '\n[Delimiter] The goal of this function is to:\n'
        where delimiter is  \""" for python, =begin for ruby and /* for the rest (see SUFFIX_PROMPT).
        :param doc: dict[str: str])
        """
        code = doc["code"]

        if self.DATASET_NAME == "python":
            # python code includes the docstring
            text = doc["docstring"]
            prompt_prefix = code[: code.index(text)]
            prompt_prefix = self.standardize_docstring_prompt(prompt_prefix)
            prompt_suffix = code[code.index(text) + len(text) :]
            prompt_suffix = prompt_suffix.replace(TRIPLE_QUOTE, "")
            prompt_suffix = prompt_suffix.replace(SINGLE_TRIPLE_QUOTE, "")

            prompt_prefix = prompt_prefix.strip().removesuffix(TRIPLE_QUOTE)
            prompt_prefix = prompt_prefix.strip().removesuffix(SINGLE_TRIPLE_QUOTE)
            prompt = prompt_prefix + prompt_suffix + SUFFIX_PROMPT["python"]
            return prompt

        elif self.DATASET_NAME == "ruby":
            return code + SUFFIX_PROMPT["ruby"]

        else:
            return code + SUFFIX_PROMPT["other"]

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
        """
        from mosestokenizer import MosesDetokenizer
        # deactivate tokenizer parallelism when calling MosesDetokenizer TODO: do it for all refs once
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # docstring_tokens are preprocessed and don't have extra context like variable defs
        docstring = " ".join(doc["docstring_tokens"]).replace("\n", "")
        # some docstrings started with r""" before tokenization but r was kept
        if docstring[0] == "r":
            docstring = docstring[1:]
        with MosesDetokenizer("en") as detokenize:
            docstring = detokenize(docstring.strip().split())
        return docstring

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this Task)
        """
        delimiters = {language: SUFFIX_PROMPT["other"] for language in LANGUAGES}
        delimiters.update(SUFFIX_PROMPT)
        output = generation.split(delimiters[self.DATASET_NAME])[1].strip()
        output = output.split("\n")[0]
        return output

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences (not needed for APPS Task)
        """
        bleu_score = compute_codexglue_code_to_text_bleu(
            (ref, gen[0]) for ref, gen in zip(references, generations)
        )
        return {"blue": bleu_score}


class LeftCodeToText(GeneralCodeToText):
    """Code to text task from CodeXGlue for Python subset in a left only setting:
    only the function signature is given as prompt similarly to Fried et al. (InCoder)

    TODO: implement function signature extraction for other languages in the dataset
    """

    def __init__(self):
        super().__init__("python")

    @staticmethod
    def standardize_docstring_prompt(prefix):
        """Strips any existing docstring delimiters from the prompt prefix and
        and adds our own delimiter (triple quote) and whitespace.
        Note an edge cases being handled here:
        - codexglue docstring text sometimes contains the docstring delimiters, inconsistently

        source: InCoder evaluation code https://github.com/dpfried/lm-evaluation-harness/
        """

        for delim in [TRIPLE_QUOTE, SINGLE_TRIPLE_QUOTE]:
            if delim in prefix:
                prefix = prefix[: prefix.index(delim)]
                break

        single_single_quote_with_trailing_spaces = re.compile(r'[^\'"][\']\s*$')
        if single_single_quote_with_trailing_spaces.search(prefix):
            prefix = prefix[
                : single_single_quote_with_trailing_spaces.search(prefix).start()
            ]

        single_double_quote_with_trailing_spaces = re.compile(r'[^\'"]["]\s*$')
        if single_double_quote_with_trailing_spaces.search(prefix):
            prefix = prefix[
                : single_double_quote_with_trailing_spaces.search(prefix).start()
            ]

        prefix += TRIPLE_QUOTE
        return prefix

    def get_prompt(self, doc):
        """Generate prompts for Code to text benchmark (documentation generation)
        Prompt =  function signature.
        :param doc: dict[str: str]
        """
        code = doc["code"]
        # python code includes the docstring
        text = doc["docstring"]
        prompt_prefix = code[: code.index(text)]
        prompt_prefix = self.standardize_docstring_prompt(prompt_prefix)
        return prompt_prefix

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this Task)
        """
        output = generation.strip().split("\n")[0].strip()
        for delimiter in [TRIPLE_QUOTE, SINGLE_TRIPLE_QUOTE]:
            if delimiter in generation:
                generation = generation[generation.index(delimiter) + 3 :]
                output = generation.strip().split("\n")[0].strip()
                output = output.split(delimiter, 1)[0]
        return output
