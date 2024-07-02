"""Mapping Language to Code in Programmatic Context (Concode)
https://arxiv.org/abs/1808.09588

CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Java code generation in CodeXGLUE text-to-code dataset (built from Concode dataset)
Available at https://huggingface.co/datasets/code_x_glue_ct_code_to_text
2000 samples are available in the test set.

Here we use two-shot evaluation (the original paper evaluates finetuned models)
"""
import json

from evaluate import load

from bigcode_eval.base import Task

_CITATION = """
@article{iyer2018mapping,
  title={Mapping language to code in programmatic context},
  author={Iyer, Srinivasan and Konstas, Ioannis and Cheung, Alvin and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:1808.09588},
  year={2018}
}
"""


class Concode(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "code_x_glue_tc_text_to_code"

    def __init__(self, max_order=4, smooth=True):
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )
        self.max_order = max_order
        self.smooth = smooth

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        # test split of the dataset doesn't have targets
        return self.dataset["validation"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        with open(
            "bigcode_eval/tasks/few_shot_examples/concode_few_shot_prompts.json", "r"
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
        text = doc["nl"].split("concode_field_sep")[0].strip()
        if text.endswith("."):
            text = text[:-1].strip()
        entry = "Answer the following instructions in a one line of Java code:\n"
        prompt = self.two_shot_prompt(entry, text, examples)
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["code"]

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
