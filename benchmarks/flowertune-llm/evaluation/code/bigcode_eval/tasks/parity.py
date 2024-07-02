"""Parity bug fixing task."""

import itertools
import re

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
import tqdm

def mutate_code(
    n_bugs: int = 5, task: str = "parity", prompt="prompt"
):
    """
    Modified from https://github.com/CarperAI/OpenELM/blob/e6402a0696096011572152334ccbe049f89c332e/src/openelm/utils/code_eval.py
    
    Mutate code to create n bugs. Output the prompt in diff format.
    Args:
        n_bugs: number of bugs to introduce (from 1 to 5).
        task: (Optional) the task to be performed.
        prompt: (Optional) 'diff', 'prompt' or 'edit'.
    Returns:
        template for code mutation
    """
    mutation_templates = {
        "diff": [
            f"<NME> {task}.py\n<BEF> ",
            "",  # placeholder for the context, e.g., the buggy code
            "\n<MSG> Fixed bugs",
        ],
        "prompt_carper": [
            "# A buggy implementation\n#!/usr/bin/python3\n",
            "",  # placeholder for the context, e.g., the buggy code
            "\n# Fixed bugs\ndef",
        ],
        "prompt": [
            "#!/usr/bin/python3\n# A buggy implementation\n", # Fixed order
            "",  # placeholder for the context, e.g., the buggy code
            "\n# Fixed bugs\ndef", # Past tense is key
        ],
        "edit": [
            "<commit_before>",
            "",  # placeholder for the context, e.g., the buggy code
            "<commit_msg>Fix bugs<commit_after>",
        ],
    }
    mutation_template = mutation_templates[prompt]
    if task == "parity":
        variables = ["b", "b", "b", "b", 2]
        for i in range(n_bugs):
            variables[i] = "c" if i < 4 else 3
        func_str = (
            'def parity(b1,b2,b3,b4):\n    """Return binary parity of a sequence of input bits.'
            ' Return 0 for even parity, 1 for odd parity."""\n    bit_sum = sum(['
            "{}1,{}2,{}3,{}4])\n    return bit_sum % {}".format(*variables)
        )
        mutation_template[1] = func_str
        return "".join(mutation_template)
    else:
        raise ValueError(f"Unknown task: {task}")

# https://github.com/CarperAI/OpenELM/blob/e6402a0696096011572152334ccbe049f89c332e/src/openelm/utils/code_eval.py#L131
def parity_reference(b1, b2, b3, b4):
    """
    Return binary parity of a sequence of input bits.
    Return 0 for even parity, 1 for odd parity.
    """
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2


class Parity(Task):
    def __init__(self, prompt="prompt"):

        super().__init__(
            stop_words=[
                "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif",
                # Special cases for edit
                "<commit_before>", "<commit_msg>", "<commit_after>", "<|endoftext|>",
            ],
            requires_execution=True,
        )
        self.prompt = prompt
        self.parity_tests = "assert " + " and ".join([
            f"({parity_reference(*i)} == parity{i})" for i in itertools.product(range(2), repeat=4)
        ])
        
        # Allow max 3 times the length of the prompt to
        # allow the model to e.g. add some comments
        self.max_length_multiplier = 3

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return [1, 2, 3, 4, 5]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return mutate_code(n_bugs=doc, task="parity", prompt=self.prompt)

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return []

    @staticmethod
    def first_block(string, stop_words):
        """Split off first block of code by scanning for class, def etc. on newlines."""
        stop_words = [re.escape(word) for word in stop_words] # Escape e.g. | in <|endoftext|>
        return re.split("|".join(stop_words), string)[0].rstrip()        

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        output = generation[len(prompt):]
        if self.prompt.startswith("prompt"):
            output = "def" + output # Add def which is in the prompt back to the output
        return self.first_block(output, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        out = {}
        # Compute metrics for each number of bugs
        for idx, gens in tqdm.tqdm(enumerate(generations), total=len(generations)):
            results, _ = compute_code_eval(
                references=[self.parity_tests for _ in gens],
                predictions=[[g] for g in gens],
            )
            out[f"{idx+1} bugs"] = results
        return out
