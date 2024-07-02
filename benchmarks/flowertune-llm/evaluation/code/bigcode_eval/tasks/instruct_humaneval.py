"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

from bigcode_eval.base import Task
from bigcode_eval.utils import remove_after_return
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = ""


def create_all_tasks():
    """Creates a dictionary of tasks corresponding for the 2 settings currently available
    - instruction with code completion: we provide function signature/imports.. to the model after the instruction
    - instruction to code generation: we only give the instruction without the function signature/imports..
    """
    return {
        "instruct-humaneval": InstructHumanEvalWithContext,
        "instruct-humaneval-nocontext": InstructHumanEvalWithoutContext,
    }


class InstructHumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "codeparrot/instructhumaneval"

    DATASET_NAME = None

    def __init__(self):
        super().__init__(
            stop_words=["if __name__", "\nprint", "\nclass"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        pass

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point


    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
        )
        return results


class InstructHumanEvalWithContext(InstructHumanEval):
    def __init__(self):
        super().__init__()

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return {"instruction": doc["instruction"], "context": doc["context"]}

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        generation = self._stop_at_stop_token(generation, self.stop_words)

        function_name = self.get_dataset()["entry_point"][idx]
        func_index = generation.find(f"def {function_name}")
        return generation[0:func_index] + remove_after_return(generation[func_index:])


class InstructHumanEvalWithoutContext(InstructHumanEval):
    def __init__(self):
        super().__init__()

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return {"instruction": doc["instruction"], "context": ""}

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        example = self.get_dataset()[idx]
        prompt, function_name = example["context"], example["entry_point"]
        prefix = prompt[0 : prompt.find(f"def {function_name}")]

        sep_index = generation.find("```")
        if sep_index == -1:
            pass
        else:
            if (
                generation[sep_index + len("```") : sep_index + len("```python")]
                == "python"
            ):
                generation = generation[sep_index + len("```python") :]
            else:
                generation = generation[sep_index + len("```") :]

        generation = self._stop_at_stop_token(generation, self.stop_words)

        func_index = generation.find(f"def {function_name}")
        if func_index == -1:
            func_index = 0
        return_index = generation[func_index:].rfind("  return ")
        if return_index == -1:
            return_index = 0

        j = func_index + return_index
        n = len(generation)

        while j < n and generation[j] != "\n":
            j += 1

        sep_index_2 = generation.find("```")
        if sep_index_2 == -1:
            return prefix.strip() + "\n" + generation[func_index:j]
        else:
            return prefix.strip() + "\n" + generation[func_index : min(j, sep_index_2)]
