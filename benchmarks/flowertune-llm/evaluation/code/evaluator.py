# This python file is adapted from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/evaluator.py

import inspect
import json
import os
import warnings
from typing import List

from generation import parallel_generations
from tasks import get_task


class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

    def generate_text(self, task_name):
        task = get_task(task_name)
        dataset = task.get_dataset()
        n_tasks = len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]

        generations = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
        )

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def evaluate(self, task_name):
        task = get_task(task_name)
        generations, references = self.generate_text(task_name)

        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}.json"
                self.save_json_files(
                    generations,
                    references,
                    save_generations_path,
                    f"references_{task_name}.json",
                )

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            results = task.process_results(generations, references)
            return results

    def save_json_files(
        self,
        generations: List[str],
        references: List[str],
        save_generations_path: str,
        save_references_path: str,
    ) -> None:
        if self.args.save_generations:
            with open(save_generations_path, "w") as fp:
                json.dump(generations, fp)
                print(f"generations were saved at {save_generations_path}")
        if self.args.save_references:
            with open(save_references_path, "w") as fp:
                json.dump(references, fp)
                print(f"references were saved at {save_references_path}")
