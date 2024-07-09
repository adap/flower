import inspect
from pprint import pprint

from . import (gsm, humaneval, humanevalplus, humanevalpack,
               instruct_humaneval, instruct_wizard_humaneval, mbpp, mbppplus,
               multiple, parity, python_bugs, quixbugs, recode, santacoder_fim)

TASK_REGISTRY = {
    **multiple.create_all_tasks(),
    **humaneval.create_all_tasks(),
    **humanevalplus.create_all_tasks(),
    **humanevalpack.create_all_tasks(),
    "mbpp": mbpp.MBPP,
    "mbppplus": mbppplus.MBPPPlus,
    "parity": parity.Parity,
    "python_bugs": python_bugs.PythonBugs,
    "quixbugs": quixbugs.QuixBugs,
    "instruct_wizard_humaneval": instruct_wizard_humaneval.HumanEvalWizardCoder,
    **gsm.create_all_tasks(),
    **instruct_humaneval.create_all_tasks(),
    **recode.create_all_tasks(),
    **santacoder_fim.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["load_data_path"] = args.load_data_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
