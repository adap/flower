# This python file is adapted from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/__init__.py

import inspect
from pprint import pprint

from . import humaneval, mbpp, multiple

TASK_REGISTRY = {
    **multiple.create_all_tasks(),
    **humaneval.create_all_tasks(),
    "mbpp": mbpp.MBPP,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        kwargs = {}
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
