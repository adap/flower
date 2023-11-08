"""This is a temporary file showing how middlware works."""


from flwr.proto.task_pb2 import TaskIns, TaskRes

from flwr.client.middleware.typing import App
from flwr.client.middleware.utils import make_app


def user_middleware1(task_ins: TaskIns, app: App) -> TaskRes:
    # Do something before passing task_ins to app.
    print("User middleware 1 before app")

    task_res = app(task_ins)

    # Do something before returning task_res.
    print("User middleware 1 after app")

    return task_res


def user_middleware2(task_ins: TaskIns, app: App) -> TaskRes:
    # Do something before passing task_ins to app.
    print("User middleware 2 before app")

    # Directly returning a TaskRes without calling app if some condition is true.
    some_condition = False
    if some_condition:
        # Some task_res
        return TaskRes()

    task_res = app(task_ins)

    # Do something before returning task_res.
    print("User middleware 2 after app")

    return task_res


def handle(task_ins: TaskIns) -> TaskRes:
    """Example message handler."""
    print(f"Received {task_ins}")
    return TaskRes(task_id="def")


# We can add an argument, such as `middleware`, so that users can
# write something like:
# app = fl.app.Flower(
#     client_fn=client_fn,
#     middleware=[user_middleware1, user_middleware2]
# )

# Just to disambiguate, the `app` word I use in `make_app` and `wrapped_app` 
# refers to the message handler of type `Callable[[TaskIns], TaskRes]`. 
# All the names are open to change.

# And within our framework we can use `make_app()` to
# wrap middleware layers around app. For example
wrapped_app = make_app(handle, [user_middleware1, user_middleware2])

# All middlewares and app will be called by order, like an onion.
# Setting the `task_id` here is only for demonstration purposes,
# and the field will actually be set by state.
task_res = wrapped_app(TaskIns(task_id="abc"))
print(f"Responded {task_res}")
