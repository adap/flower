from logging import ERROR
from typing import Callable

import lithops

from flwr.client.client_app import ClientApp
from .backend import Backend
from flwr.common.message import Message
from flwr.common.context import Context
from flwr.common.logger import log


class LithopsBackend(Backend):
    """Backend for Lithops."""

    def __init__(self, config: dict):
        self._fexec = lithops.FunctionExecutor()

    @property
    def num_workers(self) -> int:
        """Return the number of workers available in the backend."""
        return 4

    def build(self, app_fn: Callable[[], ClientApp]):
        # TODO check if you need to initialize lithops here
        self.app_fn = app_fn

    def is_worker_idle(self) -> bool:
        """Report whether a backend worker is idle and can therefore run a ClientApp."""
        return True

    def terminate(self) -> None:
        """Terminate backend."""
        pass

    def process_message(
        self,
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]:
        """Submit a job to the backend."""
        try:
            future = self._fexec.call_async(
                lambda a_fn, mssg, state: a_fn()(mssg, state),
                (self.app_fn, message, context),
            )
            out_mssg = future.result()
            return out_mssg, context
        except Exception as ex:
            log(
                ERROR,
                "An exception was raised when processing a message by %s",
                self.__class__.__name__,
            )
            raise ex
