# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`RetryInvoker` to augment other callables with error handling and retries."""


import itertools
import random
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)


def exponential(
    base_delay: float = 1,
    multiplier: float = 2,
    max_delay: Optional[int] = None,
) -> Generator[float, None, None]:
    """Wait time generator for exponential backoff strategy.

    Parameters
    ----------
    base_delay: float (default: 1)
        Initial delay duration before the first retry.
    multiplier: float (default: 2)
        Factor by which the delay is multiplied after each retry.
    max_delay: Optional[float] (default: None)
        The maximum delay duration between two consecutive retries.
    """
    delay = base_delay if max_delay is None else min(base_delay, max_delay)
    while True:
        yield delay
        delay *= multiplier
        if max_delay is not None:
            delay = min(delay, max_delay)


def constant(
    interval: Union[float, Iterable[float]] = 1,
) -> Generator[float, None, None]:
    """Wait time generator for specified intervals.

    Parameters
    ----------
    interval: Union[float, Iterable[float]] (default: 1)
        A constant value to yield or an iterable of such values.
    """
    if not isinstance(interval, Iterable):
        interval = itertools.repeat(interval)
    yield from interval


def full_jitter(max_value: float) -> float:
    """Randomize a float between 0 and the given maximum value.

    This function implements the "Full Jitter" algorithm as described in the
    AWS article discussing the efficacy of different jitter algorithms.
    Reference: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

    Parameters
    ----------
    max_value : float
        The upper limit for the randomized value.
    """
    return random.uniform(0, max_value)


@dataclass
class RetryState:
    """State for callbacks in RetryInvoker."""

    target: Callable[..., Any]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    tries: int
    elapsed_time: float
    exception: Optional[Exception] = None
    actual_wait: Optional[float] = None


# pylint: disable-next=too-many-instance-attributes
class RetryInvoker:
    """Wrapper class for retry (with backoff) triggered by exceptions.

    Parameters
    ----------
    wait_factory: Callable[[], Generator[float, None, None]]
        A generator yielding successive wait times in seconds. If the generator
        is finite, the giveup event will be triggered when the generator raises
        `StopIteration`.
    recoverable_exceptions: Union[Type[Exception], Tuple[Type[Exception]]]
        An exception type (or tuple of types) that triggers backoff.
    max_tries: Optional[int]
        The maximum number of attempts to make before giving up. Once exhausted,
        the exception will be allowed to escape. If set to None, there is no limit
        to the number of tries.
    max_time: Optional[float]
        The maximum total amount of time to try before giving up. Once this time
        has expired, this method won't be interrupted immediately, but the exception
        will be allowed to escape. If set to None, there is no limit to the total time.
    on_success: Optional[Callable[[RetryState], None]] (default: None)
        A callable to be executed in the event of success. The parameter is a
        data class object detailing the invocation.
    on_backoff: Optional[Callable[[RetryState], None]] (default: None)
        A callable to be executed in the event of a backoff. The parameter is a
        data class object detailing the invocation.
    on_giveup: Optional[Callable[[RetryState], None]] (default: None)
        A callable to be executed in the event that `max_tries` or `max_time` is
        exceeded, `should_giveup` returns True, or `wait_factory()` generator raises
        `StopInteration`. The parameter is a data class object detailing the
        invocation.
    jitter: Optional[Callable[[float], float]] (default: full_jitter)
        A function of the value yielded by `wait_factory()` returning the actual time
        to wait. This function helps distribute wait times stochastically to avoid
        timing collisions across concurrent clients. Wait times are jittered by
        default using the `full_jitter` function. To disable jittering, pass
        `jitter=None`.
    should_giveup: Optional[Callable[[Exception], bool]] (default: None)
        A function accepting an exception instance, returning whether or not
        to give up prematurely before other give-up conditions are evaluated.
        If set to None, the strategy is to never give up prematurely.

    Examples
    --------
    Initialize a `RetryInvoker` with exponential backoff and invoke a function:

    >>> invoker = RetryInvoker(
    ...     exponential,  # Or use `lambda: exponential(3, 2)` to pass arguments
    ...     grpc.RpcError,
    ...     max_tries=3,
    ...     max_time=None,
    ... )
    >>> invoker.invoke(my_func, arg1, arg2, kw1=kwarg1)
    """

    def __init__(
        self,
        wait_factory: Callable[[], Generator[float, None, None]],
        recoverable_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]],
        max_tries: Optional[int],
        max_time: Optional[float],
        *,
        on_success: Optional[Callable[[RetryState], None]] = None,
        on_backoff: Optional[Callable[[RetryState], None]] = None,
        on_giveup: Optional[Callable[[RetryState], None]] = None,
        jitter: Optional[Callable[[float], float]] = full_jitter,
        should_giveup: Optional[Callable[[Exception], bool]] = None,
    ) -> None:
        self.wait_factory = wait_factory
        self.recoverable_exceptions = recoverable_exceptions
        self.max_tries = max_tries
        self.max_time = max_time
        self.on_success = on_success
        self.on_backoff = on_backoff
        self.on_giveup = on_giveup
        self.jitter = jitter
        self.should_giveup = should_giveup

    # pylint: disable-next=too-many-locals
    def invoke(
        self,
        target: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Safely invoke the provided callable with retry mechanisms.

        This method attempts to invoke the given callable, and in the event of
        a recoverable exception, employs a retry mechanism that considers
        wait times, jitter, maximum attempts, and maximum time. During the
        retry process, various callbacks (`on_backoff`, `on_success`, and
        `on_giveup`) can be triggered based on the outcome.

        Parameters
        ----------
        target: Callable[..., Any]
            The callable to be invoked.
        *args: Tuple[Any, ...]
            Positional arguments to pass to `target`.
        **kwargs: Dict[str, Any]
            Keyword arguments to pass to `target`.

        Returns
        -------
        Any
            The result of the given callable invocation.

        Raises
        ------
        Exception
            If the number of tries exceeds `max_tries`, if the total time
            exceeds `max_time`, if `wait_factory()` generator raises `StopInteration`,
            or if the `should_giveup` returns True for a raised exception.

        Notes
        -----
        The time between retries is determined by the provided `wait_factory()`
        generator and can optionally be jittered using the `jitter` function.
        The recoverable exceptions that trigger a retry, as well as conditions to
        stop retries, are also determined by the class's initialization parameters.
        """

        def try_call_event_handler(
            handler: Optional[Callable[[RetryState], None]]
        ) -> None:
            if handler is not None:
                handler(cast(RetryState, ref_state[0]))

        try_cnt = 0
        wait_generator = self.wait_factory()
        start = time.time()
        ref_state: List[Optional[RetryState]] = [None]

        while True:
            try_cnt += 1
            elapsed_time = time.time() - start
            state = RetryState(
                target=target,
                args=args,
                kwargs=kwargs,
                tries=try_cnt,
                elapsed_time=elapsed_time,
            )
            ref_state[0] = state

            try:
                ret = target(*args, **kwargs)
            except self.recoverable_exceptions as err:
                # Check if giveup event should be triggered
                max_tries_exceeded = try_cnt == self.max_tries
                max_time_exceeded = (
                    self.max_time is not None and elapsed_time >= self.max_time
                )

                def giveup_check(_exception: Exception) -> bool:
                    if self.should_giveup is None:
                        return False
                    return self.should_giveup(_exception)

                if giveup_check(err) or max_tries_exceeded or max_time_exceeded:
                    # Trigger giveup event
                    try_call_event_handler(self.on_giveup)
                    raise

                try:
                    wait_time = next(wait_generator)
                    if self.jitter is not None:
                        wait_time = self.jitter(wait_time)
                    if self.max_time is not None:
                        wait_time = min(wait_time, self.max_time - elapsed_time)
                    state.actual_wait = wait_time
                except StopIteration:
                    # Trigger giveup event
                    try_call_event_handler(self.on_giveup)
                    raise err from None

                # Trigger backoff event
                try_call_event_handler(self.on_backoff)

                # Sleep
                time.sleep(wait_time)
            else:
                # Trigger success event
                try_call_event_handler(self.on_success)
                return ret
