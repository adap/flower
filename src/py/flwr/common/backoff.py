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
"""`SafeInvoker` to argument other functions with error handling and retries."""


import itertools
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple, Union


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


def random_jitter(value: float) -> float:
    """Jitter the value by a random number of milliseconds.

    This adds up to 1 second of additional time to the original value.
    Prior to backoff version 1.2, this was the default jitter behavior.

    Parameters
    ----------
    value : float
        The original, unjittered backoff value.
    """
    return value + random.random()


def full_jitter(value: float) -> float:
    """Jitter the value across the full range (0 to value).

    This corresponds to the "Full Jitter" algorithm specified in the
    AWS blog post on the performance of various jitter algorithms.
    See: http://www.awsarchitectureblog.com/2015/03/backoff.html

    Parameters
    ----------
    value : float
        The original, unjittered backoff value.
    """
    return random.uniform(0, value)


@dataclass
class Details:
    """Details for event handlers in SafeInvoker."""

    func: Callable[[Any], Any]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    tries: int
    elapsed: float
    exception: Optional[Exception] = None
    wait: Optional[float] = None


# pylint: disable-next=too-many-instance-attributes
class SafeInvoker:
    """Wrapper class for backoff and retry triggered by exceptions.

    Parameters
    ----------
    wait: Generator[float, None, None]
        A generator yielding successive wait times in seconds. If the generator
        is finite, the giveup event will be triggered when the generator raises
        `StopIteration`.
    exception: Union[Exception, Iterable[Exception]]
        An exception type (or iterable of types) that triggers backoff.
    max_tries: Optional[int] (default: None)
        The maximum number of attempts to make before giving up. Once exhausted,
        the exception will be allowed to escape. If set to None, there is no limit
        to the number of tries.
    max_time: Optional[float] (default: None)
        The maximum total amount of time to try before giving up. Once this time
        has expired, the exception will be allowed to escape.
    jitter: Optional[Callable[[float], float]] (default: full_jitter)
        A function of the value yielded by `wait_gen` returning the actual time
        to wait. This function helps distribute wait times stochastically to avoid
        timing collisions across concurrent clients. Wait times are jittered by
        default using the `full_jitter` function. To disable jittering, pass
        `jitter=None`.
    giveup_condition: Optional[Callable[[Exception], bool]] (default: None)
        A function accepting an exception instance, returning whether or not
        to give up. If set to None, the strategy is to always continue.
    on_success: Optional[Callable[[Details], None]] (default: None)
        A callable to be executed in the event of success. The parameter is a
        data class object detailing the invocation.
    on_backoff: Optional[Callable[[Details], None]] (default: None)
        A callable to be executed in the event of a backoff. The parameter is a
        data class object detailing the invocation.
    on_giveup: Optional[Callable[[Details], None]] (default: None)
        A callable to be executed in the event that `max_tries` or `max_time` is
        exceeded. The parameter is a data class object detailing the invocation.
    """

    def __init__(
        self,
        wait: Generator[float, None, None],
        exception: Union[Exception, Iterable[Exception]],
        *,
        max_tries: Optional[int] = None,
        max_time: Optional[float] = None,
        jitter: Optional[Callable[[float], float]] = full_jitter,
        giveup_condition: Optional[Callable[[Exception], bool]] = None,
        on_success: Optional[Callable[[Details], None]] = None,
        on_backoff: Optional[Callable[[Details], None]] = None,
        on_giveup: Optional[Callable[[Details], None]] = None,
    ) -> None:
        self.wait = wait
        self.exception = exception
        self.max_tries = max_tries
        self.max_time = max_time
        self.jitter = jitter
        self.giveup_condition = giveup_condition
        self.on_success = on_success
        self.on_backoff = on_backoff
        self.on_giveup = on_giveup

    def invoke(
        self,
        func: Callable[[Any], Any],
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ) -> Any:
        """Safely invoke the provided function with retry mechanisms.

        This method attempts to call the given function, and in the event of
        a specified exception, employs a retry mechanism that considers
        wait times, jitter, maximum attempts, and maximum time. During the
        retry process, various callbacks (`on_backoff`, `on_success`, and
        `on_giveup`) can be triggered based on the outcome.

        Parameters
        ----------
        func: Callable[[Any], Any]
            The function to be invoked.
        *args: Tuple[Any, ...]
            Positional arguments to pass to `func`.
        **kwargs: Dict[str, Any]
            Keyword arguments to pass to `func`.

        Returns
        -------
        Any
            The result of the function invocation.

        Raises
        ------
        Exception
            If the number of tries exceeds `max_tries`, if the total time
            exceeds `max_time`, if `wait` generator raises `StopInteration`,
            or if the `giveup_condition` returns True for a raised exception.

        Notes
        -----
        The time between retries is determined by the provided `wait` generator
        and can optionally be jittered using the `jitter` function. The exact
        exceptions that trigger a retry, as well as conditions to stop retries,
        are also determined by the class's initialization parameters.
        """

        def try_call_event_handler(
            handler: Optional[Callable[[Details], None]], details: Details
        ) -> None:
            if handler is not None:
                handler(details)

        tries = 0
        start = datetime.now()

        while True:
            tries += 1
            elapsed = timedelta.total_seconds(datetime.now() - start)
            details = Details(
                func=func, args=args, kwargs=kwargs, tries=tries, elapsed=elapsed
            )

            try:
                ret = func(*args, **kwargs)
            except self.exception as err:  # type: ignore
                # Check if giveup event should be triggered
                max_tries_exceeded = tries == self.max_tries
                max_time_exceeded = (
                    self.max_time is not None and elapsed >= self.max_time
                )

                def giveup_check(_exception: Exception) -> bool:
                    if self.giveup_condition is None:
                        return False
                    return self.giveup_condition(_exception)

                if giveup_check(err) or max_tries_exceeded or max_time_exceeded:
                    # Trigger giveup event
                    try_call_event_handler(self.on_giveup, details)
                    raise

                try:
                    seconds = next(self.wait)
                    if self.jitter is not None:
                        seconds = self.jitter(seconds)
                    if self.max_time is not None:
                        seconds = min(seconds, self.max_time - elapsed)
                    details.wait = seconds
                except StopIteration:
                    # Trigger giveup event
                    try_call_event_handler(self.on_giveup, details)
                    raise err from None

                # Trigger backoff event
                try_call_event_handler(self.on_backoff, details)

                # Sleep
                time.sleep(seconds)
            else:
                # Trigger success event
                try_call_event_handler(self.on_success, details)
                return ret
