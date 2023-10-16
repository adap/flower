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
"""Functions to enchance other functions with error handling and retries."""


import random
from typing import Generator, Iterable, Union, Optional, Callable, Any
import itertools


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
    delay = min(base_delay, max_delay)
    while True:
        yield delay
        delay *= multiplier
        if max_delay is not None:
            delay = min(delay, max_delay)


def interval(
    interval: Union[float, Iterable[float]] = 1,
) -> Generator[float, None, None]:
    """Wait time generator for specified intervals.

    Parameters
    ----------
    interval: Union[float, Iterable[float]] (default: 1)
        A constant value to yield or an iterable of such values.
    """
    try:
        iter(interval)
    except TypeError:
        interval = itertools.repeat(interval)
    for wait_time in interval:
        yield wait_time
        
        
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


class SafeInvoker


def safe_invoke(
    wait_gen: Generator[float, None, None],
    exception: Union[Exception, Iterable[Exception]],
    func: Callable[[Any], Any]
    *,
    max_tries: Optional[int] = None,
    max_time: Optional[float] = None,
    jitter: Optional[Callable[[float], float]] = full_jitter,
    giveup_condition: Callable[[Exception], bool] = lambda e: False,
    on_success: Optional[Callable[Exception]] = None,
    on_backoff: Union[_Handler, Iterable[_Handler], None] = None,
    on_giveup: Union[_Handler, Iterable[_Handler], None] = None,
    logger: _MaybeLogger = 'backoff',
    backoff_log_level: int = logging.INFO,
    giveup_log_level: int = logging.ERROR,
) -> Callable[[_CallableT], _CallableT]:
    """Returns decorator for backoff and retry triggered by exception.

    Args:
        wait_gen: A generator yielding successive wait times in
            seconds.
        exception: An exception type (or tuple of types) which triggers
            backoff.
        max_tries: The maximum number of attempts to make before giving
            up. Once exhausted, the exception will be allowed to escape.
            The default value of None means there is no limit to the
            number of tries. If a callable is passed, it will be
            evaluated at runtime and its return value used.
        max_time: The maximum total amount of time to try for before
            giving up. Once expired, the exception will be allowed to
            escape. If a callable is passed, it will be
            evaluated at runtime and its return value used.
        jitter: A function of the value yielded by wait_gen returning
            the actual time to wait. This distributes wait times
            stochastically in order to avoid timing collisions across
            concurrent clients. Wait times are jittered by default
            using the full_jitter function. Jittering may be disabled
            altogether by passing jitter=None.
        giveup: Function accepting an exception instance and
            returning whether or not to give up. Optional. The default
            is to always continue.
        on_success: Callable (or iterable of callables) with a unary
            signature to be called in the event of success. The
            parameter is a dict containing details about the invocation.
        on_backoff: Callable (or iterable of callables) with a unary
            signature to be called in the event of a backoff. The
            parameter is a dict containing details about the invocation.
        on_giveup: Callable (or iterable of callables) with a unary
            signature to be called in the event that max_tries
            is exceeded.  The parameter is a dict containing details
            about the invocation.
        raise_on_giveup: Boolean indicating whether the registered exceptions
            should be raised on giveup. Defaults to `True`
        logger: Name or Logger object to log to. Defaults to 'backoff'.
        backoff_log_level: log level for the backoff event. Defaults to "INFO"
        giveup_log_level: log level for the give up event. Defaults to "ERROR"
        **wait_gen_kwargs: Any additional keyword args specified will be
            passed to wait_gen when it is initialized.  Any callable
            args will first be evaluated and their return values passed.
            This is useful for runtime configuration.
    """
    ...
