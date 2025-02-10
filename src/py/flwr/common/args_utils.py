# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower argument utilities."""


import argparse
from argparse import Action, _MutuallyExclusiveGroup
from typing import Iterable, List, Optional


class SortingHelpFormatter(argparse.HelpFormatter):
    """Sort the arguments alphabetically in the help text."""

    def add_usage(
        self,
        usage: Optional[str],
        actions: Iterable[Action],
        groups: Iterable[_MutuallyExclusiveGroup],
        prefix: Optional[str] = None,
    ) -> None:

        # def add_usage(self, usage, actions, groups, prefix=None) -> None:
        # Sort the usage actions alphabetically
        sorted_actions: List[Action] = sorted(actions, key=lambda action: action.dest)
        super().add_usage(usage, sorted_actions, groups, prefix)

    def add_arguments(self, actions: Iterable[Action]) -> None:
        # Sort the argument actions alphabetically
        sorted_actions: List[Action] = sorted(actions, key=lambda action: action.dest)
        super().add_arguments(sorted_actions)
