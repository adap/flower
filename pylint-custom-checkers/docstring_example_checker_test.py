# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""DocstringExampleChecker tests."""

import unittest
from astroid import nodes
from pylint.checkers import BaseChecker
from .docstring_example_checker import DocstringExampleChecker, register


class TestDocstringExampleChecker(unittest.TestCase):

    def setUp(self):
        """Set up the DocstringExampleChecker instance for each test."""
        self.checker = DocstringExampleChecker(None)
        register(self.checker)

    def test_function_with_examples(self):
        node = nodes.FunctionDef(
            name="example_function",
            doc="""\
        This function does something.

        Examples:
            >>> print("Hello, World!")
            Hello, World!
        """,
        )
        self.checker.visit_functiondef(node)
        self.assertFalse(self.checker.messages)  # No messages should be added

    def test_function_without_examples(self):
        node = nodes.FunctionDef(
            name="example_function",
            doc="""\
        This function does something.
        """,
        )
        self.checker.visit_functiondef(node)
        self.assertTrue(
            any(
                msg.msg_id == "docstring-example-missing"
                for msg in self.checker.messages
            )
        )

    def test_function_with_no_code_example(self):
        node = nodes.FunctionDef(
            name="example_function",
            doc="""\
        This function does something.

        Examples:
            Not a code example.
        """,
        )
        self.checker.visit_functiondef(node)
        self.assertTrue(
            any(
                msg.msg_id == "docstring-example-missing"
                for msg in self.checker.messages
            )
        )

    def test_class_with_examples(self):
        node = nodes.ClassDef(
            name="ExampleClass",
            doc="""\
        This class is an example.

        Examples:
            >>> ExampleClass()
        """,
        )
        self.checker.visit_classdef(node)
        self.assertFalse(self.checker.messages)  # No messages should be added

    def test_class_without_examples(self):
        node = nodes.ClassDef(
            name="NoExamplesClass",
            doc="""\
        This class does something.
        """,
        )
        self.checker.visit_classdef(node)
        self.assertTrue(
            any(
                msg.msg_id == "docstring-example-missing"
                for msg in self.checker.messages
            )
        )


if __name__ == "__main__":
    unittest.main()
