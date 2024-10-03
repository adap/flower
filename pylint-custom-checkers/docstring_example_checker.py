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
"""DocString Example Checker"""


from pylint.checkers import BaseChecker, IAstroidChecker
from astroid import nodes


class DocstringExampleChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "docstring-example"
    msgs = {
        "C0001": (
            'Docstring should contain at least one example using ">>>".',
            "docstring-example-missing",
            'Ensure that each docstring contains an "Examples:" section with at least one code example.',
        ),
    }

    def visit_functiondef(self, node: nodes.FunctionDef) -> None:
        self.check_docstring(node)

    def visit_classdef(self, node: nodes.ClassDef) -> None:
        self.check_docstring(node)

    def check_docstring(self, node: nodes.NodeNG) -> None:
        docstring = node.doc
        if docstring:
            # Check if the docstring contains an "Examples:" section
            examples_section = docstring.split("Examples:")
            if len(examples_section) < 2:
                self.add_message("docstring-example-missing", node=node)
            else:
                # Count occurrences of the '>>>' code block in the Examples section
                example_count = examples_section[1].count(">>>")
                if example_count < 1:
                    self.add_message("docstring-example-missing", node=node)


def register(linter):
    linter.register_checker(DocstringExampleChecker(linter))
