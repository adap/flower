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
        "W9001": (
            "Docstring should contain at least 1 example.",
            "docstring-example-missing",
            "Used when a docstring does not have the required number of examples.",
        ),
    }

    def check_docstring(self, node: nodes.NodeNG) -> None:
        """Check that the docstring contains at least 1 example."""
        docstring = node.doc
        if docstring:
            examples = docstring.split("Examples:")
            if len(examples) < 2:
                self.add_message("docstring-example-missing", node=node)
            else:
                example_count = examples[1].count(">>>")
                if example_count < 1:
                    self.add_message("docstring-example-missing", node=node)

    def process_module(self, node: nodes.Module) -> None:
        """Process a module to check docstrings."""
        for child in node.get_children():
            if isinstance(child, (nodes.FunctionDef, nodes.ClassDef)):
                self.check_docstring(child)


def register(linter):
    """Register the checker to Pylint."""
    linter.register_checker(DocstringExampleChecker(linter))
