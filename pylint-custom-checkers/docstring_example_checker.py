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
