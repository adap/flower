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
"""Helper functions to load objects from a reference."""


import ast
import importlib
from importlib.util import find_spec
from typing import Any, Optional, Tuple, Type

OBJECT_REF_HELP_STR = """
\n\nThe object reference string should have the form <module>:<attribute>. Valid
examples include `client:app` and `project.package.module:wrapper.app`. It must
refer to a module on the PYTHONPATH and the module needs to have the specified
attribute.
"""


def validate(
    module_attribute_str: str,
    check_module: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Validate object reference.

    The object reference string should have the form <module>:<attribute>. Valid
    examples include `client:app` and `project.package.module:wrapper.app`. It must
    refer to a module on the PYTHONPATH and the module needs to have the specified
    attribute.

    Returns
    -------
    Tuple[bool, Optional[str]]
        A boolean indicating whether an object reference is valid and
        the reason why it might not be.
    """
    module_str, _, attributes_str = module_attribute_str.partition(":")
    if not module_str:
        return (
            False,
            f"Missing module in {module_attribute_str}{OBJECT_REF_HELP_STR}",
        )
    if not attributes_str:
        return (
            False,
            f"Missing attribute in {module_attribute_str}{OBJECT_REF_HELP_STR}",
        )

    if check_module:
        # Load module
        module = find_spec(module_str)
        if module and module.origin:
            if not _find_attribute_in_module(module.origin, attributes_str):
                return (
                    False,
                    f"Unable to find attribute {attributes_str} in module {module_str}"
                    f"{OBJECT_REF_HELP_STR}",
                )
            return (True, None)
    else:
        return (True, None)

    return (
        False,
        f"Unable to load module {module_str}{OBJECT_REF_HELP_STR}",
    )


def load_app(
    module_attribute_str: str,
    error_type: Type[Exception],
) -> Any:
    """Return the object specified in a module attribute string.

    The module/attribute string should have the form <module>:<attribute>. Valid
    examples include `client:app` and `project.package.module:wrapper.app`. It must
    refer to a module on the PYTHONPATH, the module needs to have the specified
    attribute.
    """
    valid, error_msg = validate(module_attribute_str)
    if not valid and error_msg:
        raise error_type(error_msg) from None

    module_str, _, attributes_str = module_attribute_str.partition(":")

    try:
        module = importlib.import_module(module_str)
    except ModuleNotFoundError:
        raise error_type(
            f"Unable to load module {module_str}{OBJECT_REF_HELP_STR}",
        ) from None

    # Recursively load attribute
    attribute = module
    try:
        for attribute_str in attributes_str.split("."):
            attribute = getattr(attribute, attribute_str)
    except AttributeError:
        raise error_type(
            f"Unable to load attribute {attributes_str} from module {module_str}"
            f"{OBJECT_REF_HELP_STR}",
        ) from None

    return attribute


def _find_attribute_in_module(file_path: str, attribute_name: str) -> bool:
    """Check if attribute_name exists in module's abstract symbolic tree."""
    with open(file_path, encoding="utf-8") as file:
        node = ast.parse(file.read(), filename=file_path)

    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id == attribute_name:
                    return True
                if _is_module_in_all(attribute_name, target, n):
                    return True
    return False


def _is_module_in_all(attribute_name: str, target: ast.expr, n: ast.Assign) -> bool:
    """Now check if attribute_name is in __all__."""
    if isinstance(target, ast.Name) and target.id == "__all__":
        if isinstance(n.value, ast.List):
            for elt in n.value.elts:
                if isinstance(elt, ast.Str) and elt.s == attribute_name:
                    return True
        elif isinstance(n.value, ast.Tuple):
            for elt in n.value.elts:
                if isinstance(elt, ast.Str) and elt.s == attribute_name:
                    return True
    return False
