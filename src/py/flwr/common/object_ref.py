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
import sys
from importlib.util import find_spec
from logging import WARN
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union

from .logger import log

OBJECT_REF_HELP_STR = """
\n\nThe object reference string should have the form <module>:<attribute>. Valid
examples include `client:app` and `project.package.module:wrapper.app`. It must
refer to a module on the PYTHONPATH and the module needs to have the specified
attribute.
"""

_current_sys_path: str = ""


def validate(
    module_attribute_str: str,
    check_module: bool = True,
    project_dir: Optional[Union[str, Path]] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate object reference.

    Parameters
    ----------
    module_attribute_str : str
        The reference to the object. It should have the form `<module>:<attribute>`.
        Valid examples include `client:app` and `project.package.module:wrapper.app`.
        It must refer to a module on the PYTHONPATH or in the provided `project_dir`
        and the module needs to have the specified attribute.
    check_module : bool (default: True)
        Flag indicating whether to verify the existence of the module and the
        specified attribute within it.
    project_dir : Optional[Union[str, Path]] (default: None)
        The directory containing the module. If None, the current working directory
        is used. If `check_module` is True, the `project_dir` will be inserted into
        the system path, and the previously inserted `project_dir` will be removed.

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
        # Set the system path
        _set_sys_path(project_dir)

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


def load_app(  # pylint: disable= too-many-branches
    module_attribute_str: str,
    error_type: Type[Exception],
    project_dir: Optional[Union[str, Path]] = None,
) -> Any:
    """Return the object specified in a module attribute string.

    Parameters
    ----------
    module_attribute_str : str
        The reference to the object. It should have the form `<module>:<attribute>`.
        Valid examples include `client:app` and `project.package.module:wrapper.app`.
        It must refer to a module on the PYTHONPATH or in the provided `project_dir`
        and the module needs to have the specified attribute.
    error_type : Type[Exception]
        The type of exception to be raised if the provided `module_attribute_str` is
        in an invalid format.
    project_dir : Optional[Union[str, Path]], optional (default=None)
        The directory containing the module. If None, the current working directory
        is used. The `project_dir` will be inserted into the system path, and the
        previously inserted `project_dir` will be removed.

    Returns
    -------
    Any
        The object specified by the module attribute string.
    """
    valid, error_msg = validate(module_attribute_str, check_module=False)
    if not valid and error_msg:
        raise error_type(error_msg) from None

    module_str, _, attributes_str = module_attribute_str.partition(":")

    try:
        _set_sys_path(project_dir)

        if module_str not in sys.modules:
            module = importlib.import_module(module_str)
        # Hack: `tabnet` does not work with `importlib.reload`
        elif "tabnet" in sys.modules:
            log(
                WARN,
                "Cannot reload module `%s` from disk due to compatibility issues "
                "with the `tabnet` library. The module will be loaded from the "
                "cache instead. If you experience issues, consider restarting "
                "the application.",
                module_str,
            )
            module = sys.modules[module_str]
        else:
            module = sys.modules[module_str]

            if project_dir is None:
                project_dir = Path.cwd()

            # Reload cached modules in the project directory
            for m in list(sys.modules.values()):
                path: Optional[str] = getattr(m, "__file__", None)
                if path is not None and path.startswith(str(project_dir)):
                    importlib.reload(m)

    except ModuleNotFoundError as err:
        raise error_type(
            f"Unable to load module {module_str}{OBJECT_REF_HELP_STR}",
        ) from err

    # Recursively load attribute
    attribute = module
    try:
        for attribute_str in attributes_str.split("."):
            attribute = getattr(attribute, attribute_str)
    except AttributeError as err:
        raise error_type(
            f"Unable to load attribute {attributes_str} from module {module_str}"
            f"{OBJECT_REF_HELP_STR}",
        ) from err

    return attribute


def _set_sys_path(directory: Optional[Union[str, Path]]) -> None:
    """Set the system path."""
    global _current_sys_path  # pylint: disable=global-statement
    # Assume the current working directory `""` is in `sys.path`
    if directory is None:
        return

    directory = Path(directory).absolute()

    # If the directory has already been added to `sys.path`, return
    if str(directory) == _current_sys_path:
        return

    # Remove the old path if it exists and is not `""`.
    if _current_sys_path != "":
        sys.path.remove(_current_sys_path)

    # Add the new path to sys.path
    sys.path.insert(0, str(directory))

    # Update the current_sys_path
    _current_sys_path = str(directory)


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
