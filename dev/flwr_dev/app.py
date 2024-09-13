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
"""Flower command line interface."""

from typing import Annotated
import typer
from typer.main import get_command
from flwr_dev.check_pr_title import check_title
from flwr_dev.build_example_docs import build_examples
from flwr_dev.build_docker_image_matrix import build_images
from flwr_dev.check_copyright import check_copyrights
from flwr_dev.init_py_check import check_init
from flwr_dev.init_py_fix import fix_init
from flwr_dev.protoc import compile_protos
from flwr_dev.update_changelog import generate_changelog

cli = typer.Typer(
    help=typer.style(
        "flwr is the Flower command line interface.",
        fg=typer.colors.BRIGHT_YELLOW,
        bold=True,
    ),
    no_args_is_help=True,
)


def check_title_app(
    title: Annotated[str, typer.Argument(help="Title of the PR to check")]
):
    """Check validity of a PR title."""
    check_title(title)


cli.command()(check_title_app)
cli.command()(build_examples)
cli.command()(build_images)
cli.command()(check_copyrights)
cli.command()(check_init)
cli.command()(fix_init)
cli.command()(compile_protos)
cli.command()(generate_changelog)

typer_click_object = get_command(cli)

if __name__ == "__main__":
    cli()
