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
"""Update HTML themes in Flower docs."""


import re
from pathlib import Path

# Change this if you want to search a different directory.
ROOT_DIR = Path(".")

# New fields to be added to html_theme_options.
# If empty, no changes will be made.
NEW_FIELDS = """
    "announcement": "Flower AI Summit 2025, March 26-27 "
    "(🇬🇧 London & Online) <br />"
    "<a href='https://flower.ai/events/flower-ai-summit-2025/'>"
    "<strong style='color: #f2b705;'>👉 Register Now!</strong></a>",
    "light_css_variables": {
        "color-announcement-background": "#292f36",
        "color-announcement-text": "#ffffff"
    },
    "dark_css_variables": {
        "color-announcement-background": "#292f36",
        "color-announcement-text": "#ffffff"
    },
"""


def find_conf_files(root_dir: Path):
    """Recursively find all conf.py files under root_dir."""
    return list(root_dir.rglob("conf.py"))


def update_conf_file(file_path: Path, new_fields: str):
    """Insert new_fields into the html_theme_options block of file_path."""
    if not new_fields.strip():
        print(f"Skipping {file_path} (NEW_FIELDS is empty)")
        return

    content = file_path.read_text(encoding="utf-8").splitlines()
    updated_content = []
    inside_options = False
    modified = False

    for line in content:
        updated_content.append(line)
        # Check for the start of the html_theme_options block.
        if re.match(r"^\s*html_theme_options\s*=\s*{", line):
            inside_options = True
        # When encountering the closing brace, insert new_fields before it.
        if inside_options and re.match(r"^\s*}", line):
            updated_content.insert(-1, new_fields.rstrip() + "\n")
            inside_options = False
            modified = True

    if modified:
        file_path.write_text("\n".join(updated_content) + "\n", encoding="utf-8")
        print(f"Updated: {file_path}")
    else:
        print(f"No html_theme_options block found in: {file_path}")


def main():
    conf_files = find_conf_files(ROOT_DIR)
    if not conf_files:
        print("No conf.py files found.")
        return

    for conf_file in conf_files:
        update_conf_file(conf_file, NEW_FIELDS)


if __name__ == "__main__":
    main()
