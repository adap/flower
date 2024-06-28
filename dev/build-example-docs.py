import os
import shutil
import re
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX = os.path.join(ROOT, "examples", "doc", "source", "index.rst")

initial_text = """
Flower Examples Documentation
-----------------------------

Welcome to Flower Examples' documentation. `Flower <https://flower.ai>`_ is
a friendly federated learning framework.

Join the Flower Community
-------------------------

The Flower Community is growing quickly - we're a friendly group of researchers,
engineers, students, professionals, academics, and other enthusiasts.

.. button-link:: https://flower.ai/join-slack
    :color: primary
    :shadow:

    Join us on Slack

Quickstart Examples
-------------------

Flower Quickstart Examples are a collection of demo project that show how you
can use Flower in combination with other existing frameworks or technologies.

"""

table_headers = "\n.. list-table::\n   :widths: 50 15 15 15\n   :header-rows: 1\n\n   * - Title\n     - Framework\n     - Dataset\n     - Tags\n\n"

categories = {
    "quickstart": {"table": table_headers, "list": ""},
    "advanced": {"table": table_headers, "list": ""},
    "other": {"table": table_headers, "list": ""},
}


def _convert_to_link(search_result):
    if "|" in search_result:
        name, url = search_result.replace('"', "").split("|")
        return f"`{name.strip()} <{url.strip()}>`_"

    return search_result


def _read_metadata(example):
    with open(os.path.join(example, "README.md")) as f:
        content = f.read()
    metadata = re.search(r"^---(.*?)^---", content, re.DOTALL | re.MULTILINE).group(1)
    title = re.search(r"^title:\s*(.+)$", metadata, re.MULTILINE).group(1).strip()
    labels = (
        re.search(r"^labels:\s*\[(.+?)\]$", metadata, re.MULTILINE).group(1).strip()
    )
    dataset = _convert_to_link(
        re.search(r"^dataset:\s*\[(.+?)\]$", metadata, re.MULTILINE).group(1).strip()
    )
    framework = _convert_to_link(
        re.search(r"^framework:\s*\[(.+?)\]$", metadata, re.MULTILINE).group(1).strip()
    )
    return title, labels, dataset, framework


def _add_table_entry(example, label, table_var):
    title, labels, dataset, framework = _read_metadata(example)
    example_name = Path(example).stem
    table_entry = f"   * - `{title} <{example_name}.html>`_ \n     - {framework} \n     - {dataset} \n     - {labels}\n\n"
    if label in labels:
        categories[table_var]["table"] += table_entry
        categories[table_var]["list"] += f"  {example_name}\n"
        return True
    return False


def _copy_markdown_files(example):
    for file in os.listdir(example):
        if file.endswith(".md"):
            src = os.path.join(example, file)
            dest = os.path.join(
                ROOT, "examples", "doc", "source", os.path.basename(example) + ".md"
            )
            shutil.copyfile(src, dest)


def _add_gh_button(example):
    gh_text = f'[<img src="_static/view-gh.png" alt="View on GitHub" width="200"/>](https://github.com/adap/flower/blob/main/examples/{example})'
    readme_file = os.path.join(ROOT, "examples", "doc", "source", example + ".md")
    with open(readme_file, "r+") as f:
        content = f.read()
        if gh_text not in content:
            content = re.sub(
                r"(^# .+$)", rf"\1\n\n{gh_text}", content, count=1, flags=re.MULTILINE
            )
            f.seek(0)
            f.write(content)
            f.truncate()


def _copy_images(example):
    static_dir = os.path.join(example, "_static")
    dest_dir = os.path.join(ROOT, "examples", "doc", "source", "_static")
    if os.path.isdir(static_dir):
        for file in os.listdir(static_dir):
            if file.endswith((".jpg", ".png", ".jpeg")):
                shutil.copyfile(
                    os.path.join(static_dir, file), os.path.join(dest_dir, file)
                )


def _add_all_entries(index_file):
    examples_dir = os.path.join(ROOT, "examples")
    for example in sorted(os.listdir(examples_dir)):
        example_path = os.path.join(examples_dir, example)
        if os.path.isdir(example_path) and example != "doc":
            _copy_markdown_files(example_path)
            _add_gh_button(example)
            _copy_images(example)


def _main():
    if os.path.exists(INDEX):
        os.remove(INDEX)

    with open(INDEX, "w") as index_file:
        index_file.write(initial_text)

    examples_dir = os.path.join(ROOT, "examples")
    for example in sorted(os.listdir(examples_dir)):
        example_path = os.path.join(examples_dir, example)
        if os.path.isdir(example_path) and example != "doc":
            _copy_markdown_files(example_path)
            _add_gh_button(example)
            _copy_images(example_path)
            if not _add_table_entry(example_path, "quickstart", "quickstart"):
                if not _add_table_entry(example_path, "comprehensive", "comprehensive"):
                    if not _add_table_entry(example_path, "advanced", "advanced"):
                        _add_table_entry(example_path, "", "other")

    with open(INDEX, "a") as index_file:
        index_file.write(categories["quickstart"]["table"])

        index_file.write("\nAdvanced Examples\n-----------------\n")
        index_file.write(
            "Advanced Examples are mostly for users that are both familiar with "
            "Federated Learning but also somewhat familiar with Flower's main "
            "features.\n"
        )
        index_file.write(categories["advanced"]["table"])

        index_file.write("\nOther Examples\n--------------\n")
        index_file.write(
            "Flower Examples are a collection of example projects written with "
            "Flower that explore different domains and features. You can check "
            "which examples already exist and/or contribute your own example.\n"
        )
        index_file.write(categories["other"]["table"])

        _add_all_entries(index_file)

        index_file.write(
            "\n.. toctree::\n  :maxdepth: 1\n  :caption: Quickstart\n  :hidden:\n\n"
        )
        index_file.write(categories["quickstart"]["list"])

        index_file.write(
            "\n.. toctree::\n  :maxdepth: 1\n  :caption: Advanced\n  :hidden:\n\n"
        )
        index_file.write(categories["advanced"]["list"])

        index_file.write(
            "\n.. toctree::\n  :maxdepth: 1\n  :caption: Others\n  :hidden:\n\n"
        )
        index_file.write(categories["other"]["list"])

        index_file.write("\n")

    print(f"Done! Example index written to {INDEX}")


if __name__ == "__main__":
    _main()
