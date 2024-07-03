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
"""Build the Flower Example docs."""

import os
import shutil
import re
import subprocess
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

Flower Quickstart Examples are a collection of demo projects that show how you
can use Flower in combination with other existing frameworks or technologies.

"""

table_headers = (
    "\n.. list-table::\n   :widths: 50 15 15 15\n   "
    ":header-rows: 1\n\n   * - Title\n     - Framework\n     - Dataset\n     - Tags\n\n"
)

categories = {
    "quickstart": {"table": table_headers, "list": ""},
    "advanced": {"table": table_headers, "list": ""},
    "other": {"table": table_headers, "list": ""},
}

urls = {
    # Frameworks
    "Android": "https://www.android.com/",
    "C++": "https://isocpp.org/",
    "Docker": "https://www.docker.com/",
    "JAX": "https://jax.readthedocs.io/en/latest/",
    "Java": "https://www.java.com/",
    "Keras": "https://keras.io/",
    "Kotlin": "https://kotlinlang.org/",
    "MLX": "https://ml-explore.github.io/mlx/build/html/index.html",
    "MONAI": "https://monai.io/",
    "PEFT": "https://huggingface.co/docs/peft/index",
    "Swift": "https://www.swift.org/",
    "TensorFlowLite": "https://www.tensorflow.org/lite",
    "fastai": "https://fast.ai/",
    "lifelines": "https://lifelines.readthedocs.io/en/latest/index.html",
    "lightning": "https://lightning.ai/docs/pytorch/stable/",
    "numpy": "https://numpy.org/",
    "opacus": "https://opacus.ai/",
    "pandas": "https://pandas.pydata.org/",
    "scikit-learn": "https://scikit-learn.org/",
    "tabnet": "https://github.com/titu1994/tf-TabNet",
    "tensorboard": "https://www.tensorflow.org/tensorboard",
    "tensorflow": "https://www.tensorflow.org/",
    "torch": "https://pytorch.org/",
    "torchvision": "https://pytorch.org/vision/stable/index.html",
    "transformers": "https://huggingface.co/docs/transformers/index",
    "wandb": "https://wandb.ai/home",
    "whisper": "https://huggingface.co/openai/whisper-tiny",
    "xgboost": "https://xgboost.readthedocs.io/en/stable/",
    # Datasets
    "Adult Census Income": "https://www.kaggle.com/datasets/uciml/adult-census-income/data",
    "Alpaca-GPT4": "https://huggingface.co/datasets/vicgalle/alpaca-gpt4",
    "CIFAR-10": "https://huggingface.co/datasets/uoft-cs/cifar10",
    "HIGGS": "https://archive.ics.uci.edu/dataset/280/higgs",
    "IMDB": "https://huggingface.co/datasets/stanfordnlp/imdb",
    "Iris": "https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html",
    "MNIST": "https://huggingface.co/datasets/ylecun/mnist",
    "MedNIST": "https://medmnist.com/",
    "Oxford Flower-102": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
    "SpeechCommands": "https://huggingface.co/datasets/google/speech_commands",
    "Titanic": "https://www.kaggle.com/competitions/titanic",
}


def _convert_to_link(search_result):
    if "," in search_result:
        result = ""
        for part in search_result.split(","):
            result += f"{_convert_to_link(part)}, "
        return result[:-2]
    else:
        search_result = search_result.strip()
        name, url = search_result, urls.get(search_result, None)
        if url:
            return f"`{name.strip()} <{url.strip()}>`_"
        else:
            return search_result


def _read_metadata(example):
    with open(os.path.join(example, "README.md")) as f:
        content = f.read()

    metadata_match = re.search(r"^---(.*?)^---", content, re.DOTALL | re.MULTILINE)
    if not metadata_match:
        raise ValueError("Metadata block not found")
    metadata = metadata_match.group(1)

    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    if not title_match:
        raise ValueError("Title not found in metadata")
    title = title_match.group(1).strip()

    tags_match = re.search(r"^tags:\s*\[(.+?)\]$", metadata, re.MULTILINE)
    if not tags_match:
        raise ValueError("Tags not found in metadata")
    tags = tags_match.group(1).strip()

    dataset_match = re.search(
        r"^dataset:\s*\[(.*?)\]$", metadata, re.DOTALL | re.MULTILINE
    )
    if not dataset_match:
        raise ValueError("Dataset not found in metadata")
    dataset = dataset_match.group(1).strip()

    framework_match = re.search(
        r"^framework:\s*\[(.*?|)\]$", metadata, re.DOTALL | re.MULTILINE
    )
    if not framework_match:
        raise ValueError("Framework not found in metadata")
    framework = framework_match.group(1).strip()

    dataset = _convert_to_link(re.sub(r"\s+", " ", dataset).strip())
    framework = _convert_to_link(re.sub(r"\s+", " ", framework).strip())
    return title, tags, dataset, framework


def _add_table_entry(example, tag, table_var):
    title, tags, dataset, framework = _read_metadata(example)
    example_name = Path(example).stem
    table_entry = (
        f"   * - `{title} <{example_name}.html>`_ \n     "
        f"- {framework} \n     - {dataset} \n     - {tags}\n\n"
    )
    if tag in tags:
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


def _add_all_entries():
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

        _add_all_entries()

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


if __name__ == "__main__":
    _main()
    subprocess.call(f"cd {ROOT}/examples/doc && make html", shell=True)
