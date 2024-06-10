# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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


import datetime
import os
import sys
from git import Repo
from sphinx.application import ConfigError

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# Fixing path issue for autodoc
sys.path.insert(0, os.path.abspath("../../src/py"))

# -- Versioning switcher -----------------------------------------------------

html_context = dict()

# Make current language accessible for the html templates
if "current_language" in os.environ:
    current_language = os.environ["current_language"]
else:
    current_language = "en"
html_context["current_language"] = current_language

# Make current version accessible for the html templates
repo = Repo(search_parent_directories=True)
local = False
if "current_version" in os.environ:
    current_version = os.environ["current_version"]
elif os.getenv("GITHUB_ACTIONS"):
    current_version = "main"
else:
    local = True
    current_version = repo.active_branch.name

# Format current version for the html templates
html_context["current_version"] = {}
html_context["current_version"]["url"] = current_version
html_context["current_version"]["full_name"] = (
    "main"
    if current_version == "main"
    else f"{'' if local else 'Flower Framework '}{current_version}"
)

# Make version list accessible for the html templates
html_context["versions"] = list()
versions = [
    tag.name
    for tag in repo.tags
    if int(tag.name[1]) > 0 and int(tag.name.split(".")[1]) >= 5
]
versions.append("main")
for version in versions:
    html_context["versions"].append({"name": version})


# -- Translation options -----------------------------------------------------

locale_dirs = ["../locales"]
gettext_compact = "framework-docs"


# -- Project information -----------------------------------------------------

project = "Flower"
copyright = f"{datetime.date.today().year} Flower Labs GmbH"
author = "The Flower Authors"

# The full version, including alpha/beta/rc tags
release = "1.9.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "sphinxarg.ext",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.youtube",
    "sphinx_reredirects",
    "nbsphinx",
]

# Generate .rst files
autosummary_generate = True

# Document ONLY the objects from __all__ (present in __init__ files).
# It will be done recursively starting from flwr.__init__
# Starting point is controlled in the index.rst file.
autosummary_ignore_module_all = False

# Each class and function docs start with the path to it
# Make the flwr_datasets.federated_dataset.FederatedDataset appear as FederatedDataset
# The full name is still at the top of the page
add_module_names = False

# Customizations for the sphinx_copybutton extension
# Omit prompt text when copying code blocks
copybutton_prompt_text = "$ "
# Copy all lines when line continuation character is detected
copybutton_line_continuation_character = "\\"


def find_test_modules(package_path):
    """Go through the python files and exclude every *_test.py file."""
    full_path_modules = []
    for root, dirs, files in os.walk(package_path):
        for file in files:
            if file.endswith("_test.py"):
                # Construct the module path relative to the package directory
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, package_path)
                # Convert file path to dotted module path
                module_path = os.path.splitext(relative_path)[0].replace(os.sep, ".")
                full_path_modules.append(module_path)
    modules = []
    for full_path_module in full_path_modules:
        parts = full_path_module.split(".")
        for i in range(len(parts)):
            modules.append(".".join(parts[i:]))
    return modules


# Stop from documenting the *_test.py files.
# That's the only way to do that in autosummary (make the modules as mock_imports).
autodoc_mock_imports = find_test_modules(os.path.abspath("../../src/py/flwr"))

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Sphinx redirects, implemented after the doc filename changes.
# To prevent 404 errors and redirect to the new pages.
redirects = {
    # Renamed pages
    "installation": "how-to-install-flower.html",
    "configuring-clients.html": "how-to-configure-clients.html",
    "quickstart_pytorch_lightning": "tutorial-quickstart-pytorch-lightning.html",
    "quickstart_huggingface": "tutorial-quickstart-huggingface.html",
    "quickstart_pytorch": "tutorial-quickstart-pytorch.html",
    "quickstart_tensorflow": "tutorial-quickstart-tensorflow.html",
    "quickstart_scikitlearn": "tutorial-quickstart-scikitlearn.html",
    "quickstart_xgboost": "tutorial-quickstart-xgboost.html",
    "example_walkthrough_pytorch_mnist": "example-walkthrough-pytorch-mnist.html",
    "release_process": "contributor-how-to-release-flower.html",
    "saving-progress": "how-to-save-and-load-model-checkpoints.html",
    "writing-documentation": "contributor-how-to-write-documentation.html",
    "apiref-binaries": "ref-api-cli.html",
    "fedbn-example-pytorch-from-centralized-to-federated": "example-fedbn-pytorch-from-centralized-to-federated.html",
    "how-to-use-built-in-middleware-layers": "how-to-use-built-in-mods.html",
    # Restructuring: tutorials
    "tutorial/Flower-0-What-is-FL": "tutorial-series-what-is-federated-learning.html",
    "tutorial/Flower-1-Intro-to-FL-PyTorch": "tutorial-series-get-started-with-flower-pytorch.html",
    "tutorial/Flower-2-Strategies-in-FL-PyTorch": "tutorial-series-use-a-federated-learning-strategy-pytorch.html",
    "tutorial/Flower-3-Building-a-Strategy-PyTorch": "tutorial-series-build-a-strategy-from-scratch-pytorch.html",
    "tutorial/Flower-4-Client-and-NumPyClient-PyTorch": "tutorial-series-customize-the-client-pytorch.html",
    "tutorial-what-is-federated-learning.html": "tutorial-series-what-is-federated-learning.html",
    "tutorial-get-started-with-flower-pytorch.html": "tutorial-series-get-started-with-flower-pytorch.html",
    "tutorial-use-a-federated-learning-strategy-pytorch.html": "tutorial-series-use-a-federated-learning-strategy-pytorch.html",
    "tutorial-build-a-strategy-from-scratch-pytorch.html": "tutorial-series-build-a-strategy-from-scratch-pytorch.html",
    "tutorial-customize-the-client-pytorch.html": "tutorial-series-customize-the-client-pytorch.html",
    "quickstart-pytorch": "tutorial-quickstart-pytorch.html",
    "quickstart-tensorflow": "tutorial-quickstart-tensorflow.html",
    "quickstart-huggingface": "tutorial-quickstart-huggingface.html",
    "quickstart-jax": "tutorial-quickstart-jax.html",
    "quickstart-pandas": "tutorial-quickstart-pandas.html",
    "quickstart-fastai": "tutorial-quickstart-fastai.html",
    "quickstart-pytorch-lightning": "tutorial-quickstart-pytorch-lightning.html",
    "quickstart-scikitlearn": "tutorial-quickstart-scikitlearn.html",
    "quickstart-xgboost": "tutorial-quickstart-xgboost.html",
    "quickstart-android": "tutorial-quickstart-android.html",
    "quickstart-ios": "tutorial-quickstart-ios.html",
    # Restructuring: how-to guides
    "install-flower": "how-to-install-flower.html",
    "configure-clients": "how-to-configure-clients.html",
    "strategies": "how-to-use-strategies.html",
    "implementing-strategies": "how-to-implement-strategies.html",
    "save-progress": "how-to-save-and-load-model-checkpoints.html",
    "saving-and-loading-pytorch-checkpoints": "how-to-save-and-load-model-checkpoints.html",
    "monitor-simulation": "how-to-monitor-simulation.html",
    "logging": "how-to-configure-logging.html",
    "ssl-enabled-connections": "how-to-enable-ssl-connections.html",
    "upgrade-to-flower-1.0": "how-to-upgrade-to-flower-1.0.html",
    # Restructuring: explanations
    "evaluation": "explanation-federated-evaluation.html",
    "differential-privacy-wrappers": "explanation-differential-privacy.html",
    # Restructuring: references
    "apiref-flwr": "ref-api/flwr.html",
    "ref-api-flwr": "ref-api/flwr.html",
    "apiref-cli": "ref-api-cli.html",
    "examples": "ref-example-projects.html",
    "telemetry": "ref-telemetry.html",
    "changelog": "ref-changelog.html",
    "faq": "ref-faq.html",
    # Restructuring: contributor tutorials
    "first-time-contributors": "contributor-tutorial-contribute-on-github.html",
    "getting-started-for-contributors": "contributor-tutorial-get-started-as-a-contributor.html",
    # Restructuring: contributor how-to guides
    "contributor-setup": "contributor-how-to-install-development-versions.html",
    "recommended-env-setup": "contributor-how-to-set-up-a-virtual-env.html",
    "devcontainers": "contributor-how-to-develop-in-vscode-dev-containers.html",
    "creating-new-messages": "contributor-how-to-create-new-messages.html",
    "write-documentation": "contributor-how-to-write-documentation.html",
    "release-process": "contributor-how-to-release-flower.html",
    # Restructuring: contributor explanations
    "architecture": "contributor-explanation-architecture.html",
    # Restructuring: contributor references
    "good-first-contributions": "contributor-ref-good-first-contributions.html",
    "secagg": "contributor-ref-secure-aggregation-protocols.html",
    # Deleted pages
    "people": "index.html",
    "organizations": "index.html",
    "publications": "index.html",
    "quickstart_mxnet": "index.html",
    "quickstart-mxnet": "index.html",
    "tutorial-quickstart-mxnet": "index.html",
    "example-mxnet-walk-through": "index.html",
    "ref-api/flwr.simulation.run_simulation_from_cli": "index.html",
    "contributor-how-to-create-new-messages": "index.html",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = f"Flower Framework"
html_logo = "_static/flower-logo.png"
html_favicon = "_static/favicon.ico"
html_baseurl = "https://flower.ai/docs/framework/"

html_theme_options = {
    #
    # Sphinx Book Theme
    #
    # https://sphinx-book-theme.readthedocs.io/en/latest/configure.html
    # "repository_url": "https://github.com/adap/flower",
    # "repository_branch": "main",
    # "path_to_docs": "doc/source/",
    # "home_page_in_toc": True,
    # "use_repository_button": True,
    # "use_issues_button": True,
    # "use_edit_page_button": True,
    #
    # Furo
    #
    # https://pradyunsg.me/furo/customisation/
    # "light_css_variables": {
    #     "color-brand-primary": "#292F36",
    #     "color-brand-content": "#292F36",
    #     "color-admonition-background": "#F2B705",
    # },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Set modules for custom sidebar
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        "sidebar/versioning.html",
        "sidebar/lang.html",
    ]
}
# -- Options for nbsphinx -------------------------------------------------

nbsphinx_execute = "never"

_open_in_colab_button = """
.. raw:: html

    <br/>
    <a href="https://colab.research.google.com/github/adap/flower/blob/main/doc/source/{{ env.doc2path(env.docname, base=None) }}">
        <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/>
    </a>
"""
nbsphinx_prolog = _open_in_colab_button
nbsphinx_epilog = _open_in_colab_button

# -- Options for sphinxcontrib-mermaid -------------------------------------
# Don't load it automatically through the extension as we are loading it through the
# theme (see base.html) as the inclusion of require.js by the extension `nbsphinx`
# breaks the way mermaid is loaded. The solution is to load mermaid before the
# require.js script added by `nbsphinx`. We can only enforce this in the theme
# itself.
mermaid_version = ""

# -- Options for MyST config  -------------------------------------
# Enable this option to link to headers (`#`, `##`, or `###`)
myst_heading_anchors = 3
