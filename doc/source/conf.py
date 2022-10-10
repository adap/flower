# Copyright 2020 Adap GmbH. All Rights Reserved.
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
import os
import sys
from sphinx.application import ConfigError

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# Fixing path issue for autodoc
sys.path.insert(0, os.path.abspath("../../src/py"))


# -- Project information -----------------------------------------------------

project = u"Flower"
copyright = u"2022 Adap GmbH"
author = u"The Flower Authors"

# The full version, including alpha/beta/rc tags
release = u"1.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx_reredirects",
    "nbsphinx",
]

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
    "quickstart_mxnet": "quickstart-mxnet.html",
    "quickstart_pytorch_lightning": "quickstart-pytorch-lightning.html",
    "example_walkthrough_pytorch_mnist": "example-walkthrough-pytorch-mnist.html",
    "quickstart_huggingface": "quickstart-huggingface.html",
    "quickstart_pytorch": "quickstart-pytorch.html",
    "quickstart_tensorflow": "quickstart-tensorflow.html",
    "release_process": "release-process.html",
    "quickstart_scikitlearn": "quickstart-scikitlearn.html",
    # Deleted pages
    "people": "index.html",
    "organizations": "index.html",
    "publications": "index.html",
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = f"Flower {release}"
html_logo = "_static/flower-logo.png"
html_favicon = "_static/favicon.ico"
html_baseurl = "https://flower.dev/docs/"

html_theme_options = {
    # Sphinx Book Theme
    # https://sphinx-book-theme.readthedocs.io/en/latest/configure.html
    # "repository_url": "https://github.com/adap/flower",
    # "repository_branch": "main",
    # "path_to_docs": "doc/source/",
    # "home_page_in_toc": True,
    # "use_repository_button": True,
    # "use_issues_button": True,
    # "use_edit_page_button": True,

    # Furo
    # https://pradyunsg.me/furo/customisation/
    # "light_css_variables": {
    #     "color-brand-primary": "#292F36",
    #     "color-brand-content": "#292F36",  
    #     "color-admonition-background": "#F2B705",
    # },
    "announcement": "Flower Summit 2022 <a href=\"https://flower.dev/conf/flower-summit-2022/\">watch now</a>",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for nbsphinx -------------------------------------------------

nbsphinx_execute = 'never'

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
