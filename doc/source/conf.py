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

from sphinx.application import ConfigError

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------

project = "Flower"
copyright = "2020, Adap GmbH"
author = "The Flower Authors"

# The full version, including alpha/beta/rc tags
release = "0.4.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# https://alabaster.readthedocs.io/en/latest/customization.html
html_theme_options = {
    "logo": "flower-logo.png",
    "logo_name": False,
    "sidebar_collapse": False,
    # GitHub
    "github_user": "adap",
    "github_repo": "flower",
    "github_type": "star",
    "github_banner": True,
    "github_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Analytics ---------------------------------------------------------------

googleanalytics_id = "UA-173987939-1"


def html_page_context(app, pagename, templatename, context, doctree):
    metatags = context.get("metatags", "")
    metatags += (
        """<script type="text/javascript">
      var _gaq = _gaq || [];
      _gaq.push(['_setAccount', '%s']);
      _gaq.push(['_trackPageview']);
      (function() {
        var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
        ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
        var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(ga, s);
      })();
    </script>"""
        % app.config.googleanalytics_id
    )
    context["metatags"] = metatags


def check_config(app):
    if not app.config.googleanalytics_id:
        raise ConfigError(
            "'googleanalytics_id' config value must be set for ga statistics to function properly."
        )


def setup(app):
    app.add_config_value("googleanalytics_id", "", "html")
    app.connect("builder-inited", check_config)
    app.connect("html-page-context", html_page_context)
