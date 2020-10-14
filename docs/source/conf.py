#===============================================================================
# Copyright 2014-2020 Intel Corporation
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
#===============================================================================

##  Content:
##     Intel(R) DAAL configuration file for the Sphinx documentation builder
##******************************************************************************

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'Intel&reg; oneAPI Data Analytics Library Documentation (Beta)'
copyright = '2014 - 2020, Intel Corporation' # pylint: disable=redefined-builtin
author = 'Intel'

# The full version, including alpha/beta/rc tags
release = '1.0'

rst_prolog = """
.. |full_name| replace:: Intel\ |reg|\  oneAPI Data Analytics Library
.. |short_name| replace:: oneDAL
.. |product| replace:: oneDAL
.. |namespace| replace:: daal
.. |daal_in_code| replace:: daal
.. |reg| unicode:: U+000AE
.. |copy| unicode:: U+000A9
.. |base_tk| replace:: Intel\ |reg|\  oneAPI Base Toolkit
.. |dpcpp| replace:: Intel\ |reg|\  oneAPI DPC++/C++ Compiler
"""


# for substitutions in code blocks and sphinx-prompts:
substitutions = [
    ('|short_name|', 'oneDAL'),
    ('|daal_in_code|', 'daal')
    ]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# sys.path.insert(0, path_relative_to_repo_root('source/elements/oneDAL'))

extensions = ['sphinx-prompt', 'sphinx_substitution_extensions', 'sphinx.ext.extlinks', 'sphinx_tabs.tabs', 'dalapi']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["opt-notice.rst", 'dev_guide/data-management/numeric-tables/*.rst', 'topics/*.rst',
                    'dev_guide/algorithms/dbscan/distributed-steps/*',
                    'dev_guide/algorithms/kmeans/includes/*',
                    'notes/issues/2021.1-beta06/includes/*',
                    'dev_guide/includes/*']

extlinks = {
    'cpp_example': ('https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/cpp/source/%s', ''),
    'java_example': ('https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/java/com/intel/daal/examples/%s', ''),
    'daal4py_example': ('https://github.com/IntelPython/daal4py/tree/master/examples/%s', ''),
    'daal4py_sycl_example': ('https://github.com/IntelPython/daal4py/tree/master/examples/sycl/%s', ''),
    'cpp_sample': ('https://github.com/oneapi-src/oneDAL/tree/master/samples/daal/cpp/%s', '')
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
## html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_context = {
#     'css_files': [
#         '_static/style.css',  # override wide tables in RTD theme
#         ],
#     }


# html_theme = 'otc_tcs_sphinx_theme'
# html_theme_path = ['_themes']

import sys
import os
# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
if on_rtd:
    using_rtd_theme = True

# Theme options
html_theme_options = {
    # 'typekit_id': 'hiw1hhg',
    # 'analytics_id': '',
    # 'sticky_navigation': True,  # Set to False to disable the sticky nav while scrolling.
    'logo_only': True,  # if we have a html_logo below, this shows /only/ the logo with no title text
    'collapse_navigation': False,  # Collapse navigation (False makes it tree-like)
    # 'display_version': True,  # Display the docs version
    'navigation_depth': 4  # Depth of the headers shown in the navigation bar
}

# oneDAL project directory is needed for `dalapi` extension
onedal_enable_listing = False
onedal_relative_doxyfile_dir = '../doxygen/oneapi'
onedal_relative_sources_dir = '../../cpp/oneapi/dal'
