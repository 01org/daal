#===============================================================================
# Copyright 2014-2021 Intel Corporation
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
##     Intel(R) oneDAL configuration file for the Sphinx documentation builder
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

project = 'oneDAL'
copyright = '2014 - 2021, Intel Corporation' # pylint: disable=redefined-builtin
author = 'Intel'

# The full version, including alpha/beta/rc tags
release = '2021'

rst_prolog = """
.. |short_name| replace:: oneDAL
.. |product| replace:: oneDAL
.. |namespace| replace:: daal
.. |daal_in_code| replace:: daal
.. |reg| unicode:: U+000AE
.. |copy| unicode:: U+000A9
.. |base_tk| replace:: Intel\ |reg|\  oneAPI Base Toolkit
.. |dpcpp| replace:: Intel\ |reg|\  oneAPI DPC++/C++ Compiler
"""

if tags.has('use_intelname'):
    # use supplied =t use_intelname
    rst_prolog += ".. |full_name| replace:: Intel\ |reg|\  oneAPI Data Analytics Library"
else:
    rst_prolog += ".. |full_name| replace:: oneAPI Data Analytics Library"


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

extensions = ['sphinx-prompt', 'sphinx_substitution_extensions', 'sphinx.ext.extlinks', 'sphinx_tabs.tabs', 'dalapi', 'sphinx.ext.githubpages']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["opt-notice.rst", 'daal/data-management/numeric-tables/*.rst', 'onedal/get-started/*.rst',
                    'daal/algorithms/dbscan/distributed-steps/*',
                    'daal/algorithms/kmeans/includes/*',
                    'notes/issues/2021.1-beta06/includes/*',
                    'daal/includes/*', 'onedal/algorithms/.*/includes/*']

extlinks = {
    'cpp_example': ('https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/cpp/source/%s', ''),
    'java_example': ('https://github.com/oneapi-src/oneDAL/tree/master/examples/daal/java/com/intel/daal/examples/%s', ''),
    'daal4py_example': ('https://github.com/intel/scikit-learn-intelex/tree/master/examples/daal4py/%s', ''),
    'daal4py_sycl_example': ('https://github.com/intel/scikit-learn-intelex/tree/master/examples/daal4py/sycl/%s', ''),
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


html_theme = 'sphinx_book_theme'
html_logo = '_static/oneAPI-rgb-rev-100.png'
html_favicon = '_static/favicons.png'
# html_theme_path = ['_themes']

# Theme options
html_theme_options = {
    'repository_url': 'https://github.com/oneapi-src/oneDAL',
    'path_to_docs': 'docs/source',
    'use_issues_button': True,
    'use_edit_page_button': True,
    'repository_branch': 'master'
}

# oneDAL project directory is needed for `dalapi` extension
onedal_enable_listing = False
onedal_relative_doxyfile_dir = '../doxygen/oneapi'
onedal_relative_sources_dir = '../../cpp/oneapi/dal'

# ignore these missing references during a doc build
nitpick_ignore = [
    # top level namespace
    ('cpp:identifier', 'dal'),
    # method
    ('cpp:identifier', 'method'),
    ('cpp:identifier', 'Method'),
    # task
    ('cpp:identifier', 'task'),
    ('cpp:identifier', 'task::by_default'),
    ('cpp:identifier', 'Task'),
    # detail
    ('cpp:identifier', 'detail'),
    ('cpp:identifier', 'detail::descriptor_base<>'),
    ('cpp:identifier', 'detail::descriptor_base<>::float_t'),
    ('cpp:identifier', 'detail::descriptor_base<>::method_t'),
    ('cpp:identifier', 'detail::descriptor_base<>::task_t'),
    ('cpp:identifier', 'detail::table_builder'),
    ('cpp:identifier', 'detail::is_table_impl_v<ImplType>'),
    ('cpp:identifier', 'detail::is_homogen_table_impl_v<ImplType>'),
    ('cpp:identifier', 'detail::enable_if_classification_t<T>'),
    ('cpp:identifier', 'detail::enable_if_regression_t<T>'),
    ('cpp:identifier', 'detail::descriptor_base<>::kernel_t'),
    # data types
    ('cpp:identifier', 'int64_t'),
    ('cpp:identifier', 'data_t'),
    ('cpp:identifier', 'kernel_t'),
    ('cpp:identifier', 'distance_t'),
    # knn
    ('cpp:identifier', 'knn'),
    ('cpp:identifier', 'knn::descriptor'),
    ('cpp:identifier', 'knn::train_result'),
    ('cpp:identifier', 'knn::train_input'),
    ('cpp:identifier', 'knn::infer_result'),
    ('cpp:identifier', 'knn::infer_input'),
    ('cpp:identifier', 'detail::enable_if_brute_force_t<M>'),
    ('cpp:identifier', 'detail::enable_if_search_t<T>'),
    # minkowski_distance
    ('cpp:identifier', 'minkowski_distance'),
    ('cpp:identifier', 'minkowski_distance::descriptor'),
    ('cpp:identifier', 'oneapi::dal::minkowski_distance'),
    ('cpp:identifier', 'oneapi::dal::minkowski_distance::descriptor<Float>'),
    # chebyshev_distance
    ('cpp:identifier', 'chebyshev_distance'),
    ('cpp:identifier', 'chebyshev_distance::descriptor'),
    # kmeans
    ('cpp:identifier', 'kmeans'),
    ('cpp:identifier', 'kmeans::descriptor'),
    ('cpp:identifier', 'kmeans::train_result'),
    ('cpp:identifier', 'kmeans::train_input'),
    ('cpp:identifier', 'kmeans::infer_result'),
    ('cpp:identifier', 'kmeans::infer_input'),
    ('cpp:identifier', 'i'),
    ('cpp:identifier', 'kmeans::model'),
    ('cpp:identifier', 'infer_input'),
    ('cpp:identifier', 'infer_input::model'),
    ('cpp:identifier', 'infer_input::model::centroids'),
    ('cpp:identifier', ),
    # kmeans_init
    ('cpp:identifier', 'kmeans_init'),
    ('cpp:identifier', 'kmeans_init::descriptor'),
    ('cpp:identifier', 'kmeans_init::compute_input'),
    ('cpp:identifier', 'kmeans_init::compute_result'),
    ('cpp:identifier', 'compute'),
    # pca
    ('cpp:identifier', 'pca'),
    ('cpp:identifier', 'pca::descriptor'),
    ('cpp:identifier', 'pca::train_result'),
    ('cpp:identifier', 'pca::train_input'),
    ('cpp:identifier', 'pca::infer_result'),
    ('cpp:identifier', 'pca::infer_input'),
    # svm
    ('cpp:identifier', 'svm'),
    ('cpp:identifier', 'svm::descriptor'),
    ('cpp:identifier', 'svm::train_result'),
    ('cpp:identifier', 'svm::train_input'),
    ('cpp:identifier', 'svm::infer_result'),
    ('cpp:identifier', 'svm::infer_input'),
    ('cpp:identifier', 'Kernel'),
    ('cpp:identifier', 'oneapi::dal::svm'),
    ('cpp:identifier', 'oneapi::dal::svm::method'),
    ('cpp:identifier', 'oneapi::dal::svm::method::v1'),
    ('cpp:identifier', 'Kernel'),
    ('cpp:identifier', 'task::regression'),
    ('cpp:identifier', 'task::classification'),
    ('cpp:identifier', 'detail::enable_if_regression_t<T>'),
    # linear kernel
    ('cpp:identifier', 'linear_kernel'),
    ('cpp:identifier', 'linear_kernel::descriptor'),
    ('cpp:identifier', 'linear_kernel::compute_result'),
    ('cpp:identifier', 'linear_kernel::compute_input'),
    # polynomial kernel
    ('cpp:identifier', 'polynomial_kernel'),
    ('cpp:identifier', 'polynomial_kernel::descriptor'),
    ('cpp:identifier', 'polynomial_kernel::compute_result'),
    ('cpp:identifier', 'polynomial_kernel::compute_input'),
    # rbf kernel
    ('cpp:identifier', 'rbf_kernel'),
    ('cpp:identifier', 'rbf_kernel::descriptor'),
    ('cpp:identifier', 'rbf_kernel::compute_result'),
    ('cpp:identifier', 'rbf_kernel::compute_input'),
    # sigmoid kernel
    ('cpp:identifier', 'sigmoid_kernel'),
    ('cpp:identifier', 'sigmoid_kernel::descriptor'),
    ('cpp:identifier', 'sigmoid_kernel::compute_result'),
    ('cpp:identifier', 'sigmoid_kernel::compute_input'),
    # decision forest
    ('cpp:identifier', 'decision_forest'),
    ('cpp:identifier', 'decision_forest::infer_result'),
    ('cpp:identifier', 'decision_forest::infer_input'),
    ('cpp:identifier', 'decision_forest::train_result'),
    ('cpp:identifier', 'decision_forest::train_input'),
    ('cpp:identifier', 'decision_forest::descriptor'),
    ('cpp:identifier', 'variable_importance_mode'),
    ('cpp:identifier', 'variable_importance_mode::none'),
    ('cpp:identifier', 'variable_importance_mode::mda_raw'),
    ('cpp:identifier', 'variable_importance_mode::mda_scaled'),
    ('cpp:identifier', 'error_metric_mode'),
    ('cpp:identifier', 'error_metric_mode::none'),
    ('cpp:identifier', 'error_metric_mode::out_of_bag_error'),
    ('cpp:identifier', 'error_metric_mode::out_of_bag_error_per_observation'),
    # common for algorithms
    ('cpp:identifier', 'result'),
    # common for result options
    ('cpp:identifier', 'result_option_id_base'),
    ('cpp:identifier', 'result_option_id'),
    # tables
    ('cpp:identifier', 'table'),
    ('cpp:identifier', 'row_count'),
    ('cpp:identifier', 'column_count'),
    ('cpp:identifier', 'is_readonly'),
    ('cpp:identifier', 'range'),
    ('cpp:identifier', 'empty_table_kind'),
    ('cpp:identifier', 'data_layout'),
    ('cpp:identifier', 'data_layout::row_major'),
    ('cpp:identifier', 'data_layout::unknown'),
    ('cpp:identifier', 'feature_type'),
    ('cpp:identifier', 'data_type'),
    ('cpp:identifier', 'table_metadata'),
    ('cpp:identifier', 'mutable_data'),
    ('cpp:identifier', 'data'),
    ('cpp:identifier', 'count'),
    # array
    ('cpp:identifier', 'array'),
    ('cpp:identifier', 'T'),
    ('cpp:identifier', 'array<T>'),
    ('cpp:identifier', 'array<Y>'),
    ('cpp:identifier', 'has_mutable_data'),
    # csv
    ('cpp:identifier', 'csv'),
    ('cpp:identifier', 'read'),
    ('cpp:identifier', 'read_options'),
    ('cpp:identifier', 'default_read_options'),
    ('cpp:identifier', 'default_delimiter'),
    ('cpp:identifier', 'Object'),
    # oneapi
    ('cpp:identifier', 'oneapi'),
    ('cpp:identifier', 'oneapi::dal'),
    # oneapi - kmeans
    ('cpp:identifier', 'oneapi::dal::kmeans'),
    ('cpp:identifier', 'oneapi::dal::kmeans::task'),
    ('cpp:identifier', 'oneapi::dal::decision_forest'),
    ('cpp:identifier', 'oneapi::dal::decision_forest::task'),
    ('cpp:identifier', 'oneapi::dal::decision_forest::task::v1'),

    ('cpp:identifier', 'oneapi::dal::svm'),
    ('cpp:identifier', 'oneapi::dal::svm::method'),
    ('cpp:identifier', 'oneapi::dal::svm::task'),
]
