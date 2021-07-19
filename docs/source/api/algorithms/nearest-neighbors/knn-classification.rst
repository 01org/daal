.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. highlight:: cpp
.. default-domain:: cpp

.. _api_knn:

=========================================
k-Nearest Neighbors Classification (k-NN)
=========================================

.. include:: ../../../includes/nearest-neighbors/knn-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: k-Nearest Neighbors Classification<alg_knn>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::knn`` namespace and be available via inclusion of the
``oneapi/dal/algo/knn.hpp`` header file.

Enum classes
------------
.. onedal_enumclass:: oneapi::dal::knn::voting_mode

Result options
--------------
.. onedal_class:: oneapi::dal::knn::result_option_id

Descriptor
----------
.. onedal_class:: oneapi::dal::knn::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::knn::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::knn::task

Model
-----
.. onedal_class:: oneapi::dal::knn::model


.. _knn_t_api:

Training :cpp:expr:`train(...)`
--------------------------------
.. _knn_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::knn::train_input


.. _knn_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::knn::train_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              knn::train_result train(const Descriptor& desc, \
                                         const knn::train_input& input)

   :param desc: k-NN algorithm descriptor :expr:`knn::descriptor`
   :param input: Input data for the training operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.labels.has_data == true`
      | :expr:`input.data.row_count == input.labels.row_count`
      | :expr:`input.labels.column_count == 1`
      | :expr:`input.labels[i] >= 0`
      | :expr:`input.labels[i] < desc.class_count`

.. _knn_i_api:

Inference :cpp:expr:`infer(...)`
---------------------------------
.. _knn_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::knn::infer_input


.. _knn_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::knn::infer_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              knn::infer_result infer(const Descriptor& desc, \
                                         const knn::infer_input& input)

   :param desc: k-NN algorithm descriptor :expr:`knn::descriptor`
   :param input: Input data for the inference operation

   Preconditions
      | :expr:`input.data.has_data == true`
   Postconditions
     | :expr:`result.labels.row_count == input.data.row_count`
     | :expr:`result.labels.column_count == 1`
     | :expr:`result.labels[i] >= 0`
     | :expr:`result.labels[i] < desc.class_count`

-------------
Usage example
-------------

.. include:: ../../../includes/nearest-neighbors/knn-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/nearest-neighbors/knn-examples.rst
