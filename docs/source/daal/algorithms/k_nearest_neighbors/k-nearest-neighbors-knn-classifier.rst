.. ******************************************************************************
.. * Copyright 2019-2021 Intel Corporation
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

.. _knn:

k-Nearest Neighbors (kNN) Classifier
************************************

.. note::

   k-Nearest Neighbors Classifier is also available with oneAPI interfaces:

   - :ref:`alg_knn`

.. toctree::
   :glob:
   :maxdepth: 4


k-Nearest Neighbors (kNN) classification is a non-parametric
classification algorithm. The model of the kNN classifier is based on
feature vectors and class labels from the training data set. This
classifier induces the class of the query vector from the labels of
the feature vectors in the training data set to which the query
vector is similar. A similarity between feature vectors is determined
by the type of distance (for example, Euclidian) in a
multidimensional feature space.

Details
=======

Given n feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of size :math:`p` and a vector
of class labels :math:`y = (y_1, \ldots, y_n)`, where :math:`y_i \in \{0, 1, \ldots, C-1\}`
and :math:`C` is the number of classes, describes the
class to which the feature vector :math:`x_i` belongs, the problem is
to build a kNN classifier.

Given a positive integer parameter :math:`k` and a test observation
:math:`x_0`, the kNN classifier does the following:

#. Identifies the set :math:`N_0` of the k feature vectors in the
   training data that are closest to :math:`x_0` according to the
   distance metric
#. Estimates the conditional probability for the class :math:`j` as the
   fraction of vectors in :math:`N_0` whose labels y are equal to :math:`j`
#. Assigns the class with the largest probability to the test
   observation :math:`x_0`

On CPU, kNN classification might use `K-D tree`_, a space-partitioning data structure,
or `Brute Force`_ search to find nearest neighbors,
while on GPU only `Brute Force`_ search is available.

K-D tree
--------

On CPU, the library provides kNN classification based on multidimensional binary search tree
(K-D tree, where D means the dimension and K means the number of dimensions in the feature space).
For more details, see [James2013]_, [Patwary2016]_.

|product| version of the kNN algorithm with K-D trees uses the PANDA algorithm
[Patwary2016]_.

Each non-leaf node of a tree contains the identifier of a feature along
which to split the feature space and an appropriate feature value (a
cut-point) that defines the splitting hyperplane to partition the
feature space into two parts. Each leaf node of the tree has an
associated subset (a bucket) of elements of the training data set.
Feature vectors from any bucket belong to the region of the space
defined by tree nodes on the path from the root node to the
respective leaf.

Brute Force
-----------

Brute Force kNN algorithm calculates the squared distances from each query feature vector
to each reference feature vector in the training data set. Then,
for each query feature vector it selects :math:`k` objects from the training set that are closest to that query feature vector.
For details, see [Li2015]_, [Verma2014]_.

Training Stage
--------------

Training using K-D Tree
+++++++++++++++++++++++

For each non-leaf node, the process of building a K-D tree
involves the choice of the feature (that is, dimension in the
feature space) and the value for this feature (a cut-point) to
split the feature space. This procedure starts with the entire
feature space for the root node of the tree, and for every next
level of the tree deals with ever smaller part of the feature
space.

The PANDA algorithm constructs the K-D tree by choosing the
dimension with the maximum variance for splitting
[Patwary2016]_.

Therefore, for each new non-leaf node of the tree, the algorithm
computes the variance of values that belong to the respective
region of the space for each of the features and chooses the
feature with the largest variance. Due to high computational cost
of this operation, PANDA uses a subset of feature values to
compute the variance.

PANDA uses a sampling heuristic to estimate the data distribution
for the chosen feature and chooses the median estimate as the
cut-point.

PANDA generates new K-D tree levels until the number of feature
vectors in a leaf node gets less or equal to a predefined
threshold. Once the threshold is reached, PANDA stops growing the
tree and associates the feature vectors with the bucket of the
respective leaf node.


Training using Brute Force
++++++++++++++++++++++++++

During training with the Brute Force approach, the algorithm stores all feature vectors from the training data set
to calculate their distances to the query feature vectors.

Prediction Stage
----------------

Given kNN classifier and query vectors :math:`x_0, \ldots, x_r`,
the problem is to calculate the labels for those vectors. 

Prediction using K-D Tree
+++++++++++++++++++++++++

To solve the problem for each given query vector
:math:`x_i`, the algorithm traverses the K-D tree to find feature
vectors associated with a leaf node that are closest to
:math:`x_i`. During the search, the algorithm limits exploration
of the nodes for which the distance between the query vector and
respective part of the feature space is not less than the distance
from the :math:`k^{th}` neighbor. This distance is progressively
updated during the tree traverse.

Prediction using Brute Force
++++++++++++++++++++++++++++

To solve the problem, the algorithm computes distances between vectors from training and testing sets:
:math:`d_{ij}=\mathrm{distance\_metric}(x_i^\mathrm{test}, x_j^\mathrm{train})`.
For example, if Euclidean distance is used, :math:`d_{ij}` would be the following:

.. math::
  d_{ij} = \sum_{k=1}^p (x_{ik}^{\mathrm{test}} - x_{jk}^{\mathrm{train}})^2

K training vectors with minimal distance to the testing vector are the nearest neighbors the algorithms searches for.

Batch Processing
================

kNN classification follows the general workflow described in
:ref:`classification_usage_model`.

Training
--------

For a description of the input and output, refer to Usage Model: Training and Prediction.

At the training stage, both Brute Force and K-D tree based kNN classifier have the
following parameters:

.. list-table::
   :widths: 10 20 30
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - The computation method used by kNN classification.
       The only training method supported so far is the default dense method.
   * - ``nClasses``
     - :math:`2`
     - The number of classes.
   * - ``dataUseInModel``
     - ``doNotUse``
     - A parameter to enable/disable use of the input data set in the kNN
       model. Possible values:

        + ``doNotUse`` - the algorithm does not include the input data and labels
          in the trained kNN model but creates a copy of the input data set.
        + ``doUse`` - the algorithm includes the input data and labels in the trained kNN model.

       K-D tree based kNN reorders feature vectors and corresponding labels in the
       input data set or its copy to improve performance at the prediction stage.

       If the value is ``doUse``, do not deallocate the memory for input data and labels.

   * - ``engine``
     - `SharePtr< engines:: mt19937:: Batch>()`
     - Pointer to the random number generator engine that is used internally to
       perform sampling needed to choose dimensions and cut-points for the K-D tree.

Prediction
----------

For a description of the input and output, refer to Usage Model: Training and Prediction.

At the prediction stage, both Brute Force and K-D tree based kNN classifier have the
following parameters:

.. list-table::
   :widths: 10 10 60
   :header-rows: 1
   :align: left

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - The computation method used kNN classification.
       The only prediction method supported so far is the default dense method.
   * - ``nClasses``
     - :math:`2`
     - The number of classes.
   * - :math:`k`
     - :math:`1`
     - The number of neighbors.
   * - ``resultsToCompute``
     - :math:`0`
     - The 64-bit integer flag that specifies which extra characteristics of the kNN algorithm to compute.
       Provide one of the following values to request a single characteristic or use bitwise OR to request a combination of the characteristics:

       - ``computeIndicesOfNeighbors``
       - ``computeDistances``

   * - ``voteWeights``
     - ``voteUniform``
     - The voting method for prediction:

       - ``voteUniform`` – Uniform weighting is used. All neighbors weight equally.
       - ``voteDistance`` – Inverse-distance weighting is used.
         The closer to the query point the neighbor is, the more it weights.

Output
------

In addition to classifier output, kNN calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the result of your algorithm.

.. list-table::
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - ``indices`` 
     - A numeric table :math:`n \times k` containing indices of rows from training dataset that are nearest neighbors computed when the ``computeIndicesOfNeigtbors`` option is on.
     
       .. note::
       
          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``.
   * - ``distances``
     - A numeric table :math:`n \times k` containing distances to nearest neighbors computed when the ``computeDistances`` option is on.
     
       .. note::
          
          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``.

Examples
========

.. tabs::

  .. tab:: oneAPI DPC++

    Batch Processing:

    - :ref:`dpc_knn_cls_brute_force_dense_batch.cpp`
  
  .. tab:: oneAPI C++

    Batch Processing:

    - :ref:`cpp_knn_cls_kd_tree_dense_batch.cpp`

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`kdtree_knn_dense_batch.cpp <k_nearest_neighbors/kdtree_knn_dense_batch.cpp>`
    - :cpp_example:`bf_knn_dense_batch.cpp <k_nearest_neighbors/bf_knn_dense_batch.cpp>`
    - :cpp_example:`kdtree_knn_dense_search_batch.cpp <k_nearest_neighbors/kdtree_knn_dense_search_batch.cpp>`
    - :cpp_example:`bf_knn_dense_search_batch.cpp <k_nearest_neighbors/bf_knn_dense_search_batch.cpp>`

  .. tab:: Java*
  
    .. note:: There is no support for Java on GPU.

    Batch Processing:

    - :java_example:`KDTreeKNNDenseBatch.java <knn_classification/KDTreeKNNDenseBatch.java>`
    - :java_example:`BFKNNDenseBatch.java <knn_classification/BFKNNDenseBatch.java>`

  .. tab:: Python* with DPC++ support

    Batch Processing:

    - :daal4py_sycl_example:`bf_knn_classification_batch.py`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`kdtree_knn_classification_batch.py`
    - :daal4py_example:`bf_knn_classification_batch.py`
