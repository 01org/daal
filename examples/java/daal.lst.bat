@echo off
rem ============================================================================
rem Copyright 2014-2019 Intel Corporation
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem ============================================================================

rem  Content:
rem      Intel(R) Data Analytics Acceleration Library examples list
rem ============================================================================

set Java_example_list=association_rules\AssocRulesAprioriBatch ^
boosting\adaboost\AdaBoostSammeTwoClassBatch ^
boosting\adaboost\AdaBoostSammeMultiClassBatch ^
boosting\adaboost\AdaBoostSammeRTwoClassBatch ^
boosting\adaboost\AdaBoostSammeRMultiClassBatch ^
boosting\brownboost\BrownBoostDenseBatch ^
boosting\adaboost\AdaBoostDenseBatch ^
boosting\logitboost\LogitBoostDenseBatch ^
cholesky\CholeskyDenseBatch ^
compression\CompressorExample ^
compression\CompressionBatch ^
compression\CompressionOnline ^
covariance\CorCSRBatch ^
covariance\CorCSRDistr ^
covariance\CorCSROnline ^
covariance\CorDenseBatch ^
covariance\CorDenseDistr ^
covariance\CorDenseOnline ^
covariance\CovCSRBatch ^
covariance\CovCSRDistr ^
covariance\CovCSROnline ^
covariance\CovDenseBatch ^
covariance\CovDenseDistr ^
covariance\CovDenseOnline ^
datasource\DataSourceFeatureExtraction ^
datasource\DataStructuresHomogen ^
datasource\DataStructuresAOS ^
datasource\DataStructuresSOA ^
datasource\DataStructuresCSR ^
datasource\DataStructuresMerged ^
datasource\DataStructuresMatrix ^
datasource\DataStructuresPackedSymmetric ^
datasource\DataStructuresPackedTriangular ^
datasource\DataStructuresRowMerged ^
dbscan\DBSCANDenseBatch ^
dbscan\DBSCANDenseDistr ^
decision_forest\DfClsDenseBatch ^
decision_forest\DfClsDenseBatchModelBuilder ^
decision_forest\DfClsTraversedModelBuilder ^
decision_forest\DfClsTraverseModel ^
decision_forest\DfRegDenseBatch ^
decision_forest\DfRegTraverseModel ^
decision_tree\DtClsDenseBatch ^
decision_tree\DtClsTraverseModel ^
decision_tree\DtRegDenseBatch ^
decision_tree\DtRegTraverseModel ^
distance\CorDistDenseBatch ^
distance\CosDistDenseBatch ^
em\EmGmmDenseBatch ^
gbt\GbtClsDenseBatch ^
gbt\GbtClsTraversedModelBuilder ^
gbt\GbtRegDenseBatch ^
gbt\GbtRegTraversedModelBuilder ^
gbt\GbtRegDenseDistr ^
kdtree_knn_classification\KDTreeKNNDenseBatch ^
kmeans\KMeansDenseBatch ^
kmeans\KMeansCSRBatch ^
kmeans\KMeansDenseBatchAssign ^
kmeans\KMeansInitDenseBatch ^
kmeans\KMeansCSRBatchAssign ^
kmeans\KMeansInitDenseDistr ^
kmeans\KMeansInitCSRDistr ^
kmeans\KMeansDenseDistr ^
kmeans\KMeansCSRDistr ^
lasso_regression\LassoRegDenseBatch ^
linear_regression\LinRegModelBuilder ^
linear_regression\LinRegNormEqDenseBatch ^
linear_regression\LinRegNormEqDenseDistr ^
linear_regression\LinRegNormEqDenseOnline ^
linear_regression\LinRegQRDenseBatch ^
linear_regression\LinRegQRDenseDistr ^
linear_regression\LinRegQRDenseOnline ^
logistic_regression\LogRegBinaryDenseBatch ^
logistic_regression\LogRegDenseBatch ^
logistic_regression\LogRegModelBuilder ^
moments\LowOrderMomsDenseBatch ^
moments\LowOrderMomsDenseDistr ^
moments\LowOrderMomsDenseOnline ^
moments\LowOrderMomsCSRBatch ^
moments\LowOrderMomsCSRDistr ^
moments\LowOrderMomsCSROnline ^
naive_bayes\MnNaiveBayesDenseBatch ^
naive_bayes\MnNaiveBayesCSRBatch ^
naive_bayes\MnNaiveBayesDenseOnline ^
naive_bayes\MnNaiveBayesDenseDistr ^
naive_bayes\MnNaiveBayesCSROnline ^
naive_bayes\MnNaiveBayesCSRDistr ^
outlier_detection\OutDetectUniDenseBatch ^
outlier_detection\OutDetectBaconDenseBatch ^
outlier_detection\OutDetectMultDenseBatch ^
pca\PCACorDenseBatch ^
pca\PCACorDenseDistr ^
pca\PCACorDenseOnline ^
pca\PCACorCSRBatch ^
pca\PCACorCSRDistr ^
pca\PCACorCSROnline ^
pca\PCASVDDenseBatch ^
pca\PCASVDDenseDistr ^
pca\PCASVDDenseOnline ^
pca_transform\PCATransformDenseBatch ^
qr\QRDenseBatch ^
qr\QRDenseDistr ^
qr\QRDenseOnline ^
ridge_regression\RidgeRegNormEqDenseBatch ^
ridge_regression\RidgeRegNormEqDenseOnline ^
ridge_regression\RidgeRegNormEqDistr ^
serialization\SerializationExample ^
stump\StumpClsGiniDenseBatch ^
stump\StumpClsInfogainDenseBatch ^
stump\StumpRegMseDenseBatch ^
stump\StumpDenseBatch ^
svd\SVDDenseBatch ^
svd\SVDDenseDistr ^
svd\SVDDenseOnline ^
svm\SVMMultiClassDenseBatch ^
svm\SVMMultiClassCSRBatch ^
svm\SVMMultiClassModelBuilder ^
svm\SVMTwoClassDenseBatch ^
svm\SVMTwoClassCSRBatch ^
svm\SVMTwoClassModelBuilder ^
services\LibraryVersionInfoExample ^
quantiles\QuantilesDenseBatch ^
pivoted_qr\PivotedQRDenseBatch ^
quality_metrics\LinRegMetricsDenseBatch ^
quality_metrics\PCAMetricsDenseBatch ^
quality_metrics\SVMTwoClassMetricsDenseBatch ^
quality_metrics\SVMMultiClassMetricsDenseBatch ^
kernel_function\KernelFuncLinDenseBatch ^
kernel_function\KernelFuncLinCSRBatch ^
kernel_function\KernelFuncRbfDenseBatch ^
kernel_function\KernelFuncRbfCSRBatch ^
implicit_als\ImplAlsCSRBatch ^
implicit_als\ImplAlsCSRDistr ^
implicit_als\ImplAlsDenseBatch ^
set_number_of_threads\SetNumberOfThreads ^
datasource\DataStructuresMerged ^
sorting\SortingDenseBatch ^
error_handling\ErrorHandling ^
normalization\ZScoreDenseBatch ^
normalization\MinMaxDenseBatch ^
optimization_solvers\CDDenseBatch ^
optimization_solvers\MSEDenseBatch ^
optimization_solvers\SGDDenseBatch ^
optimization_solvers\SGDMiniDenseBatch ^
optimization_solvers\SGDMomentDenseBatch ^
optimization_solvers\SGDMomentOptResDenseBatch ^
optimization_solvers\LBFGSDenseBatch ^
optimization_solvers\LBFGSOptResDenseBatch ^
optimization_solvers\AdagradDenseBatch ^
optimization_solvers\AdagradOptResDenseBatch ^
optimization_solvers\SAGADenseBatch ^
optimization_solvers\SAGALogisticLossDenseBatch ^
distributions\UniformDenseBatch ^
distributions\BernoulliDenseBatch ^
distributions\NormalDenseBatch ^
enable_thread_pinning\EnableThreadPinning
