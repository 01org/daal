/* file: bf_knn_dense_regression_batch.cpp */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
!  Content:
!    C++ example of k-Nearest Neighbor for GPU in the batch processing mode.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-BF_KNN_DENSE_REGRESSION_BATCH"></a>
 * \example bf_knn_dense_regression_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"
#include <cstdio>

using namespace std;
using namespace daal;
using namespace daal::algorithms;

using daal::data_management::internal::SyclHomogenNumericTable;
using daal::services::internal::SyclExecutionContext;

/* Input data set parameters */
const string trainDatasetFileName = "../data/batch/knn_regression_train.csv";
const string testDatasetFileName  = "../data/batch/knn_regression_test.csv";

const size_t nFeatures  = 3; /* Number of features in training and testing data sets */
const size_t nResponses = 4; /* Number of dependent variables that correspond to each observation */
const size_t nNeighbors = 4; /* The best number of neighbors defined by hyperparameter search */

void loadData(const string & fileName, NumericTablePtr & dataSamples, NumericTablePtr & dataResponses);
void trainModel(NumericTablePtr & trainSamples, bf_knn_classification::training::ResultPtr & trainingResult);
void testModel(bf_knn_classification::training::ResultPtr & trainingResult, NumericTablePtr & testSamples,
               bf_knn_classification::prediction::ResultPtr & searchResult);
void doRegression(cl::sycl::queue & q, bf_knn_classification::prediction::ResultPtr & searchResult, NumericTablePtr & trainResponses,
                  NumericTablePtr & testResponses);
void printResults(NumericTablePtr & testGroundTruth, NumericTablePtr & testPredictResult);

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;

        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        NumericTablePtr trainSamples, trainResponses;
        loadData(trainDatasetFileName, trainSamples, trainResponses);
        bf_knn_classification::training::ResultPtr trainingResult;
        trainModel(trainSamples, trainingResult);

        NumericTablePtr testSamples, testResponses;
        loadData(testDatasetFileName, testSamples, testResponses);
        bf_knn_classification::prediction::ResultPtr searchResult;
        testModel(trainingResult, testSamples, searchResult);

        const size_t nTestSamples   = testSamples->getNumberOfRows();
        NumericTablePtr testResults = SyclHomogenNumericTable<>::create(nResponses, nTestSamples, NumericTable::doAllocate);
        doRegression(queue, searchResult, trainResponses, testResults);

        printResults(testResponses, testResults);
    }
    return 0;
}

void loadData(const string & fileName, NumericTablePtr & dataSamples, NumericTablePtr & dataResponses)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data
   * from a .csv file */
    FileDataSource<CSVFeatureManager> fileDataSource(fileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);
    /* Create Numeric Tables for training data and labels */
    dataSamples   = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::notAllocate);
    dataResponses = SyclHomogenNumericTable<>::create(nResponses, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData(new MergedNumericTable(dataSamples, dataResponses));
    /* Retrieve the data from the input file */
    fileDataSource.loadDataBlock(mergedData.get());
}

void trainModel(NumericTablePtr & trainSamples, bf_knn_classification::training::ResultPtr & trainingResult)
{
    /* Create an algorithm object to train the BF kNN model */
    bf_knn_classification::training::Batch<> algorithm;
    /* Pass the training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainSamples);
    algorithm.parameter().k                 = nNeighbors;
    algorithm.parameter().resultsToEvaluate = classifier::none;
    algorithm.parameter().resultsToCompute  = bf_knn_classification::computeIndicesOfNeighbors;
    /* Train the BF kNN model */
    algorithm.compute();
    /* Retrieve the results of the training algorithm  */
    trainingResult = algorithm.getResult();
}

void testModel(bf_knn_classification::training::ResultPtr & trainingResult, NumericTablePtr & testSamples,
               bf_knn_classification::prediction::ResultPtr & searchResult)
{
    /* Create algorithm objects for BF kNN prediction with the default method */
    bf_knn_classification::prediction::Batch<> algorithm;

    /* Pass the testing data set and trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testSamples);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));
    algorithm.parameter().resultsToEvaluate = classifier::none;
    algorithm.parameter().k                 = nNeighbors;
    algorithm.parameter().resultsToCompute  = bf_knn_classification::computeIndicesOfNeighbors;

    /* Compute prediction results */
    algorithm.compute();

    /* Retrieve algorithm results */
    searchResult = algorithm.getResult();
}

void doRegression(cl::sycl::queue & q, bf_knn_classification::prediction::ResultPtr & searchResult, NumericTablePtr & trainResponses,
                  NumericTablePtr & testResponses)
{
    NumericTablePtr neighborsIndicesTable = searchResult->get(bf_knn_classification::prediction::indices);
    BlockDescriptor<int> neighborsIndicesBlock;
    BlockDescriptor<float> trainResponsesBlock;
    BlockDescriptor<float> testResponsesBlock;
    const size_t nTrainSamples = trainResponses->getNumberOfRows();
    const size_t nTestSamples  = testResponses->getNumberOfRows();
    {
        neighborsIndicesTable->getBlockOfRows(0, nTestSamples, readOnly, neighborsIndicesBlock);
        testResponses->getBlockOfRows(0, nTestSamples, writeOnly, testResponsesBlock);
        trainResponses->getBlockOfRows(0, nTrainSamples, readOnly, trainResponsesBlock);
    }
    {
        auto neighborsIndicesSharedPtr = neighborsIndicesBlock.getBuffer().toUSM(q, data_management::readOnly);
        auto testResponsesSharedPtr = testResponsesBlock.getBuffer().toUSM(q, data_management::writeOnly);
        auto trainResponsesSharedPtr = trainResponsesBlock.getBuffer().toUSM(q, data_management::readOnly);
        const float kn              = static_cast<float>(nNeighbors);
        q.submit([&](cl::sycl::handler & h) {
            auto neighborsIndicesPtr = neighborsIndicesSharedPtr.get();
            auto testResponsesPtr    = testResponsesSharedPtr.get();
            auto trainResponsesPtr   = trainResponsesSharedPtr.get();
            h.parallel_for<class regressor>(cl::sycl::range<2>(nTestSamples, nResponses), [=](cl::sycl::id<2> idx) {
                float acc(0);
                int trIndex;
                for (size_t i = 0; i < nNeighbors; ++i)
                {
                    trIndex = neighborsIndicesPtr[idx[0] * nNeighbors + i];
                    acc += trainResponsesPtr[trIndex * nResponses + idx[1]];
                }
                testResponsesPtr[idx[0] * nResponses + idx[1]] = acc / kn;
            });
        });
        q.wait();
    }
    {
        neighborsIndicesTable->releaseBlockOfRows(neighborsIndicesBlock);
        testResponses->releaseBlockOfRows(testResponsesBlock);
        trainResponses->releaseBlockOfRows(trainResponsesBlock);
    }
}

void printResults(NumericTablePtr & testGroundTruth, NumericTablePtr & testPredictResult)
{
    printNumericTable(testGroundTruth, "Regression Ground Truth (first 10 rows):", 10);
    printNumericTable(testPredictResult, "Computed Responses (first 10 rows):", 10);
}
