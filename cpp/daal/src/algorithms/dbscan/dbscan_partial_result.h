/* file: dbscan_partial_result.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//++
//  Implementation of dbscan classes.
//--
*/

#ifndef __DBSCAN_PARTIAL_RESULT__
#define __DBSCAN_PARTIAL_RESULT__

#include "algorithms/dbscan/dbscan_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep1::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const DistributedInput<step1Local> * algInput = static_cast<const DistributedInput<step1Local> *>(input);
    const size_t nRows                            = algInput->get(step1Data)->getNumberOfRows();

    services::Status status;
    set(partialOrder, HomogenNumericTable<int>::create(2, nRows, NumericTable::doAllocate, &status));
    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep2::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const DistributedInput<step2Local> * algInput = static_cast<const DistributedInput<step2Local> *>(input);
    const size_t nFeatures                        = NumericTable::cast((*algInput->get(partialData))[0])->getNumberOfColumns();

    services::Status status;
    set(boundingBox, HomogenNumericTable<algorithmFPType>::create(nFeatures, 2, NumericTable::doAllocate, &status));
    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep3::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status status;
    set(split, HomogenNumericTable<algorithmFPType>::create(2, 1, NumericTable::doAllocate, &status));
    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep4::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const Parameter * par = static_cast<const Parameter *>(parameter);
    const size_t nBlocks  = par->leftBlocks + par->rightBlocks;

    const DistributedInput<step4Local> * algInput = static_cast<const DistributedInput<step4Local> *>(input);
    const size_t nFeatures                        = NumericTable::cast((*algInput->get(partialData))[0])->getNumberOfColumns();

    services::Status status;
    DataCollectionPtr dcPartitionedData(new DataCollection(nBlocks));
    DataCollectionPtr dcPartitionedPartialOrders(new DataCollection(nBlocks));

    DAAL_CHECK_MALLOC(dcPartitionedData.get());
    DAAL_CHECK_MALLOC(dcPartitionedPartialOrders.get());

    for (size_t i = 0; i < nBlocks; i++)
    {
        (*dcPartitionedData)[i]          = HomogenNumericTable<algorithmFPType>::create(nFeatures, 0, NumericTable::notAllocate, &status);
        (*dcPartitionedPartialOrders)[i] = HomogenNumericTable<int>::create(2, 0, NumericTable::notAllocate, &status);
    }

    set(partitionedData, dcPartitionedData);
    set(partitionedPartialOrders, dcPartitionedPartialOrders);
    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep5::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const Parameter * par = static_cast<const Parameter *>(parameter);
    const size_t nBlocks  = par->nBlocks;

    const DistributedInput<step5Local> * algInput = static_cast<const DistributedInput<step5Local> *>(input);
    const size_t nFeatures                        = NumericTable::cast((*algInput->get(partialData))[0])->getNumberOfColumns();

    services::Status status;
    DataCollectionPtr dcPartitionedHaloData(new DataCollection(nBlocks));
    DAAL_CHECK_MALLOC(dcPartitionedHaloData.get());
    DataCollectionPtr dcPartitionedHaloDataIndices(new DataCollection(nBlocks));
    DAAL_CHECK_MALLOC(dcPartitionedHaloDataIndices.get());

    for (size_t i = 0; i < nBlocks; i++)
    {
        (*dcPartitionedHaloData)[i]        = HomogenNumericTable<algorithmFPType>::create(nFeatures, 0, NumericTable::notAllocate, &status);
        (*dcPartitionedHaloDataIndices)[i] = HomogenNumericTable<int>::create(1, 0, NumericTable::notAllocate, &status);
    }

    set(partitionedHaloData, dcPartitionedHaloData);
    set(partitionedHaloDataIndices, dcPartitionedHaloDataIndices);

    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep6::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const Parameter * par = static_cast<const Parameter *>(parameter);
    const size_t nBlocks  = par->nBlocks;

    const DistributedInput<step6Local> * algInput = static_cast<const DistributedInput<step6Local> *>(input);

    DataCollectionPtr dcPartialData = algInput->get(partialData);
    size_t nDataBlocks              = dcPartialData->size();

    size_t nRows = 0;
    for (size_t i = 0; i < nDataBlocks; i++)
    {
        nRows += NumericTable::cast((*algInput->get(partialData))[i])->getNumberOfRows();
    }

    services::Status status;

    if (nRows != 0)
    {
        set(step6ClusterStructure, HomogenNumericTable<int>::create(4, nRows, NumericTable::doAllocate, &status));
    }
    else
    {
        set(step6ClusterStructure, HomogenNumericTable<int>::create(4, 0, NumericTable::notAllocate, &status));
    }
    set(step6FinishedFlag, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));
    set(step6NClusters, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));

    DataCollectionPtr dcQueries(new DataCollection(nBlocks));
    DAAL_CHECK_MALLOC(dcQueries.get());

    for (size_t i = 0; i < nBlocks; i++)
    {
        (*dcQueries)[i] = HomogenNumericTable<int>::create(3, 0, NumericTable::notAllocate, &status);
    }

    set(step6Queries, dcQueries);

    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep7::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status status;
    set(finishedFlag, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));
    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep8::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const Parameter * par = static_cast<const Parameter *>(parameter);
    const size_t nBlocks  = par->nBlocks;

    const DistributedInput<step8Local> * algInput = static_cast<const DistributedInput<step8Local> *>(input);
    const size_t nRows                            = algInput->get(step8InputClusterStructure)->getNumberOfRows();

    services::Status status;

    if (nRows != 0)
    {
        set(step8ClusterStructure, HomogenNumericTable<int>::create(4, nRows, NumericTable::doAllocate, &status));
    }
    else
    {
        set(step8ClusterStructure, HomogenNumericTable<int>::create(4, 0, NumericTable::notAllocate, &status));
    }
    set(step8FinishedFlag, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));
    set(step8NClusters, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));

    DataCollectionPtr dcQueries(new DataCollection(nBlocks));
    DAAL_CHECK_MALLOC(dcQueries.get());

    for (size_t i = 0; i < nBlocks; i++)
    {
        (*dcQueries)[i] = HomogenNumericTable<int>::create(3, 0, NumericTable::notAllocate, &status);
    }

    set(step8Queries, dcQueries);

    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedResultStep9::allocate(const daal::algorithms::PartialResult * pres,
                                                              const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status status;
    set(step9NClusters, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));
    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep9::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const DistributedInput<step9Master> * algInput = static_cast<const DistributedInput<step9Master> *>(input);
    const size_t nBlocks                           = algInput->get(partialNClusters)->size();

    services::Status status;

    DataCollectionPtr dcClusterOffsets(new DataCollection(nBlocks + 1));
    DAAL_CHECK_MALLOC(dcClusterOffsets.get());

    for (size_t i = 0; i < nBlocks + 1; i++)
    {
        (*dcClusterOffsets)[i] = HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status);
    }

    set(clusterOffsets, dcClusterOffsets);

    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep10::allocate(const daal::algorithms::Input * input,
                                                                      const daal::algorithms::Parameter * parameter, const int method)
{
    const Parameter * par = static_cast<const Parameter *>(parameter);
    const size_t nBlocks  = par->nBlocks;

    const DistributedInput<step10Local> * algInput = static_cast<const DistributedInput<step10Local> *>(input);
    const size_t nRows                             = algInput->get(step10InputClusterStructure)->getNumberOfRows();

    services::Status status;

    if (nRows != 0)
    {
        set(step10ClusterStructure, HomogenNumericTable<int>::create(4, nRows, NumericTable::doAllocate, &status));
    }
    else
    {
        set(step10ClusterStructure, HomogenNumericTable<int>::create(4, 0, NumericTable::notAllocate, &status));
    }
    set(step10FinishedFlag, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));

    DataCollectionPtr dcQueries(new DataCollection(nBlocks));
    DAAL_CHECK_MALLOC(dcQueries.get());

    for (size_t i = 0; i < nBlocks; i++)
    {
        (*dcQueries)[i] = HomogenNumericTable<int>::create(4, 0, NumericTable::notAllocate, &status);
    }

    set(step10Queries, dcQueries);

    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep11::allocate(const daal::algorithms::Input * input,
                                                                      const daal::algorithms::Parameter * parameter, const int method)
{
    const Parameter * par = static_cast<const Parameter *>(parameter);
    const size_t nBlocks  = par->nBlocks;

    const DistributedInput<step11Local> * algInput = static_cast<const DistributedInput<step11Local> *>(input);
    const size_t nRows                             = algInput->get(step11InputClusterStructure)->getNumberOfRows();

    services::Status status;

    if (nRows != 0)
    {
        set(step11ClusterStructure, HomogenNumericTable<int>::create(4, nRows, NumericTable::doAllocate, &status));
    }
    else
    {
        set(step11ClusterStructure, HomogenNumericTable<int>::create(4, 0, NumericTable::notAllocate, &status));
    }
    set(step11FinishedFlag, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));

    DataCollectionPtr dcQueries(new DataCollection(nBlocks));
    DAAL_CHECK_MALLOC(dcQueries.get());

    for (size_t i = 0; i < nBlocks; i++)
    {
        (*dcQueries)[i] = HomogenNumericTable<int>::create(4, 0, NumericTable::notAllocate, &status);
    }

    set(step11Queries, dcQueries);

    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep12::allocate(const daal::algorithms::Input * input,
                                                                      const daal::algorithms::Parameter * parameter, const int method)
{
    const Parameter * par = static_cast<const Parameter *>(parameter);
    const size_t nBlocks  = par->nBlocks;

    services::Status status;

    DataCollectionPtr dcAssignmentQueries(new DataCollection(nBlocks));
    DAAL_CHECK_MALLOC(dcAssignmentQueries.get());

    for (size_t i = 0; i < nBlocks; i++)
    {
        (*dcAssignmentQueries)[i] = HomogenNumericTable<int>::create(2, 0, NumericTable::notAllocate, &status);
    }

    set(assignmentQueries, dcAssignmentQueries);

    return status;
}

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedResultStep13::allocate(const daal::algorithms::PartialResult * pres,
                                                               const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status status;
    set(step13Assignments, HomogenNumericTable<int>::create(1, 0, NumericTable::notAllocate, &status));
    return status;
}

/**
 * Allocates memory to store the partial results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep13::allocate(const daal::algorithms::Input * input,
                                                                      const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status status;
    set(step13AssignmentQueries, HomogenNumericTable<int>::create(2, 0, NumericTable::notAllocate, &status));
    return status;
}

} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
