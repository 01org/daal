/* file: dbscan_dense_default_distr_impl.i */
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
//  Implementation of default method for DBSCAN algorithm.
//--
*/

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/services/service_data_utils.h"

#include "src/threading/threading.h"
#include "algorithms/dbscan/dbscan_types.h"
#include "src/algorithms/dbscan/dbscan_utils.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
#define __DBSCAN_PREFETCHED_NEIGHBORHOODS_COUNT 64

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep1Kernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntData, NumericTable * ntPartialOrder,
                                                                     const Parameter * par)
{
    const size_t nRows      = ntData->getNumberOfRows();
    const size_t blockIndex = par->blockIndex;

    const size_t defaultBlockSize = 256;
    const size_t nDataBlocks      = nRows / defaultBlockSize + int(nRows % defaultBlockSize > 0);

    for (size_t block = 0; block < nDataBlocks; block++)
    {
        const size_t i1    = block * defaultBlockSize;
        const size_t i2    = (block + 1 == nDataBlocks ? nRows : i1 + defaultBlockSize);
        const size_t iSize = i2 - i1;

        WriteRows<int, cpu> partialOrderRows(ntPartialOrder, i1, iSize);
        DAAL_CHECK_BLOCK_STATUS(partialOrderRows);
        int * const partialOrder = partialOrderRows.get();

        for (size_t i = 0; i < iSize; i++)
        {
            partialOrder[i * 2]     = blockIndex;
            partialOrder[i * 2 + 1] = i + i1;
        }
    }

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep2Kernel<algorithmFPType, method, cpu>::compute(const DataCollection * dcPartialData, NumericTable * ntBoundingBox,
                                                                     const Parameter * par)
{
    const size_t nFeatures = NumericTable::cast((*dcPartialData)[0])->getNumberOfColumns();

    WriteRows<algorithmFPType, cpu> boundingBoxRows(ntBoundingBox, 0, 2);
    DAAL_CHECK_BLOCK_STATUS(boundingBoxRows);
    algorithmFPType * boundingBox = boundingBoxRows.get();

    for (size_t i = 0; i < nFeatures; i++)
    {
        boundingBox[i]             = MaxVal<algorithmFPType>::get();
        boundingBox[i + nFeatures] = -MaxVal<algorithmFPType>::get();
    }

    const size_t defaultBlockSize = 256;

    for (size_t part = 0; part < dcPartialData->size(); part++)
    {
        NumericTablePtr ntData   = NumericTable::cast((*dcPartialData)[part]);
        const size_t nRows       = ntData->getNumberOfRows();
        const size_t nDataBlocks = nRows / defaultBlockSize + int(nRows % defaultBlockSize > 0);

        for (size_t block = 0; block < nDataBlocks; block++)
        {
            const size_t i1    = block * defaultBlockSize;
            const size_t i2    = (block + 1 == nDataBlocks ? nRows : i1 + defaultBlockSize);
            const size_t iSize = i2 - i1;

            WriteRows<algorithmFPType, cpu> dataRows(ntData.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(dataRows);
            algorithmFPType * const data = dataRows.get();

            for (size_t i = 0; i < iSize; i++)
            {
                for (size_t j = 0; j < nFeatures; j++)
                {
                    boundingBox[j]             = min<cpu, algorithmFPType>(boundingBox[j], data[i * nFeatures + j]);
                    boundingBox[j + nFeatures] = max<cpu, algorithmFPType>(boundingBox[j + nFeatures], data[i * nFeatures + j]);
                }
            }
        }
    }

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep3Kernel<algorithmFPType, method, cpu>::compute(const DataCollection * dcPartialData,
                                                                     const DataCollection * dcPartialBoundingBoxes, NumericTable * ntSplit,
                                                                     const Parameter * par)
{
    const size_t leftBlocks  = par->leftBlocks;
    const size_t rightBlocks = par->rightBlocks;

    const size_t nFeatures = NumericTable::cast((*dcPartialData)[0])->getNumberOfColumns();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, 2);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, 2 * nFeatures, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> boundingBoxArray(2 * nFeatures);
    DAAL_CHECK_MALLOC(boundingBoxArray.get());
    algorithmFPType * const boundingBox = boundingBoxArray.get();

    for (size_t i = 0; i < nFeatures; i++)
    {
        boundingBox[i]             = MaxVal<algorithmFPType>::get();
        boundingBox[i + nFeatures] = -MaxVal<algorithmFPType>::get();
    }

    for (size_t part = 0; part < dcPartialBoundingBoxes->size(); part++)
    {
        NumericTablePtr ntPartialBoundingBox = NumericTable::cast((*dcPartialBoundingBoxes)[part]);

        WriteRows<algorithmFPType, cpu> partialBoundingBoxRows(ntPartialBoundingBox.get(), 0, 2);
        DAAL_CHECK_BLOCK_STATUS(partialBoundingBoxRows);
        algorithmFPType * const partialBoundingBox = partialBoundingBoxRows.get();

        for (size_t i = 0; i < nFeatures; i++)
        {
            boundingBox[i]             = min<cpu, algorithmFPType>(boundingBox[i], partialBoundingBox[i]);
            boundingBox[i + nFeatures] = max<cpu, algorithmFPType>(boundingBox[i + nFeatures], partialBoundingBox[i + nFeatures]);
        }
    }

    algorithmFPType splitDiff = -MaxVal<algorithmFPType>::get();
    size_t splitDim           = -1;

    for (size_t i = 0; i < nFeatures; i++)
    {
        algorithmFPType curDiff = boundingBox[i + nFeatures] - boundingBox[i];
        if (curDiff > splitDiff)
        {
            splitDiff = curDiff;
            splitDim  = i;
        }
    }

    DAAL_ASSERT(splitDim != -1);

    size_t totalNRows = 0;
    for (size_t part = 0; part < dcPartialData->size(); part++)
    {
        NumericTablePtr ntData = NumericTable::cast((*dcPartialData)[part]);
        totalNRows += ntData->getNumberOfRows();
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, totalNRows, sizeof(algorithmFPType));

    if (totalNRows == 0)
    {
        WriteRows<algorithmFPType, cpu> splitRows(ntSplit, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(splitRows);
        algorithmFPType * const split = splitRows.get();
        split[0]                      = 0;
        split[1]                      = (algorithmFPType)(-1);
        return Status();
    }

    TArray<algorithmFPType, cpu> splitColumnArray(totalNRows);
    DAAL_CHECK_MALLOC(splitColumnArray.get());
    algorithmFPType * const splitColumn = splitColumnArray.get();

    size_t pos = 0;
    int result = 0;
    for (size_t part = 0; part < dcPartialData->size(); part++)
    {
        NumericTablePtr ntData = NumericTable::cast((*dcPartialData)[part]);
        const size_t nRows     = ntData->getNumberOfRows();

        ReadColumns<algorithmFPType, cpu> dataColumns(ntData.get(), splitDim, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(dataColumns);
        const algorithmFPType * const data = dataColumns.get();
        result |=
            daal::services::internal::daal_memcpy_s(&(splitColumn[pos]), sizeof(algorithmFPType) * nRows, data, sizeof(algorithmFPType) * nRows);

        pos += nRows;
    }
    DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);

    const size_t splitPosition       = (size_t)(totalNRows * ((double)leftBlocks / (leftBlocks + rightBlocks)));
    const algorithmFPType splitValue = findKthStatistic<algorithmFPType, cpu>(splitColumn, totalNRows, splitPosition);

    WriteRows<algorithmFPType, cpu> splitRows(ntSplit, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(splitRows);
    algorithmFPType * const split = splitRows.get();
    split[0]                      = splitValue;
    split[1]                      = (algorithmFPType)splitDim;

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep4Kernel<algorithmFPType, method, cpu>::compute(const DataCollection * dcPartialData, const DataCollection * dcPartialSplits,
                                                                     const DataCollection * dcPartialOrders, DataCollection * dcPartitionedData,
                                                                     DataCollection * dcPartitionedPartialOrders, const Parameter * par)
{
    const size_t leftBlocks  = par->leftBlocks;
    const size_t rightBlocks = par->rightBlocks;

    const size_t nFeatures = NumericTable::cast((*dcPartialData)[0])->getNumberOfColumns();
    const size_t nBlocks   = dcPartialSplits->size();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> partialSplitValuesArray(nBlocks);
    DAAL_CHECK_MALLOC(partialSplitValuesArray.get());
    algorithmFPType * const partialSplitValues = partialSplitValuesArray.get();

    int result = 0;

    size_t splitDim   = -1;
    size_t valuedPart = 0;
    for (size_t part = 0; part < nBlocks; part++)
    {
        NumericTablePtr ntPartialSplit = NumericTable::cast((*dcPartialSplits)[part]);
        ReadRows<algorithmFPType, cpu> partialSplitRows(ntPartialSplit.get(), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(partialSplitRows);
        const algorithmFPType * const partialSplit = partialSplitRows.get();

        if ((size_t)partialSplit[1] == -1)
        {
            continue;
        }

        partialSplitValues[valuedPart] = partialSplit[0];
        valuedPart++;

        DAAL_ASSERT(partialSplit[1] >= 0)
        if (part == 0)
        {
            splitDim = (size_t)partialSplit[1];
        }
        else
        {
            DAAL_ASSERT(splitDim == (size_t)partialSplit[1]);
        }
    }

    algorithmFPType splitValue = findKthStatistic<algorithmFPType, cpu>(partialSplitValues, valuedPart, valuedPart / 2);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, sizeof(int));

    TArray<int, cpu> partitionedDataNRowsArray(nBlocks);
    DAAL_CHECK_MALLOC(partitionedDataNRowsArray.get());
    int * const partitionedDataNRows = partitionedDataNRowsArray.get();

    TArray<int, cpu> partitionedDataPosArray(nBlocks);
    DAAL_CHECK_MALLOC(partitionedDataPosArray.get());
    int * const partitionedDataPos = partitionedDataPosArray.get();

    for (size_t i = 0; i < nBlocks; i++)
    {
        partitionedDataNRows[i] = 0;
        partitionedDataPos[i]   = 0;
    }

    size_t curLeftBlock  = 0;
    size_t curRightBlock = 0;

    for (size_t part = 0; part < dcPartialData->size(); part++)
    {
        NumericTablePtr ntData = NumericTable::cast((*dcPartialData)[part]);
        const size_t nRows     = ntData->getNumberOfRows();

        ReadColumns<algorithmFPType, cpu> dataColumns(ntData.get(), splitDim, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(dataColumns);
        const algorithmFPType * const data = dataColumns.get();

        for (size_t i = 0; i < nRows; i++)
        {
            size_t targetBlock;

            if (data[i] < splitValue)
            {
                targetBlock  = curLeftBlock;
                curLeftBlock = (curLeftBlock + 1) % leftBlocks;
            }
            else
            {
                targetBlock   = leftBlocks + curRightBlock;
                curRightBlock = (curRightBlock + 1) % rightBlocks;
            }

            partitionedDataNRows[targetBlock]++;
        }
    }

    for (size_t part = 0; part < nBlocks; part++)
    {
        if (partitionedDataNRows[part] == 0)
        {
            continue;
        }

        NumericTablePtr ntPartitionedData          = NumericTable::cast((*dcPartitionedData)[part]);
        NumericTablePtr ntPartitionedPartialOrders = NumericTable::cast((*dcPartitionedPartialOrders)[part]);

        DAAL_CHECK_STATUS_VAR(ntPartitionedData->resize(partitionedDataNRows[part]));
        DAAL_CHECK_STATUS_VAR(ntPartitionedPartialOrders->resize(partitionedDataNRows[part]));
    }

    const size_t defaultBlockSize = 256;

    curLeftBlock  = 0;
    curRightBlock = 0;
    for (size_t part = 0; part < dcPartialData->size(); part++)
    {
        NumericTablePtr ntData          = NumericTable::cast((*dcPartialData)[part]);
        NumericTablePtr ntPartialOrders = NumericTable::cast((*dcPartialOrders)[part]);

        const size_t nRows       = ntData->getNumberOfRows();
        const size_t nDataBlocks = nRows / defaultBlockSize + int(nRows % defaultBlockSize > 0);

        for (size_t block = 0; block < nDataBlocks; block++)
        {
            const size_t i1    = block * defaultBlockSize;
            const size_t i2    = (block + 1 == nDataBlocks ? nRows : i1 + defaultBlockSize);
            const size_t iSize = i2 - i1;

            ReadRows<algorithmFPType, cpu> dataRows(ntData.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(dataRows);
            const algorithmFPType * const data = dataRows.get();

            ReadRows<int, cpu> partialOrdersRows(ntPartialOrders.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(partialOrdersRows);
            const int * const partialOrders = partialOrdersRows.get();

            for (size_t i = 0; i < iSize; i++)
            {
                size_t targetBlock;

                if (data[i * nFeatures + splitDim] < splitValue)
                {
                    targetBlock  = curLeftBlock;
                    curLeftBlock = (curLeftBlock + 1) % leftBlocks;
                }
                else
                {
                    targetBlock   = leftBlocks + curRightBlock;
                    curRightBlock = (curRightBlock + 1) % rightBlocks;
                }

                NumericTablePtr ntPartitionedData          = NumericTable::cast((*dcPartitionedData)[targetBlock]);
                NumericTablePtr ntPartitionedPartialOrders = NumericTable::cast((*dcPartitionedPartialOrders)[targetBlock]);

                const size_t targetPosition = partitionedDataPos[targetBlock];
                partitionedDataPos[targetBlock]++;

                WriteRows<algorithmFPType, cpu> partitionedDataRows(ntPartitionedData.get(), targetPosition, 1);
                DAAL_CHECK_BLOCK_STATUS(partitionedDataRows);
                algorithmFPType * const partitionedData = partitionedDataRows.get();

                WriteRows<int, cpu> partitionedPartialOrdersRows(ntPartitionedPartialOrders.get(), targetPosition, 1);
                DAAL_CHECK_BLOCK_STATUS(partitionedPartialOrdersRows);
                int * const partitionedPartialOrders = partitionedPartialOrdersRows.get();

                result |= daal::services::internal::daal_memcpy_s(partitionedData, sizeof(algorithmFPType) * nFeatures, &(data[i * nFeatures]),
                                                                  sizeof(algorithmFPType) * nFeatures);
                result |=
                    daal::services::internal::daal_memcpy_s(partitionedPartialOrders, sizeof(int) * 2, &(partialOrders[i * 2]), sizeof(int) * 2);
            }
        }
    }
    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep5Kernel<algorithmFPType, method, cpu>::compute(const DataCollection * dcPartialData,
                                                                     const DataCollection * dcPartialBoundingBoxes,
                                                                     DataCollection * dcPartitionedHaloData,
                                                                     DataCollection * dcPartitionedHaloDataIndices, const Parameter * par)
{
    const size_t blockIndex       = par->blockIndex;
    const size_t nBlocks          = par->nBlocks;
    const algorithmFPType epsilon = par->epsilon;

    const size_t nFeatures        = NumericTable::cast((*dcPartialData)[0])->getNumberOfColumns();
    const size_t defaultBlockSize = 256;

    int result = 0;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, sizeof(int));

    TArray<int, cpu> partitionedHaloDataNRowsArray(nBlocks);
    DAAL_CHECK_MALLOC(partitionedHaloDataNRowsArray.get());
    int * const partitionedHaloDataNRows = partitionedHaloDataNRowsArray.get();

    TArray<int, cpu> partitionedHaloDataPosArray(nBlocks);
    DAAL_CHECK_MALLOC(partitionedHaloDataPosArray.get());
    int * const partitionedHaloDataPos = partitionedHaloDataPosArray.get();

    for (size_t extPart = 0; extPart < nBlocks; extPart++)
    {
        partitionedHaloDataNRows[extPart] = 0;
        partitionedHaloDataPos[extPart]   = 0;
    }

    for (size_t part = 0; part < dcPartialData->size(); part++)
    {
        NumericTablePtr ntData   = NumericTable::cast((*dcPartialData)[part]);
        const size_t nRows       = ntData->getNumberOfRows();
        const size_t nDataBlocks = nRows / defaultBlockSize + int(nRows % defaultBlockSize > 0);

        for (size_t block = 0; block < nDataBlocks; block++)
        {
            const size_t i1    = block * defaultBlockSize;
            const size_t i2    = (block + 1 == nDataBlocks ? nRows : i1 + defaultBlockSize);
            const size_t iSize = i2 - i1;

            ReadRows<algorithmFPType, cpu> dataRows(ntData.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(dataRows);
            const algorithmFPType * const data = dataRows.get();

            for (size_t extPart = 0; extPart < nBlocks; extPart++)
            {
                if (extPart == blockIndex)
                {
                    continue;
                }

                NumericTablePtr ntPartialBoundingBox = NumericTable::cast((*dcPartialBoundingBoxes)[extPart]);
                ReadRows<algorithmFPType, cpu> partialBoundingBoxRows(ntPartialBoundingBox.get(), 0, 2);
                DAAL_CHECK_BLOCK_STATUS(partialBoundingBoxRows);
                const algorithmFPType * const partialBoundingBox = partialBoundingBoxRows.get();

                for (size_t i = 0; i < iSize; i++)
                {
                    int isInside = 1;
                    for (size_t j = 0; j < nFeatures && isInside; j++)
                    {
                        isInside &= int(partialBoundingBox[j] - epsilon <= data[i * nFeatures + j]
                                        && partialBoundingBox[j + nFeatures] + epsilon >= data[i * nFeatures + j]);
                    }
                    partitionedHaloDataNRows[extPart] += isInside;
                }
            }
        }
    }

    for (size_t extPart = 0; extPart < nBlocks; extPart++)
    {
        if (partitionedHaloDataNRows[extPart] == 0)
        {
            continue;
        }

        NumericTablePtr ntPartitionedHaloData        = NumericTable::cast((*dcPartitionedHaloData)[extPart]);
        NumericTablePtr ntPartitionedHaloDataIndices = NumericTable::cast((*dcPartitionedHaloDataIndices)[extPart]);

        DAAL_CHECK_STATUS_VAR(ntPartitionedHaloData->resize(partitionedHaloDataNRows[extPart]));
        DAAL_CHECK_STATUS_VAR(ntPartitionedHaloDataIndices->resize(partitionedHaloDataNRows[extPart]));
    }

    size_t ind = 0;

    for (size_t part = 0; part < dcPartialData->size(); part++)
    {
        NumericTablePtr ntData   = NumericTable::cast((*dcPartialData)[part]);
        const size_t nRows       = ntData->getNumberOfRows();
        const size_t nDataBlocks = nRows / defaultBlockSize + int(nRows % defaultBlockSize > 0);

        for (size_t block = 0; block < nDataBlocks; block++)
        {
            const size_t i1    = block * defaultBlockSize;
            const size_t i2    = (block + 1 == nDataBlocks ? nRows : i1 + defaultBlockSize);
            const size_t iSize = i2 - i1;

            ReadRows<algorithmFPType, cpu> dataRows(ntData.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(dataRows);
            const algorithmFPType * const data = dataRows.get();

            for (size_t extPart = 0; extPart < nBlocks; extPart++)
            {
                if (extPart == blockIndex)
                {
                    continue;
                }

                NumericTablePtr ntPartialBoundingBox = NumericTable::cast((*dcPartialBoundingBoxes)[extPart]);

                ReadRows<algorithmFPType, cpu> partialBoundingBoxRows(ntPartialBoundingBox.get(), 0, 2);
                DAAL_CHECK_BLOCK_STATUS(partialBoundingBoxRows);
                const algorithmFPType * const partialBoundingBox = partialBoundingBoxRows.get();

                NumericTablePtr ntPartitionedHaloData        = NumericTable::cast((*dcPartitionedHaloData)[extPart]);
                NumericTablePtr ntPartitionedHaloDataIndices = NumericTable::cast((*dcPartitionedHaloDataIndices)[extPart]);

                for (size_t i = 0; i < iSize; i++)
                {
                    int isInside = 1;
                    for (size_t j = 0; j < nFeatures && isInside; j++)
                    {
                        isInside &= int(partialBoundingBox[j] - epsilon <= data[i * nFeatures + j]
                                        && partialBoundingBox[j + nFeatures] + epsilon >= data[i * nFeatures + j]);
                    }
                    if (isInside)
                    {
                        WriteRows<algorithmFPType, cpu> partitionedHaloDataRows(ntPartitionedHaloData.get(), partitionedHaloDataPos[extPart], 1);
                        DAAL_CHECK_BLOCK_STATUS(partitionedHaloDataRows);
                        algorithmFPType * const partitionedHaloData = partitionedHaloDataRows.get();

                        WriteRows<int, cpu> partitionedHaloDataIndicesRows(ntPartitionedHaloDataIndices.get(), partitionedHaloDataPos[extPart], 1);
                        DAAL_CHECK_BLOCK_STATUS(partitionedHaloDataIndicesRows);
                        int * const partitionedHaloDataIndices = partitionedHaloDataIndicesRows.get();

                        result |= daal::services::internal::daal_memcpy_s(partitionedHaloData, sizeof(algorithmFPType) * nFeatures,
                                                                          &(data[i * nFeatures]), sizeof(algorithmFPType) * nFeatures);
                        partitionedHaloDataIndices[0] = ind + i;

                        partitionedHaloDataPos[extPart]++;
                    }
                }
            }

            ind += iSize;
        }
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep6Kernel<algorithmFPType, method, cpu>::processNeighborhood(size_t clusterId, size_t startObs, int * const clusterStructure,
                                                                                 const Neighborhood<algorithmFPType, cpu> & neigh,
                                                                                 Queue<size_t, cpu> & qu)
{
    DAAL_CHECK(clusterId <= services::internal::MaxVal<int>::get(), ErrorIncorrectNumberOfPartialClusters)
    DAAL_CHECK(startObs <= services::internal::MaxVal<int>::get(), ErrorIncorrectNumberOfPartialClusters)
    for (size_t j = 0; j < neigh.size(); j++)
    {
        const size_t nextObs = neigh.get(j);
        if (clusterStructure[nextObs * 4 + 0] == noise)
        {
            clusterStructure[nextObs * 4 + 0] = (int)clusterId;
            clusterStructure[nextObs * 4 + 3] = (int)startObs;
        }
        else if (clusterStructure[nextObs * 4 + 0] == undefined)
        {
            clusterStructure[nextObs * 4 + 0] = (int)clusterId;
            clusterStructure[nextObs * 4 + 3] = (int)startObs;
            DAAL_CHECK_STATUS_VAR(qu.push(nextObs));
        }
    }

    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep6Kernel<algorithmFPType, method, cpu>::processHaloNeighborhood(size_t startObs, int * const haloAssignments,
                                                                                     const int * const haloBlocks, const int * const haloDataIndices,
                                                                                     const Neighborhood<algorithmFPType, cpu> & haloNeigh,
                                                                                     Vector<int, cpu> * const queries)
{
    for (size_t j = 0; j < haloNeigh.size(); j++)
    {
        const size_t haloObs = haloNeigh.get(j);
        if (haloAssignments[haloObs] != 0)
        {
            continue;
        }
        haloAssignments[haloObs] = 1;

        const size_t haloIndex = haloDataIndices[haloObs];
        const size_t haloBlock = haloBlocks[haloObs];

        queries[haloBlock].push_back(haloIndex);
        queries[haloBlock].push_back(startObs);
    }

    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep6Kernel<algorithmFPType, method, cpu>::generateQueries(size_t blockIndex, size_t nBlocks, Vector<int, cpu> * const queries,
                                                                             DataCollection * const dcQueries, bool & totalFinishedFlag)
{
    for (size_t part = 0; part < nBlocks; part++)
    {
        const size_t nCurQueries = queries[part].size() / 2;

        if (nCurQueries == 0)
        {
            continue;
        }
        totalFinishedFlag = false;

        NumericTablePtr ntCurQueries = NumericTable::cast((*dcQueries)[part]);
        DAAL_CHECK_STATUS_VAR(ntCurQueries->resize(nCurQueries));

        WriteRows<int, cpu> curQueriesRows(ntCurQueries.get(), 0, nCurQueries);
        DAAL_CHECK_BLOCK_STATUS(curQueriesRows);
        int * const curQueries = curQueriesRows.get();

        for (size_t i = 0; i < nCurQueries; i++)
        {
            curQueries[i * 3 + 0] = queries[part][i * 2];
            curQueries[i * 3 + 1] = blockIndex;
            curQueries[i * 3 + 2] = queries[part][i * 2 + 1];
        }
    }

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
template <typename T>
Status DBSCANDistrStep6Kernel<algorithmFPType, method, cpu>::repackIntoSingleNT(const DataCollection * dcInput, NumericTablePtr & ntOutput)
{
    size_t nRows     = 0;
    size_t nFeatures = 0;
    int result       = 0;

    for (size_t i = 0; i < dcInput->size(); i++)
    {
        NumericTablePtr ntInput = NumericTable::cast((*dcInput)[i]);
        const size_t nInputRows = ntInput->getNumberOfRows();
        nRows += nInputRows;

        if (i == 0)
        {
            nFeatures = ntInput->getNumberOfColumns();
        }
        else
        {
            DAAL_ASSERT(nFeatures == ntInput->getNumberOfColumns());
        }
    }

    Status s;

    if (nRows == 0)
    {
        ntOutput = HomogenNumericTable<T>::create(nFeatures, nRows, NumericTable::notAllocate, &s);

        return s;
    }

    ntOutput = HomogenNumericTable<T>::create(nFeatures, nRows, NumericTable::doAllocate, &s);
    DAAL_CHECK_STATUS_VAR(s);

    size_t pos = 0;
    for (size_t i = 0; i < dcInput->size(); i++)
    {
        NumericTablePtr ntInput = NumericTable::cast((*dcInput)[i]);
        const size_t nInputRows = ntInput->getNumberOfRows();

        const size_t defaultBlockSize = 256;
        const size_t nDataBlocks      = nInputRows / defaultBlockSize + int(nInputRows % defaultBlockSize > 0);

        for (size_t block = 0; block < nDataBlocks; block++)
        {
            const size_t i1    = block * defaultBlockSize;
            const size_t i2    = (block + 1 == nDataBlocks ? nInputRows : i1 + defaultBlockSize);
            const size_t iSize = i2 - i1;

            ReadRows<T, cpu> inRows(ntInput.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(inRows);
            const T * const in = inRows.get();

            WriteRows<T, cpu> outRows(ntOutput.get(), pos + i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(outRows);
            T * const out = outRows.get();

            result |= daal::services::internal::daal_memcpy_s(out, sizeof(T) * iSize * nFeatures, in, sizeof(T) * iSize * nFeatures);
        }

        pos += nInputRows;
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep6Kernel<algorithmFPType, method, cpu>::computeNoMemSave(const DataCollection * dcPartialData, const DataCollection * dcHaloData,
                                                                              const DataCollection * dcHaloDataIndices,
                                                                              const DataCollection * dcHaloBlocks, NumericTable * ntClusterStructure,
                                                                              NumericTable * ntFinishedFlag, NumericTable * ntNClusters,
                                                                              DataCollection * dcQueries, const Parameter * par)
{
    NumericTablePtr ntWeights;     // currently unsupported
    NumericTablePtr ntHaloWeights; // currently unsupported

    const algorithmFPType epsilon         = par->epsilon;
    const algorithmFPType minObservations = par->minObservations;
    const algorithmFPType minkowskiPower  = (algorithmFPType)2.0;

    NumericTablePtr ntData;
    NumericTablePtr ntHaloData;
    NumericTablePtr ntHaloDataIndices;

    DAAL_CHECK_STATUS_VAR(repackIntoSingleNT<algorithmFPType>(dcPartialData, ntData));
    DAAL_CHECK_STATUS_VAR(repackIntoSingleNT<algorithmFPType>(dcHaloData, ntHaloData));
    DAAL_CHECK_STATUS_VAR(repackIntoSingleNT<int>(dcHaloDataIndices, ntHaloDataIndices));

    const size_t nRows      = ntData->getNumberOfRows();
    const size_t nHaloRows  = ntHaloData->getNumberOfRows();
    const size_t blockIndex = par->blockIndex;
    const size_t nBlocks    = par->nBlocks;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nHaloRows, sizeof(int));

    TArray<int, cpu> haloBlocksArray(nHaloRows);
    if (nHaloRows)
    {
        DAAL_CHECK_MALLOC(haloBlocksArray.get());
    }
    int * const haloBlocks = haloBlocksArray.get();

    size_t pos = 0;

    for (size_t part = 0; part < dcHaloData->size(); part++)
    {
        NumericTablePtr ntSingleHaloData = NumericTable::cast((*dcHaloData)[part]);
        NumericTablePtr ntHaloBlock      = NumericTable::cast((*dcHaloBlocks)[part]);
        const size_t nRows               = ntSingleHaloData->getNumberOfRows();
        const size_t partId              = ntHaloBlock->getValue<int>(0, 0);
        for (size_t i = 0; i < nRows; i++)
        {
            haloBlocks[i + pos] = partId;
        }
        pos += nRows;
    }

    ReadRows<int, cpu> haloDataIndicesRows(ntHaloDataIndices.get(), 0, nHaloRows);
    if (nHaloRows)
    {
        DAAL_CHECK_BLOCK_STATUS(haloDataIndicesRows);
    }
    const int * const haloDataIndices = haloDataIndicesRows.get();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, sizeof(Neighborhood<algorithmFPType, cpu>));

    TArray<Neighborhood<algorithmFPType, cpu>, cpu> neighs(nRows);
    if (nRows)
    {
        DAAL_CHECK_MALLOC(neighs.get());
    }
    TArray<Neighborhood<algorithmFPType, cpu>, cpu> haloNeighs(nRows);
    if (nRows)
    {
        DAAL_CHECK_MALLOC(haloNeighs.get());
    }

    TArray<int, cpu> haloAssignmentsArray(nHaloRows);
    if (nHaloRows)
    {
        DAAL_CHECK_MALLOC(haloAssignmentsArray.get());
    }
    int * const haloAssignments = haloAssignmentsArray.get();

    for (size_t i = 0; i < nHaloRows; i++)
    {
        haloAssignments[i] = 0;
    }

    NeighborhoodEngine<method, algorithmFPType, cpu> nEngine(ntData.get(), ntData.get(), ntWeights.get(), epsilon, minkowskiPower);
    DAAL_CHECK_STATUS_VAR(nEngine.queryFull(neighs.get()));

    NeighborhoodEngine<method, algorithmFPType, cpu> nHaloEngine(ntData.get(), ntHaloData.get(), ntHaloWeights.get(), epsilon, minkowskiPower);
    DAAL_CHECK_STATUS_VAR(nHaloEngine.queryFull(haloNeighs.get()));

    if (nRows)
    {
        DAAL_CHECK_STATUS_VAR(ntClusterStructure->resize(nRows))
    }

    WriteRows<int, cpu> clusterStructureRows(ntClusterStructure, 0, nRows);
    if (nRows)
    {
        DAAL_CHECK_BLOCK_STATUS(clusterStructureRows);
    }
    int * const clusterStructure = clusterStructureRows.get();

    for (size_t i = 0; i < nRows; i++)
    {
        clusterStructure[i * 4 + 0] = undefined;  // current assignment of observation
        clusterStructure[i * 4 + 1] = 0;          // core-observation flag: = 1 for core-observations, = 0 - otherwise
        clusterStructure[i * 4 + 2] = blockIndex; // partition of parent observation in union-find structure
        clusterStructure[i * 4 + 3] = -1;         // index of parent observation in union-find structure
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, sizeof(Vector<int, cpu>));

    TArray<Vector<int, cpu>, cpu> queriesArray(nBlocks);
    DAAL_CHECK_MALLOC(queriesArray.get());
    Vector<int, cpu> * queries = queriesArray.get();

    size_t nClusters = 0;
    Queue<size_t, cpu> qu;

    for (size_t i = 0; i < nRows; i++)
    {
        if (clusterStructure[i * 4 + 0] != undefined)
        {
            continue;
        }

        Neighborhood<algorithmFPType, cpu> & curNeigh     = neighs[i];
        Neighborhood<algorithmFPType, cpu> & curHaloNeigh = haloNeighs[i];

        if (curNeigh.weight() + curHaloNeigh.weight() < minObservations)
        {
            clusterStructure[i * 4 + 0] = noise;
            continue;
        }

        const size_t clusterId = nClusters;
        nClusters++;

        DAAL_ASSERT(clusterId <= services::internal::MaxVal<int>::get())
        DAAL_ASSERT(blockIndex <= services::internal::MaxVal<int>::get())
        DAAL_ASSERT(i <= services::internal::MaxVal<int>::get())
        clusterStructure[i * 4 + 0] = (int)(clusterId);
        clusterStructure[i * 4 + 1] = 1;
        clusterStructure[i * 4 + 2] = (int)blockIndex;
        clusterStructure[i * 4 + 3] = (int)i;

        qu.reset();

        DAAL_CHECK_STATUS_VAR(processNeighborhood(clusterId, i, clusterStructure, curNeigh, qu));
        DAAL_CHECK_STATUS_VAR(processHaloNeighborhood(i, haloAssignments, haloBlocks, haloDataIndices, curHaloNeigh, queries));

        while (!qu.empty())
        {
            const size_t curObs = qu.pop();

            Neighborhood<algorithmFPType, cpu> & curNeigh     = neighs[curObs];
            Neighborhood<algorithmFPType, cpu> & curHaloNeigh = haloNeighs[curObs];

            if (curNeigh.weight() + curHaloNeigh.weight() < minObservations) continue;

            clusterStructure[curObs * 4 + 1] = 1;

            DAAL_CHECK_STATUS_VAR(processNeighborhood(clusterId, i, clusterStructure, curNeigh, qu));
            DAAL_CHECK_STATUS_VAR(processHaloNeighborhood(i, haloAssignments, haloBlocks, haloDataIndices, curHaloNeigh, queries));
        }
    }

    {
        WriteRows<int, cpu> nClustersRows(ntNClusters, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(nClustersRows);
        int * const outNClusters = nClustersRows.get();
        outNClusters[0]          = nClusters;
    }

    bool totalFinishedFlag = true;
    DAAL_CHECK_STATUS_VAR(generateQueries(blockIndex, nBlocks, queries, dcQueries, totalFinishedFlag));

    {
        WriteRows<int, cpu> finishedFlagRows(ntFinishedFlag, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(finishedFlagRows);
        int * const finishedFlag = finishedFlagRows.get();
        finishedFlag[0]          = int(totalFinishedFlag);
    }

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep6Kernel<algorithmFPType, method, cpu>::computeMemSave(const DataCollection * dcPartialData, const DataCollection * dcHaloData,
                                                                            const DataCollection * dcHaloDataIndices,
                                                                            const DataCollection * dcHaloBlocks, NumericTable * ntClusterStructure,
                                                                            NumericTable * ntFinishedFlag, NumericTable * ntNClusters,
                                                                            DataCollection * dcQueries, const Parameter * par)
{
    NumericTablePtr ntWeights;     // currently unsupported
    NumericTablePtr ntHaloWeights; // currently unsupported

    const algorithmFPType epsilon         = par->epsilon;
    const algorithmFPType minObservations = par->minObservations;
    const algorithmFPType minkowskiPower  = (algorithmFPType)2.0;

    NumericTablePtr ntData;
    NumericTablePtr ntHaloData;
    NumericTablePtr ntHaloDataIndices;

    DAAL_CHECK_STATUS_VAR(repackIntoSingleNT<algorithmFPType>(dcPartialData, ntData));
    DAAL_CHECK_STATUS_VAR(repackIntoSingleNT<algorithmFPType>(dcHaloData, ntHaloData));
    DAAL_CHECK_STATUS_VAR(repackIntoSingleNT<int>(dcHaloDataIndices, ntHaloDataIndices));

    const size_t nRows      = ntData->getNumberOfRows();
    const size_t nHaloRows  = ntHaloData->getNumberOfRows();
    const size_t blockIndex = par->blockIndex;
    const size_t nBlocks    = par->nBlocks;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nHaloRows, sizeof(int));

    TArray<int, cpu> haloBlocksArray(nHaloRows);
    if (nHaloRows)
    {
        DAAL_CHECK_MALLOC(haloBlocksArray.get());
    }
    int * const haloBlocks = haloBlocksArray.get();

    size_t pos = 0;

    for (size_t part = 0; part < dcHaloData->size(); part++)
    {
        NumericTablePtr ntSingleHaloData = NumericTable::cast((*dcHaloData)[part]);
        NumericTablePtr ntHaloBlock      = NumericTable::cast((*dcHaloBlocks)[part]);
        const size_t nRows               = ntSingleHaloData->getNumberOfRows();
        const size_t partId              = ntHaloBlock->getValue<int>(0, 0);
        for (size_t i = 0; i < nRows; i++)
        {
            haloBlocks[i + pos] = partId;
        }
        pos += nRows;
    }

    ReadRows<int, cpu> haloDataIndicesRows(ntHaloDataIndices.get(), 0, nHaloRows);
    if (nHaloRows)
    {
        DAAL_CHECK_BLOCK_STATUS(haloDataIndicesRows);
    }
    const int * const haloDataIndices = haloDataIndicesRows.get();

    const size_t prefetchBlockSize = __DBSCAN_PREFETCHED_NEIGHBORHOODS_COUNT;
    TArray<Neighborhood<algorithmFPType, cpu>, cpu> prefetchedNeighs(prefetchBlockSize);
    DAAL_CHECK_MALLOC(prefetchedNeighs.get());
    TArray<Neighborhood<algorithmFPType, cpu>, cpu> prefetchedHaloNeighs(prefetchBlockSize);
    DAAL_CHECK_MALLOC(prefetchedHaloNeighs.get());

    TArray<int, cpu> haloAssignmentsArray(nHaloRows);
    if (nHaloRows)
    {
        DAAL_CHECK_MALLOC(haloAssignmentsArray.get());
    }
    int * const haloAssignments = haloAssignmentsArray.get();

    for (size_t i = 0; i < nHaloRows; i++)
    {
        haloAssignments[i] = 0;
    }

    NeighborhoodEngine<method, algorithmFPType, cpu> nEngine(ntData.get(), ntData.get(), ntWeights.get(), epsilon, minkowskiPower);
    NeighborhoodEngine<method, algorithmFPType, cpu> nHaloEngine(ntData.get(), ntHaloData.get(), ntHaloWeights.get(), epsilon, minkowskiPower);

    if (nRows)
    {
        DAAL_CHECK_STATUS_VAR(ntClusterStructure->resize(nRows))
    }

    WriteRows<int, cpu> clusterStructureRows(ntClusterStructure, 0, nRows);
    if (nRows)
    {
        DAAL_CHECK_BLOCK_STATUS(clusterStructureRows);
    }
    int * const clusterStructure = clusterStructureRows.get();

    for (size_t i = 0; i < nRows; i++)
    {
        clusterStructure[i * 4 + 0] = undefined;  // current assignment of observation
        clusterStructure[i * 4 + 1] = 0;          // core-observation flag: = 1 for core-observations, = 0 - otherwise
        clusterStructure[i * 4 + 2] = blockIndex; // partition of parent observation in union-find structure
        clusterStructure[i * 4 + 3] = -1;         // index of parent observation in union-find structure
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, sizeof(Vector<int, cpu>));

    TArray<Vector<int, cpu>, cpu> queriesArray(nBlocks);
    DAAL_CHECK_MALLOC(queriesArray.get());
    Vector<int, cpu> * queries = queriesArray.get();

    size_t nClusters = 0;
    Queue<size_t, cpu> qu;

    for (size_t i = 0; i < nRows; i++)
    {
        if (clusterStructure[i * 4 + 0] != undefined)
        {
            continue;
        }

        Neighborhood<algorithmFPType, cpu> curNeigh;
        Neighborhood<algorithmFPType, cpu> curHaloNeigh;
        nEngine.query(&i, 1, &curNeigh);
        nHaloEngine.query(&i, 1, &curHaloNeigh);

        if (curNeigh.weight() + curHaloNeigh.weight() < minObservations)
        {
            clusterStructure[i * 4 + 0] = noise;
            continue;
        }

        const size_t clusterId = nClusters;
        nClusters++;

        DAAL_ASSERT(clusterId <= services::internal::MaxVal<int>::get())
        DAAL_ASSERT(blockIndex <= services::internal::MaxVal<int>::get())
        DAAL_ASSERT(i <= services::internal::MaxVal<int>::get())
        clusterStructure[i * 4 + 0] = (int)(clusterId);
        clusterStructure[i * 4 + 1] = 1;
        clusterStructure[i * 4 + 2] = (int)blockIndex;
        clusterStructure[i * 4 + 3] = (int)i;

        qu.reset();

        DAAL_CHECK_STATUS_VAR(processNeighborhood(clusterId, i, clusterStructure, curNeigh, qu));
        DAAL_CHECK_STATUS_VAR(processHaloNeighborhood(i, haloAssignments, haloBlocks, haloDataIndices, curHaloNeigh, queries));

        size_t firstPrefetchedPos = 0;
        size_t lastPrefetchedPos  = 0;

        while (!qu.empty())
        {
            const size_t quPos  = qu.head();
            const size_t curObs = qu.pop();

            if (quPos >= lastPrefetchedPos)
            {
                firstPrefetchedPos = lastPrefetchedPos = quPos;
            }

            if (lastPrefetchedPos - firstPrefetchedPos < prefetchBlockSize && lastPrefetchedPos < qu.tail())
            {
                const size_t nextPrefetchPos =
                    firstPrefetchedPos + prefetchBlockSize < qu.tail() ? firstPrefetchedPos + prefetchBlockSize : qu.tail();
                DAAL_CHECK_STATUS_VAR(nEngine.query(qu.getInternalPtr(lastPrefetchedPos), nextPrefetchPos - lastPrefetchedPos,
                                                    &(prefetchedNeighs[lastPrefetchedPos - firstPrefetchedPos]), true));
                DAAL_CHECK_STATUS_VAR(nHaloEngine.query(qu.getInternalPtr(lastPrefetchedPos), nextPrefetchPos - lastPrefetchedPos,
                                                        &(prefetchedHaloNeighs[lastPrefetchedPos - firstPrefetchedPos]), true));
                lastPrefetchedPos = nextPrefetchPos;
            }

            Neighborhood<algorithmFPType, cpu> & curNeigh     = prefetchedNeighs[quPos - firstPrefetchedPos];
            Neighborhood<algorithmFPType, cpu> & curHaloNeigh = prefetchedHaloNeighs[quPos - firstPrefetchedPos];

            if (curNeigh.weight() + curHaloNeigh.weight() < minObservations) continue;

            clusterStructure[curObs * 4 + 1] = 1;

            DAAL_CHECK_STATUS_VAR(processNeighborhood(clusterId, i, clusterStructure, curNeigh, qu));
            DAAL_CHECK_STATUS_VAR(processHaloNeighborhood(i, haloAssignments, haloBlocks, haloDataIndices, curHaloNeigh, queries));
        }
    }

    {
        WriteRows<int, cpu> nClustersRows(ntNClusters, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(nClustersRows);
        int * const outNClusters = nClustersRows.get();
        outNClusters[0]          = nClusters;
    }

    bool totalFinishedFlag = true;
    DAAL_CHECK_STATUS_VAR(generateQueries(blockIndex, nBlocks, queries, dcQueries, totalFinishedFlag));

    {
        WriteRows<int, cpu> finishedFlagRows(ntFinishedFlag, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(finishedFlagRows);
        int * const finishedFlag = finishedFlagRows.get();
        finishedFlag[0]          = int(totalFinishedFlag);
    }

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep7Kernel<algorithmFPType, method, cpu>::compute(const DataCollection * dcPartialFinishedFlags, NumericTable * ntFinishedFlag)
{
    bool totalFinishedFlag = true;
    for (size_t i = 0; i < dcPartialFinishedFlags->size(); i++)
    {
        NumericTablePtr ntCurFinishedFlag = NumericTable::cast((*dcPartialFinishedFlags)[i]);
        totalFinishedFlag &= int(ntCurFinishedFlag->getValue<int>(0, 0) == 1);
    }

    WriteRows<int, cpu> finishedFlagRows(ntFinishedFlag, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(finishedFlagRows);
    int * const finishedFlag = finishedFlagRows.get();
    finishedFlag[0]          = int(totalFinishedFlag);

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep8Kernel<algorithmFPType, method, cpu>::addQuery(Vector<int, cpu> * const queries, size_t dstBlockIndex, size_t dstId,
                                                                      size_t srcBlockIndex, size_t srcId)
{
    DAAL_CHECK_STATUS_VAR(queries[dstBlockIndex].push_back(dstId));
    DAAL_CHECK_STATUS_VAR(queries[dstBlockIndex].push_back(srcBlockIndex));
    DAAL_CHECK_STATUS_VAR(queries[dstBlockIndex].push_back(srcId));

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep8Kernel<algorithmFPType, method, cpu>::sortQueries(Vector<int, cpu> & queries)
{
    int i, ir, j, k, jstack = -1, l = 0;
    int a, b, c;
    const int M = 7, NSTACK = 128;
    int istack[NSTACK];

    ir = (queries.size() / 3) - 1;

    for (;;)
    {
        if (ir - l < M)
        {
            for (j = l + 1; j <= ir; j++)
            {
                a = queries[j * 3 + 0];
                b = queries[j * 3 + 1];
                c = queries[j * 3 + 2];

                for (i = j - 1; i >= l; i--)
                {
                    if (queries[i * 3 + 0] < a
                        || (queries[i * 3 + 0] == a
                            && (queries[i * 3 + 1] < b || (queries[i * 3 + 0] == a && queries[i * 3 + 1] == b && queries[i * 3 + 2] <= c))))
                    {
                        break;
                    }
                    queries[3 * (i + 1) + 0] = queries[3 * i + 0];
                    queries[3 * (i + 1) + 1] = queries[3 * i + 1];
                    queries[3 * (i + 1) + 2] = queries[3 * i + 2];
                }

                queries[3 * (i + 1) + 0] = a;
                queries[3 * (i + 1) + 1] = b;
                queries[3 * (i + 1) + 2] = c;
            }

            if (jstack < 0)
            {
                break;
            }

            ir = istack[jstack--];
            l  = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;

            daal::services::internal::swap<cpu, int>(queries[3 * k + 0], queries[3 * (l + 1) + 0]);
            daal::services::internal::swap<cpu, int>(queries[3 * k + 1], queries[3 * (l + 1) + 1]);
            daal::services::internal::swap<cpu, int>(queries[3 * k + 2], queries[3 * (l + 1) + 2]);

            if (queries[3 * l + 0] > queries[3 * ir + 0]
                || (queries[3 * l + 0] == queries[3 * ir + 0]
                    && (queries[3 * l + 1] > queries[3 * ir + 1]
                        || (queries[3 * l + 0] == queries[3 * ir + 0] && queries[3 * l + 1] == queries[3 * ir + 1]
                            && queries[3 * l + 2] > queries[3 * ir + 2]))))
            {
                daal::services::internal::swap<cpu, int>(queries[3 * l + 0], queries[3 * ir + 0]);
                daal::services::internal::swap<cpu, int>(queries[3 * l + 1], queries[3 * ir + 1]);
                daal::services::internal::swap<cpu, int>(queries[3 * l + 2], queries[3 * ir + 2]);
            }

            if (queries[3 * (l + 1) + 0] > queries[3 * ir + 0]
                || (queries[3 * (l + 1) + 0] == queries[3 * ir + 0]
                    && (queries[3 * (l + 1) + 1] > queries[3 * ir + 1]
                        || (queries[3 * (l + 1) + 0] == queries[3 * ir + 0] && queries[3 * (l + 1) + 1] == queries[3 * ir + 1]
                            && queries[3 * (l + 1) + 2] > queries[3 * ir + 2]))))
            {
                daal::services::internal::swap<cpu, int>(queries[3 * (l + 1) + 0], queries[3 * ir + 0]);
                daal::services::internal::swap<cpu, int>(queries[3 * (l + 1) + 1], queries[3 * ir + 1]);
                daal::services::internal::swap<cpu, int>(queries[3 * (l + 1) + 2], queries[3 * ir + 2]);
            }

            if (queries[3 * l + 0] > queries[3 * (l + 1) + 0]
                || (queries[3 * l + 0] == queries[3 * (l + 1) + 0]
                    && (queries[3 * l + 1] > queries[3 * (l + 1) + 1]
                        || (queries[3 * l + 0] == queries[3 * (l + 1) + 0] && queries[3 * l + 1] == queries[3 * (l + 1) + 1]
                            && queries[3 * l + 2] > queries[3 * (l + 1) + 2]))))
            {
                daal::services::internal::swap<cpu, int>(queries[3 * l + 0], queries[3 * (l + 1) + 0]);
                daal::services::internal::swap<cpu, int>(queries[3 * l + 1], queries[3 * (l + 1) + 1]);
                daal::services::internal::swap<cpu, int>(queries[3 * l + 2], queries[3 * (l + 1) + 2]);
            }

            i = l + 1;
            j = ir;
            a = queries[3 * (l + 1) + 0];
            b = queries[3 * (l + 1) + 1];
            c = queries[3 * (l + 1) + 2];
            for (;;)
            {
                do
                {
                    i++;
                } while (queries[3 * i + 0] < a
                         || (queries[3 * i + 0] == a
                             && (queries[3 * i + 1] < b || (queries[3 * i + 0] == a && queries[3 * i + 1] == b && queries[3 * i + 2] < c))));

                do
                {
                    j--;
                } while (queries[3 * j + 0] > a
                         || (queries[3 * j + 0] == a
                             && (queries[3 * j + 1] > b || (queries[3 * j + 0] == a && queries[3 * j + 1] == b && queries[3 * j + 2] > c))));

                if (j < i)
                {
                    break;
                }

                daal::services::internal::swap<cpu, int>(queries[3 * i + 0], queries[3 * j + 0]);
                daal::services::internal::swap<cpu, int>(queries[3 * i + 1], queries[3 * j + 1]);
                daal::services::internal::swap<cpu, int>(queries[3 * i + 2], queries[3 * j + 2]);
            }

            queries[3 * (l + 1) + 0] = queries[3 * j + 0];
            queries[3 * (l + 1) + 1] = queries[3 * j + 1];
            queries[3 * (l + 1) + 2] = queries[3 * j + 2];

            queries[3 * j + 0] = a;
            queries[3 * j + 1] = b;
            queries[3 * j + 2] = c;

            jstack += 2;

            if (ir - i + 1 >= j - l)
            {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            }
            else
            {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep8Kernel<algorithmFPType, method, cpu>::removeDuplicateQueries(Vector<int, cpu> & queries, size_t & nUniqueQueries)
{
    DAAL_CHECK_STATUS_VAR(sortQueries(queries));

    const size_t nQueries = queries.size() / 3;
    nUniqueQueries        = 0;

    for (size_t i = 0; i < nQueries; i++)
    {
        if (nUniqueQueries == 0 || queries[3 * i + 0] != queries[3 * (nUniqueQueries - 1) + 0]
            || queries[3 * i + 1] != queries[3 * (nUniqueQueries - 1) + 1] || queries[3 * i + 2] != queries[3 * (nUniqueQueries - 1) + 2])
        {
            queries[3 * nUniqueQueries + 0] = queries[3 * i + 0];
            queries[3 * nUniqueQueries + 1] = queries[3 * i + 1];
            queries[3 * nUniqueQueries + 2] = queries[3 * i + 2];
            nUniqueQueries++;
        }
    }

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep8Kernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntInputClusterStructure,
                                                                     const NumericTable * ntInputNClusters, const DataCollection * dcPartialQueries,
                                                                     NumericTable * ntClusterStructure, NumericTable * ntFinishedFlag,
                                                                     NumericTable * ntNClusters, DataCollection * dcQueries, const Parameter * par)
{
    const size_t blockIndex = par->blockIndex;
    const size_t nBlocks    = par->nBlocks;
    int result              = 0;

    const size_t nRows = ntInputClusterStructure->getNumberOfRows();

    size_t nClusters = ntInputNClusters->getValue<int>(0, 0);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, sizeof(Vector<int, cpu>));

    TArray<Vector<int, cpu>, cpu> queriesArray(nBlocks);
    DAAL_CHECK_MALLOC(queriesArray.get());
    Vector<int, cpu> * queries = queriesArray.get();

    ReadRows<int, cpu> inputClusterStructureRows(const_cast<NumericTable *>(ntInputClusterStructure), 0, nRows);
    if (nRows)
    {
        DAAL_CHECK_BLOCK_STATUS(inputClusterStructureRows);
    }
    const int * const inputClusterStructure = inputClusterStructureRows.get();

    WriteRows<int, cpu> clusterStructureRows(ntClusterStructure, 0, nRows);
    if (nRows)
    {
        DAAL_CHECK_BLOCK_STATUS(clusterStructureRows);
    }
    int * const clusterStructure = clusterStructureRows.get();

    result |= daal::services::internal::daal_memcpy_s(clusterStructure, sizeof(int) * 4 * nRows, inputClusterStructure, sizeof(int) * 4 * nRows);
    DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);

    for (size_t part = 0; part < dcPartialQueries->size(); part++)
    {
        NumericTablePtr ntCurPartialQueries = NumericTable::cast((*dcPartialQueries)[part]);

        const size_t nCurPartialQueries = ntCurPartialQueries->getNumberOfRows();

        const size_t defaultBlockSize = 256;
        const size_t nQueriesBlocks   = nCurPartialQueries / defaultBlockSize + int(nCurPartialQueries % defaultBlockSize > 0);

        for (size_t block = 0; block < nQueriesBlocks; block++)
        {
            const size_t i1    = block * defaultBlockSize;
            const size_t i2    = (block + 1 == nQueriesBlocks ? nCurPartialQueries : i1 + defaultBlockSize);
            const size_t iSize = i2 - i1;

            ReadRows<int, cpu> curPartialQueriesRows(ntCurPartialQueries.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(curPartialQueriesRows);
            const int * const curPartialQueries = curPartialQueriesRows.get();

            for (size_t i = 0; i < iSize; i++)
            {
                size_t id             = curPartialQueries[i * 3 + 0];
                size_t haloblockIndex = curPartialQueries[i * 3 + 1];
                size_t haloId         = curPartialQueries[i * 3 + 2];

                if (haloblockIndex == blockIndex)
                {
                    while (clusterStructure[haloId * 4 + 2] == blockIndex && clusterStructure[haloId * 4 + 3] != haloId)
                    {
                        haloId = clusterStructure[haloId * 4 + 3];
                    }
                }

                if (clusterStructure[id * 4 + 0] == noise)
                {
                    clusterStructure[id * 4 + 0] = undefined;
                    clusterStructure[id * 4 + 2] = haloblockIndex;
                    clusterStructure[id * 4 + 3] = haloId;
                    continue;
                }
                if (clusterStructure[id * 4 + 1] == 0)
                {
                    continue;
                }

                while (clusterStructure[id * 4 + 2] == blockIndex && clusterStructure[id * 4 + 3] != id)
                {
                    id = clusterStructure[id * 4 + 3];
                }

                if (haloblockIndex == blockIndex && haloId == id)
                {
                    continue;
                }

                size_t parentBlockIndex = clusterStructure[id * 4 + 2];
                size_t parentId         = clusterStructure[id * 4 + 3];
                if (parentBlockIndex == blockIndex && parentId == id)
                {
                    if (haloblockIndex == blockIndex)
                    {
                        size_t haloParentblockIndex = clusterStructure[haloId * 4 + 2];
                        size_t haloParentId         = clusterStructure[haloId * 4 + 3];
                        if (haloId < id)
                        {
                            clusterStructure[id * 4 + 2] = haloblockIndex;
                            clusterStructure[id * 4 + 3] = haloId;
                            nClusters--;
                        }
                        else // haloId > id
                        {
                            if (haloParentblockIndex == haloblockIndex)
                            {
                                clusterStructure[haloId * 4 + 2] = blockIndex;
                                clusterStructure[haloId * 4 + 3] = id;
                                nClusters--;
                            }
                            else // haloParentblockIndex != haloblockIndex
                            {
                                DAAL_CHECK_STATUS_VAR(addQuery(queries, haloParentblockIndex, haloParentId, blockIndex, id));
                            }
                        }
                    }
                    else
                    {
                        if (haloblockIndex < blockIndex)
                        {
                            clusterStructure[id * 4 + 2] = haloblockIndex;
                            clusterStructure[id * 4 + 3] = haloId;
                            nClusters--;
                        }
                        else // haloblockIndex > blockIndex
                        {
                            DAAL_CHECK_STATUS_VAR(addQuery(queries, haloblockIndex, haloId, blockIndex, id));
                        }
                    }
                }
                else
                {
                    if (haloblockIndex == blockIndex)
                    {
                        size_t haloParentblockIndex = clusterStructure[haloId * 4 + 2];
                        size_t haloParentId         = clusterStructure[haloId * 4 + 3];
                        if (haloId < id)
                        {
                            DAAL_CHECK_STATUS_VAR(addQuery(queries, parentBlockIndex, parentId, haloblockIndex, haloId));
                        }
                        else // haloId > id
                        {
                            if (haloParentblockIndex == blockIndex)
                            {
                                clusterStructure[haloId * 4 + 2] = blockIndex;
                                clusterStructure[haloId * 4 + 3] = id;
                                nClusters--;
                            }
                            else // haloParentblockIndex != blockIndex
                            {
                                DAAL_CHECK_STATUS_VAR(addQuery(queries, haloParentblockIndex, haloParentId, blockIndex, id));
                            }
                        }
                    }
                    else
                    {
                        if (haloblockIndex < blockIndex)
                        {
                            DAAL_CHECK_STATUS_VAR(addQuery(queries, parentBlockIndex, parentId, haloblockIndex, haloId));
                        }
                        else
                        {
                            DAAL_CHECK_STATUS_VAR(addQuery(queries, haloblockIndex, haloId, blockIndex, id));
                        }
                    }
                }
            }
        }
    }
    bool totalFinishedFlag = true;

    for (size_t part = 0; part < nBlocks; part++)
    {
        size_t nCurQueries;
        DAAL_CHECK_STATUS_VAR(removeDuplicateQueries(queries[part], nCurQueries));

        if (nCurQueries == 0)
        {
            continue;
        }
        totalFinishedFlag = false;

        NumericTablePtr ntCurQueries = NumericTable::cast((*dcQueries)[part]);
        DAAL_CHECK_STATUS_VAR(ntCurQueries->resize(nCurQueries));

        WriteRows<int, cpu> curQueriesRows(ntCurQueries.get(), 0, nCurQueries);
        DAAL_CHECK_BLOCK_STATUS(curQueriesRows);
        int * const curQueries = curQueriesRows.get();
        result |=
            daal::services::internal::daal_memcpy_s(curQueries, sizeof(int) * 3 * nCurQueries, queries[part].ptr(), sizeof(int) * 3 * nCurQueries);
    }

    {
        WriteRows<int, cpu> nClustersRows(ntNClusters, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(nClustersRows);
        int * const outNClusters = nClustersRows.get();
        outNClusters[0]          = nClusters;
    }

    {
        WriteRows<int, cpu> finishedFlagRows(ntFinishedFlag, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(finishedFlagRows);
        int * const finishedFlag = finishedFlagRows.get();
        finishedFlag[0]          = int(totalFinishedFlag);
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep9Kernel<algorithmFPType, method, cpu>::compute(const DataCollection * dcPartialNClusters, DataCollection * dcClusterOffsets)
{
    const size_t nBlocks = dcPartialNClusters->size();

    size_t totalNClusters = 0;

    for (size_t part = 0; part < nBlocks; part++)
    {
        NumericTablePtr ntCurNClusters = NumericTable::cast((*dcPartialNClusters)[part]);
        int curNClusters               = ntCurNClusters->getValue<int>(0, 0);

        NumericTablePtr ntClusterOffset = NumericTable::cast((*dcClusterOffsets)[part]);
        WriteRows<int, cpu> clusterOffsetRows(ntClusterOffset.get(), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(clusterOffsetRows);
        int * const clusterOffset = clusterOffsetRows.get();

        clusterOffset[0] = totalNClusters;
        totalNClusters += curNClusters;
    }

    NumericTablePtr ntClusterOffset = NumericTable::cast((*dcClusterOffsets)[nBlocks]);
    WriteRows<int, cpu> clusterOffsetRows(ntClusterOffset.get(), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(clusterOffsetRows);
    int * const clusterOffset = clusterOffsetRows.get();

    clusterOffset[0] = totalNClusters;

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep9Kernel<algorithmFPType, method, cpu>::finalizeCompute(const DataCollection * dcClusterOffsets, NumericTable * ntNClusters)
{
    const size_t nBlocks = dcClusterOffsets->size();

    NumericTablePtr ntCurNClusters = NumericTable::cast((*dcClusterOffsets)[nBlocks - 1]);
    const size_t totalNClusters    = ntCurNClusters->getValue<int>(0, 0);

    WriteRows<int, cpu> nClustersRows(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(nClustersRows);
    int * const nClusters = nClustersRows.get();
    nClusters[0]          = totalNClusters;

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep10Kernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntInputClusterStructure,
                                                                      const NumericTable * ntClusterOffset, NumericTable * ntClusterStructure,
                                                                      NumericTable * ntFinishedFlag, DataCollection * dcQueries,
                                                                      const Parameter * par)
{
    const size_t blockIndex = par->blockIndex;
    const size_t nBlocks    = par->nBlocks;

    const size_t offset = ntClusterOffset->getValue<int>(0, 0);
    const size_t nRows  = ntInputClusterStructure->getNumberOfRows();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, sizeof(Vector<int, cpu>));

    TArray<Vector<int, cpu>, cpu> queriesArray(nBlocks);
    DAAL_CHECK_MALLOC(queriesArray.get());
    Vector<int, cpu> * queries = queriesArray.get();

    const size_t defaultBlockSize = 256;
    const size_t nInputBlocks     = nRows / defaultBlockSize + int(nRows % defaultBlockSize > 0);

    size_t nClusters = 0;
    int result       = 0;

    for (size_t block = 0; block < nInputBlocks; block++)
    {
        const size_t i1    = block * defaultBlockSize;
        const size_t i2    = (block + 1 == nInputBlocks ? nRows : i1 + defaultBlockSize);
        const size_t iSize = i2 - i1;

        ReadRows<int, cpu> inputClusterStructureRows(const_cast<NumericTable *>(ntInputClusterStructure), i1, iSize);
        DAAL_CHECK_BLOCK_STATUS(inputClusterStructureRows);
        const int * const inputClusterStructure = inputClusterStructureRows.get();

        WriteRows<int, cpu> clusterStructureRows(ntClusterStructure, i1, iSize);
        DAAL_CHECK_BLOCK_STATUS(clusterStructureRows);
        int * const clusterStructure = clusterStructureRows.get();

        result |= daal::services::internal::daal_memcpy_s(clusterStructure, sizeof(int) * 4 * iSize, inputClusterStructure, sizeof(int) * 4 * iSize);

        for (size_t i = 0; i < iSize; i++)
        {
            size_t id = i + i1;
            if (clusterStructure[i * 4 + 0] != noise)
            {
                size_t parentBlockIndex = clusterStructure[i * 4 + 2];
                size_t parentId         = clusterStructure[i * 4 + 3];
                if (parentBlockIndex == blockIndex && parentId == id)
                {
                    clusterStructure[i * 4 + 0] = offset + nClusters;
                    nClusters++;
                }
                else if (parentBlockIndex != blockIndex)
                {
                    queries[parentBlockIndex].push_back(parentId);
                    queries[parentBlockIndex].push_back(id);
                }
            }
        }
    }

    bool totalFinishedFlag = true;

    for (size_t part = 0; part < nBlocks; part++)
    {
        const size_t nCurQueries = queries[part].size() / 2;

        if (nCurQueries == 0)
        {
            continue;
        }
        totalFinishedFlag = false;

        NumericTablePtr ntCurQueries = NumericTable::cast((*dcQueries)[part]);
        DAAL_CHECK_STATUS_VAR(ntCurQueries->resize(nCurQueries));

        WriteRows<int, cpu> curQueriesRows(ntCurQueries.get(), 0, nCurQueries);
        DAAL_CHECK_BLOCK_STATUS(curQueriesRows);
        int * const curQueries = curQueriesRows.get();
        for (size_t i = 0; i < nCurQueries; i++)
        {
            curQueries[i * 4 + 0] = 0;
            curQueries[i * 4 + 1] = queries[part][i * 2];
            curQueries[i * 4 + 2] = blockIndex;
            curQueries[i * 4 + 3] = queries[part][i * 2 + 1];
        }
    }

    {
        WriteRows<int, cpu> finishedFlagRows(ntFinishedFlag, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(finishedFlagRows);
        int * const finishedFlag = finishedFlagRows.get();
        finishedFlag[0]          = int(totalFinishedFlag);
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep11Kernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntInputClusterStructure,
                                                                      const DataCollection * dcPartialQueries, NumericTable * ntClusterStructure,
                                                                      NumericTable * ntFinishedFlag, DataCollection * dcQueries,
                                                                      const Parameter * par)
{
    const size_t blockIndex = par->blockIndex;
    const size_t nBlocks    = par->nBlocks;

    const size_t nRows = ntInputClusterStructure->getNumberOfRows();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, sizeof(Vector<int, cpu>));

    TArray<Vector<int, cpu>, cpu> queriesArray(nBlocks);
    DAAL_CHECK_MALLOC(queriesArray.get());
    Vector<int, cpu> * queries = queriesArray.get();

    int result = 0;

    ReadRows<int, cpu> inputClusterStructureRows(const_cast<NumericTable *>(ntInputClusterStructure), 0, nRows);
    if (nRows)
    {
        DAAL_CHECK_BLOCK_STATUS(inputClusterStructureRows);
    }
    const int * const inputClusterStructure = inputClusterStructureRows.get();

    WriteRows<int, cpu> clusterStructureRows(ntClusterStructure, 0, nRows);
    if (nRows)
    {
        DAAL_CHECK_BLOCK_STATUS(clusterStructureRows);
    }
    int * const clusterStructure = clusterStructureRows.get();

    result |= daal::services::internal::daal_memcpy_s(clusterStructure, sizeof(int) * 4 * nRows, inputClusterStructure, sizeof(int) * 4 * nRows);

    for (size_t part = 0; part < dcPartialQueries->size(); part++)
    {
        NumericTablePtr ntCurPartialQueries = NumericTable::cast((*dcPartialQueries)[part]);

        const size_t nCurPartialQueries = ntCurPartialQueries->getNumberOfRows();

        const size_t defaultBlockSize = 256;
        const size_t nQueriesBlocks   = nCurPartialQueries / defaultBlockSize + int(nCurPartialQueries % defaultBlockSize > 0);

        for (size_t block = 0; block < nQueriesBlocks; block++)
        {
            const size_t i1    = block * defaultBlockSize;
            const size_t i2    = (block + 1 == nQueriesBlocks ? nCurPartialQueries : i1 + defaultBlockSize);
            const size_t iSize = i2 - i1;

            ReadRows<int, cpu> curPartialQueriesRows(ntCurPartialQueries.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(curPartialQueriesRows);
            const int * const curPartialQueries = curPartialQueriesRows.get();

            for (size_t i = 0; i < iSize; i++)
            {
                size_t queryType = curPartialQueries[i * 4 + 0];

                if (queryType == 0)
                {
                    size_t id             = curPartialQueries[i * 4 + 1];
                    size_t haloblockIndex = curPartialQueries[i * 4 + 2];
                    size_t haloId         = curPartialQueries[i * 4 + 3];

                    while (clusterStructure[id * 4 + 2] == blockIndex && clusterStructure[id * 4 + 3] != id)
                    {
                        id = clusterStructure[id * 4 + 3];
                    }

                    size_t parentBlockIndex = clusterStructure[id * 4 + 2];
                    size_t parentId         = clusterStructure[id * 4 + 3];

                    if (parentBlockIndex == blockIndex)
                    {
                        if (haloblockIndex == blockIndex)
                        {
                            clusterStructure[haloId * 4 + 0] = clusterStructure[id * 4 + 0];
                        }
                        else
                        {
                            queries[haloblockIndex].push_back(1);
                            queries[haloblockIndex].push_back(haloId);
                            queries[haloblockIndex].push_back(clusterStructure[id * 4 + 0]);
                            queries[haloblockIndex].push_back(0);
                        }
                    }
                    else
                    {
                        queries[parentBlockIndex].push_back(0);
                        queries[parentBlockIndex].push_back(parentId);
                        queries[parentBlockIndex].push_back(haloblockIndex);
                        queries[parentBlockIndex].push_back(haloId);
                    }
                }
                else
                {
                    size_t id                    = curPartialQueries[i * 4 + 1];
                    size_t clusterId             = curPartialQueries[i * 4 + 2];
                    clusterStructure[id * 4 + 0] = clusterId;
                }
            }
        }
    }

    bool totalFinishedFlag = true;

    for (size_t part = 0; part < nBlocks; part++)
    {
        const size_t nCurQueries = queries[part].size() / 4;

        if (nCurQueries == 0)
        {
            continue;
        }
        totalFinishedFlag = false;

        NumericTablePtr ntCurQueries = NumericTable::cast((*dcQueries)[part]);
        DAAL_CHECK_STATUS_VAR(ntCurQueries->resize(nCurQueries));

        WriteRows<int, cpu> curQueriesRows(ntCurQueries.get(), 0, nCurQueries);
        DAAL_CHECK_BLOCK_STATUS(curQueriesRows);
        int * const curQueries = curQueriesRows.get();

        result |=
            daal::services::internal::daal_memcpy_s(curQueries, sizeof(int) * 4 * nCurQueries, queries[part].ptr(), sizeof(int) * 4 * nCurQueries);
    }

    {
        WriteRows<int, cpu> finishedFlagRows(ntFinishedFlag, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(finishedFlagRows);
        int * const finishedFlag = finishedFlagRows.get();
        finishedFlag[0]          = int(totalFinishedFlag);
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep12Kernel<algorithmFPType, method, cpu>::compute(NumericTable * ntInputClusterStructure, const DataCollection * dcPartialOrders,
                                                                      DataCollection * dcAssignmentQueries, const Parameter * par)
{
    const size_t blockIndex = par->blockIndex;
    const size_t nBlocks    = par->nBlocks;

    const size_t nRows = ntInputClusterStructure->getNumberOfRows();

    WriteRows<int, cpu> inputClusterStructureRows(ntInputClusterStructure, 0, nRows);
    if (nRows)
    {
        DAAL_CHECK_BLOCK_STATUS(inputClusterStructureRows);
    }
    int * const inputClusterStructure = inputClusterStructureRows.get();

    for (size_t id = 0; id < nRows; id++)
    {
        if (inputClusterStructure[id * 4 + 0] != noise)
        {
            size_t rootId = id;
            while (inputClusterStructure[rootId * 4 + 2] == blockIndex && inputClusterStructure[rootId * 4 + 3] != rootId)
            {
                rootId = inputClusterStructure[rootId * 4 + 3];
            }

            size_t clusterId = inputClusterStructure[rootId * 4 + 0];

            size_t curId = id;
            while (curId != rootId)
            {
                const size_t nextId                  = inputClusterStructure[curId * 4 + 3];
                inputClusterStructure[curId * 4 + 3] = rootId;
                curId                                = nextId;
            }

            inputClusterStructure[id * 4 + 0] = clusterId;
        }
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, sizeof(int));

    TArray<int, cpu> initialBlocksNRowsArray(nBlocks);
    DAAL_CHECK_MALLOC(initialBlocksNRowsArray.get());
    int * const initialBlocksNRows = initialBlocksNRowsArray.get();

    TArray<int, cpu> initialBlocksPosArray(nBlocks);
    DAAL_CHECK_MALLOC(initialBlocksPosArray.get());
    int * const initialBlocksPos = initialBlocksPosArray.get();

    for (size_t part = 0; part < nBlocks; part++)
    {
        initialBlocksNRows[part] = 0;
        initialBlocksPos[part]   = 0;
    }

    for (size_t part = 0; part < dcPartialOrders->size(); part++)
    {
        NumericTablePtr ntPartialOrders = NumericTable::cast((*dcPartialOrders)[part]);
        const size_t nPartialOrdersRows = ntPartialOrders->getNumberOfRows();

        const size_t defaultBlockSize = 256;
        const size_t nDataBlocks      = nPartialOrdersRows / defaultBlockSize + int(nPartialOrdersRows % defaultBlockSize > 0);

        for (size_t block = 0; block < nDataBlocks; block++)
        {
            const size_t i1    = block * defaultBlockSize;
            const size_t i2    = (block + 1 == nDataBlocks ? nPartialOrdersRows : i1 + defaultBlockSize);
            const size_t iSize = i2 - i1;

            ReadRows<int, cpu> partialOrdersRows(ntPartialOrders.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(partialOrdersRows);
            const int * const partialOrders = partialOrdersRows.get();

            for (size_t i = 0; i < iSize; i++)
            {
                size_t initialBlock = partialOrders[i * 2 + 0];
                initialBlocksNRows[initialBlock]++;
            }
        }
    }

    for (size_t part = 0; part < nBlocks; part++)
    {
        if (initialBlocksNRows[part] == 0)
        {
            continue;
        }

        NumericTablePtr ntAssignmentQueries = NumericTable::cast((*dcAssignmentQueries)[part]);
        DAAL_CHECK_STATUS_VAR(ntAssignmentQueries->resize(initialBlocksNRows[part]));
    }

    size_t pos = 0;

    for (size_t part = 0; part < dcPartialOrders->size(); part++)
    {
        NumericTablePtr ntPartialOrders = NumericTable::cast((*dcPartialOrders)[part]);
        const size_t nPartialOrdersRows = ntPartialOrders->getNumberOfRows();

        const size_t defaultBlockSize = 256;
        const size_t nDataBlocks      = nPartialOrdersRows / defaultBlockSize + int(nPartialOrdersRows % defaultBlockSize > 0);

        for (size_t block = 0; block < nDataBlocks; block++)
        {
            const size_t i1    = block * defaultBlockSize;
            const size_t i2    = (block + 1 == nDataBlocks ? nPartialOrdersRows : i1 + defaultBlockSize);
            const size_t iSize = i2 - i1;

            ReadRows<int, cpu> partialOrdersRows(ntPartialOrders.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS(partialOrdersRows);
            const int * const partialOrders = partialOrdersRows.get();

            for (size_t i = 0; i < iSize; i++)
            {
                const size_t initialBlock = partialOrders[i * 2 + 0];
                const size_t initialIndex = partialOrders[i * 2 + 1];
                const size_t clusterId    = inputClusterStructure[(pos + i) * 4 + 0];

                NumericTablePtr ntAssignmentQueries = NumericTable::cast((*dcAssignmentQueries)[initialBlock]);

                const size_t initialBlockPos = initialBlocksPos[initialBlock];
                initialBlocksPos[initialBlock]++;

                WriteRows<int, cpu> assignmentQueriesRows(ntAssignmentQueries.get(), initialBlockPos, 1);
                DAAL_CHECK_BLOCK_STATUS(assignmentQueriesRows);
                int * const assignmentQueries = assignmentQueriesRows.get();

                assignmentQueries[0] = initialIndex;
                assignmentQueries[1] = clusterId;
            }

            pos += iSize;
        }
    }

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep13Kernel<algorithmFPType, method, cpu>::compute(const DataCollection * dcPartialAssignmentQueries,
                                                                      NumericTable * ntAssignmentQueries, const Parameter * par)
{
    SafeStatus safeStat;

    const size_t nBlocks = dcPartialAssignmentQueries->size();
    size_t nRows         = 0;
    for (size_t part = 0; part < nBlocks; part++)
    {
        NumericTablePtr ntPartialAssignmentQueries = NumericTable::cast((*dcPartialAssignmentQueries)[part]);
        nRows += ntPartialAssignmentQueries->getNumberOfRows();
    }

    DAAL_CHECK_STATUS_VAR(ntAssignmentQueries->resize(nRows));

    size_t pos = 0;

    for (size_t part = 0; part < nBlocks; part++)
    {
        NumericTablePtr ntCurQueries = NumericTable::cast((*dcPartialAssignmentQueries)[part]);

        const size_t nCurQueries = ntCurQueries->getNumberOfRows();

        const size_t defaultBlockSize = 256;
        const size_t nQueriesBlocks   = nCurQueries / defaultBlockSize + int(nCurQueries % defaultBlockSize > 0);

        daal::threader_for(nQueriesBlocks, nQueriesBlocks, [&](size_t block) {
            const size_t i1    = block * defaultBlockSize;
            const size_t i2    = (block + 1 == nQueriesBlocks ? nCurQueries : i1 + defaultBlockSize);
            const size_t iSize = i2 - i1;
            int result         = 0;

            ReadRows<int, cpu> curQueriesRows(ntCurQueries.get(), i1, iSize);
            DAAL_CHECK_BLOCK_STATUS_THR(curQueriesRows);
            const int * const curQueries = curQueriesRows.get();

            WriteRows<int, cpu> assignmentQueriesRows(ntAssignmentQueries, pos + i1, iSize);
            DAAL_CHECK_BLOCK_STATUS_THR(assignmentQueriesRows);
            int * const assignmentQueries = assignmentQueriesRows.get();

            result |= daal::services::internal::daal_memcpy_s(assignmentQueries, sizeof(int) * iSize * 2, curQueries, sizeof(int) * iSize * 2);
            if (result) safeStat.add(services::Status(services::ErrorMemoryCopyFailedInternal));
        });

        pos += nCurQueries;
    }
    return safeStat.detach();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANDistrStep13Kernel<algorithmFPType, method, cpu>::finalizeCompute(const NumericTable * ntAssignmentQueries, NumericTable * ntAssignments,
                                                                              const Parameter * par)
{
    SafeStatus safeStat;

    const size_t nQueries = ntAssignmentQueries->getNumberOfRows();

    DAAL_CHECK_STATUS_VAR(ntAssignments->resize(nQueries));

    WriteRows<int, cpu> assignmentsRows(ntAssignments, 0, nQueries);
    DAAL_CHECK_BLOCK_STATUS(assignmentsRows);
    int * const assignments = assignmentsRows.get();

    const size_t defaultBlockSize = 256;
    const size_t nQueriesBlocks   = nQueries / defaultBlockSize + int(nQueries % defaultBlockSize > 0);

    daal::threader_for(nQueriesBlocks, nQueriesBlocks, [&](size_t block) {
        const size_t i1    = block * defaultBlockSize;
        const size_t i2    = (block + 1 == nQueriesBlocks ? nQueries : i1 + defaultBlockSize);
        const size_t iSize = i2 - i1;

        ReadRows<int, cpu> queriesRows(const_cast<NumericTable *>(ntAssignmentQueries), i1, iSize);
        DAAL_CHECK_BLOCK_STATUS_THR(queriesRows);
        const int * const queries = queriesRows.get();

        for (size_t i = 0; i < iSize; i++)
        {
            const int id        = queries[i * 2 + 0];
            const int clusterId = queries[i * 2 + 1];

            assignments[id] = clusterId;
        }
    });
    Status status = safeStat.detach();
    return status;
}

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal
