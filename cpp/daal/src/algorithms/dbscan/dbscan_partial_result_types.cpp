/* file: dbscan_partial_result_types.cpp */
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
//  Implementation of DBSCAN algorithm and types methods.
//--
*/

#include "algorithms/dbscan/dbscan_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep1, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP1_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep2, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep3, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep4, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP4_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep5, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP5_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep6, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP6_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep7, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP7_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep8, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP8_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedResultStep9, SERIALIZATION_DBSCAN_DISTRIBUTED_RESULT_STEP9_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep9, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP9_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep10, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP10_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep11, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP11_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep12, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP12_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedResultStep13, SERIALIZATION_DBSCAN_DISTRIBUTED_RESULT_STEP13_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep13, SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP13_ID);

DistributedPartialResultStep1::DistributedPartialResultStep1() : daal::algorithms::PartialResult(lastDistributedPartialResultStep1Id + 1) {}

NumericTablePtr DistributedPartialResultStep1::get(DistributedPartialResultStep1Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep1::set(DistributedPartialResultStep1Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep1::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const DistributedInput<step1Local> * algInput = static_cast<const DistributedInput<step1Local> *>(input);
    const size_t nRows                            = algInput->get(step1Data)->getNumberOfRows();

    NumericTablePtr ntPartialOrder = get(partialOrder);
    DAAL_CHECK_EX(ntPartialOrder, ErrorNullNumericTable, ArgumentName, partialOrderStr());

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS_VAR(checkNumericTable(ntPartialOrder.get(), partialOrderStr(), unexpectedLayouts, 0, 2, nRows));

    return Status();
}

DistributedPartialResultStep2::DistributedPartialResultStep2() : daal::algorithms::PartialResult(lastDistributedPartialResultStep2Id + 1) {}

NumericTablePtr DistributedPartialResultStep2::get(DistributedPartialResultStep2Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep2::set(DistributedPartialResultStep2Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep2::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const DistributedInput<step2Local> * algInput = static_cast<const DistributedInput<step2Local> *>(input);
    const size_t nFeatures                        = NumericTable::cast((*algInput->get(partialData))[0])->getNumberOfColumns();

    NumericTablePtr ntBoundingBox = get(boundingBox);
    DAAL_CHECK_EX(ntBoundingBox, ErrorNullNumericTable, ArgumentName, boundingBoxStr());

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS_VAR(checkNumericTable(ntBoundingBox.get(), boundingBoxStr(), unexpectedLayouts, 0, nFeatures, 2));

    return Status();
}

DistributedPartialResultStep3::DistributedPartialResultStep3() : daal::algorithms::PartialResult(lastDistributedPartialResultStep3Id + 1) {}

NumericTablePtr DistributedPartialResultStep3::get(DistributedPartialResultStep3Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep3::set(DistributedPartialResultStep3Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep3::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    NumericTablePtr ntSplit = get(split);
    DAAL_CHECK_EX(ntSplit, ErrorNullNumericTable, ArgumentName, splitStr());

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS_VAR(checkNumericTable(ntSplit.get(), splitStr(), unexpectedLayouts, 0, 2, 1));

    return Status();
}

DistributedPartialResultStep4::DistributedPartialResultStep4() : daal::algorithms::PartialResult(lastDistributedPartialResultStep4Id + 1) {}

DataCollectionPtr DistributedPartialResultStep4::get(DistributedPartialResultStep4Id id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep4::set(DistributedPartialResultStep4Id id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep4::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    const size_t nBlocks = par->leftBlocks + par->rightBlocks;

    const DistributedInput<step4Local> * algInput = static_cast<const DistributedInput<step4Local> *>(input);
    const size_t nFeatures                        = NumericTable::cast((*algInput->get(partialData))[0])->getNumberOfColumns();

    DataCollectionPtr dcPartitionedData          = get(partitionedData);
    DataCollectionPtr dcPartitionedPartialOrders = get(partitionedPartialOrders);

    DAAL_CHECK_EX(dcPartitionedData.get(), ErrorNullPartialResult, ArgumentName, partitionedDataStr());
    if (NumericTable::cast((*dcPartitionedData)[0])->getNumberOfRows() != 0)
    {
        DAAL_CHECK_EX(dcPartitionedData->size() == nBlocks, ErrorIncorrectDataCollectionSize, ArgumentName, partitionedDataStr());
    }

    DAAL_CHECK_EX(dcPartitionedPartialOrders.get(), ErrorNullPartialResult, ArgumentName, partitionedPartialOrdersStr());
    if (NumericTable::cast((*dcPartitionedPartialOrders)[0])->getNumberOfRows() != 0)
    {
        DAAL_CHECK_EX(dcPartitionedPartialOrders->size() == nBlocks, ErrorIncorrectDataCollectionSize, ArgumentName, partitionedPartialOrdersStr());
    }

    const int unexpectedLayouts = (int)packed_mask;

    if (NumericTable::cast((*dcPartitionedData)[0])->getNumberOfRows() != 0)
    {
        for (size_t i = 0; i < nBlocks; i++)
        {
            DAAL_CHECK_EX((*dcPartitionedData)[i], ErrorNullNumericTable, ArgumentName, partitionedDataStr());
            NumericTablePtr ntPartitionedData = NumericTable::cast((*dcPartitionedData)[i]);
            DAAL_CHECK_EX(ntPartitionedData, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partitionedDataStr());
            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntPartitionedData.get(), partitionedDataStr(), unexpectedLayouts, 0, nFeatures, 0, false));

            DAAL_CHECK_EX((*dcPartitionedPartialOrders)[i], ErrorNullNumericTable, ArgumentName, partitionedPartialOrdersStr());
            NumericTablePtr ntPartitionedPartialOrders = NumericTable::cast((*dcPartitionedPartialOrders)[i]);
            DAAL_CHECK_EX(ntPartitionedPartialOrders, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partitionedPartialOrdersStr());
            DAAL_CHECK_STATUS_VAR(
                checkNumericTable(ntPartitionedPartialOrders.get(), partitionedPartialOrdersStr(), unexpectedLayouts, 0, 2, 0, false));
        }
    }
    return Status();
}

DistributedPartialResultStep5::DistributedPartialResultStep5() : daal::algorithms::PartialResult(lastDistributedPartialResultStep5Id + 1) {}

DataCollectionPtr DistributedPartialResultStep5::get(DistributedPartialResultStep5Id id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep5::set(DistributedPartialResultStep5Id id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep5::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    const size_t nBlocks = par->nBlocks;

    const DistributedInput<step5Local> * algInput = static_cast<const DistributedInput<step5Local> *>(input);
    const size_t nFeatures                        = NumericTable::cast((*algInput->get(partialData))[0])->getNumberOfColumns();

    DataCollectionPtr dcPartitionedHaloData        = get(partitionedHaloData);
    DataCollectionPtr dcPartitionedHaloDataIndices = get(partitionedHaloDataIndices);

    DAAL_CHECK_EX(dcPartitionedHaloData.get(), ErrorNullPartialResult, ArgumentName, partitionedHaloDataStr());
    DAAL_CHECK_EX(dcPartitionedHaloData->size() == nBlocks, ErrorIncorrectDataCollectionSize, ArgumentName, partitionedHaloDataStr());

    DAAL_CHECK_EX(dcPartitionedHaloDataIndices.get(), ErrorNullPartialResult, ArgumentName, partitionedHaloDataIndicesStr());
    DAAL_CHECK_EX(dcPartitionedHaloDataIndices->size() == nBlocks, ErrorIncorrectDataCollectionSize, ArgumentName, partitionedHaloDataIndicesStr());

    const int unexpectedLayouts = (int)packed_mask;

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcPartitionedHaloData)[i], ErrorNullNumericTable, ArgumentName, partitionedHaloDataStr());
        NumericTablePtr ntPartitionedHaloData = NumericTable::cast((*dcPartitionedHaloData)[i]);
        DAAL_CHECK_EX(ntPartitionedHaloData, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partitionedHaloDataStr());
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntPartitionedHaloData.get(), partitionedHaloDataStr(), unexpectedLayouts, 0, nFeatures, 0, false));

        DAAL_CHECK_EX((*dcPartitionedHaloDataIndices)[i], ErrorNullNumericTable, ArgumentName, partitionedHaloDataIndicesStr());
        NumericTablePtr ntPartitionedHaloDataIndices = NumericTable::cast((*dcPartitionedHaloDataIndices)[i]);
        DAAL_CHECK_EX(ntPartitionedHaloDataIndices, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partitionedHaloDataIndicesStr());
        DAAL_CHECK_STATUS_VAR(
            checkNumericTable(ntPartitionedHaloDataIndices.get(), partitionedHaloDataIndicesStr(), unexpectedLayouts, 0, 1, 0, false));
    }

    return Status();
}

DistributedPartialResultStep6::DistributedPartialResultStep6() : daal::algorithms::PartialResult(lastDistributedPartialResultStep6CollectionId + 1) {}

NumericTablePtr DistributedPartialResultStep6::get(DistributedPartialResultStep6NumericTableId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

DataCollectionPtr DistributedPartialResultStep6::get(DistributedPartialResultStep6CollectionId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep6::set(DistributedPartialResultStep6NumericTableId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedPartialResultStep6::set(DistributedPartialResultStep6CollectionId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep6::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    const size_t nBlocks = par->nBlocks;

    const DistributedInput<step6Local> * algInput = static_cast<const DistributedInput<step6Local> *>(input);
    DataCollectionPtr dcPartialData               = algInput->get(partialData);
    size_t nRows                                  = 0;
    for (size_t i = 0; i < dcPartialData->size(); i++)
    {
        nRows += NumericTable::cast((*dcPartialData)[i])->getNumberOfRows();
    }

    if (nRows != 0)
    {
        NumericTablePtr ntClusterStructure = get(step6ClusterStructure);
        DAAL_CHECK_EX(ntClusterStructure, ErrorNullNumericTable, ArgumentName, step6ClusterStructureStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntClusterStructure.get(), step6ClusterStructureStr(), unexpectedLayouts, 0, 4, nRows));
    }

    {
        NumericTablePtr ntFinishedFlag = get(step6FinishedFlag);
        DAAL_CHECK_EX(ntFinishedFlag, ErrorNullNumericTable, ArgumentName, step6FinishedFlagStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntFinishedFlag.get(), step6FinishedFlagStr(), unexpectedLayouts, 0, 1, 1));
    }

    {
        NumericTablePtr ntNClusters = get(step6NClusters);
        DAAL_CHECK_EX(ntNClusters, ErrorNullNumericTable, ArgumentName, step6NClustersStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntNClusters.get(), step6NClustersStr(), unexpectedLayouts, 0, 1, 1));
    }

    DataCollectionPtr dcQueries = get(step6Queries);

    DAAL_CHECK_EX(dcQueries.get(), ErrorNullPartialResult, ArgumentName, step6QueriesStr());
    DAAL_CHECK_EX(dcQueries->size() == nBlocks, ErrorIncorrectDataCollectionSize, ArgumentName, step6QueriesStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcQueries)[i], ErrorNullNumericTable, ArgumentName, step6QueriesStr());
        NumericTablePtr ntQueries = NumericTable::cast((*dcQueries)[i]);
        DAAL_CHECK_EX(ntQueries, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step6QueriesStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntQueries.get(), step6QueriesStr(), unexpectedLayouts, 0, 3, 0, false));
    }

    return Status();
}

DistributedPartialResultStep7::DistributedPartialResultStep7() : daal::algorithms::PartialResult(lastDistributedPartialResultStep7Id + 1) {}

NumericTablePtr DistributedPartialResultStep7::get(DistributedPartialResultStep7Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep7::set(DistributedPartialResultStep7Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep7::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    NumericTablePtr ntFinishedFlag = get(finishedFlag);
    DAAL_CHECK_EX(ntFinishedFlag, ErrorNullNumericTable, ArgumentName, finishedFlagStr());

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS_VAR(checkNumericTable(ntFinishedFlag.get(), finishedFlagStr(), unexpectedLayouts, 0, 1, 1));

    return Status();
}

DistributedPartialResultStep8::DistributedPartialResultStep8() : daal::algorithms::PartialResult(lastDistributedPartialResultStep8CollectionId + 1) {}

NumericTablePtr DistributedPartialResultStep8::get(DistributedPartialResultStep8NumericTableId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

DataCollectionPtr DistributedPartialResultStep8::get(DistributedPartialResultStep8CollectionId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep8::set(DistributedPartialResultStep8NumericTableId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedPartialResultStep8::set(DistributedPartialResultStep8CollectionId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep8::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    const size_t nBlocks = par->nBlocks;

    const DistributedInput<step8Local> * algInput = static_cast<const DistributedInput<step8Local> *>(input);
    const size_t nRows                            = algInput->get(step8InputClusterStructure)->getNumberOfRows();

    if (nRows != 0)
    {
        NumericTablePtr ntClusterStructure = get(step8ClusterStructure);
        DAAL_CHECK_EX(ntClusterStructure, ErrorNullNumericTable, ArgumentName, step8ClusterStructureStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntClusterStructure.get(), step8ClusterStructureStr(), unexpectedLayouts, 0, 4, nRows));
    }

    {
        NumericTablePtr ntFinishedFlag = get(step8FinishedFlag);
        DAAL_CHECK_EX(ntFinishedFlag, ErrorNullNumericTable, ArgumentName, step8FinishedFlagStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntFinishedFlag.get(), step8FinishedFlagStr(), unexpectedLayouts, 0, 1, 1));
    }

    {
        NumericTablePtr ntNClusters = get(step8NClusters);
        DAAL_CHECK_EX(ntNClusters, ErrorNullNumericTable, ArgumentName, step8NClustersStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntNClusters.get(), step8NClustersStr(), unexpectedLayouts, 0, 1, 1));
    }

    DataCollectionPtr dcQueries = get(step8Queries);

    DAAL_CHECK_EX(dcQueries.get(), ErrorNullPartialResult, ArgumentName, step8QueriesStr());
    DAAL_CHECK_EX(dcQueries->size() == nBlocks, ErrorIncorrectDataCollectionSize, ArgumentName, step8QueriesStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcQueries)[i], ErrorNullNumericTable, ArgumentName, step8QueriesStr());
        NumericTablePtr ntQueries = NumericTable::cast((*dcQueries)[i]);
        DAAL_CHECK_EX(ntQueries, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step8QueriesStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntQueries.get(), step8QueriesStr(), unexpectedLayouts, 0, 3, 0, false));
    }

    return Status();
}

DistributedResultStep9::DistributedResultStep9() : daal::algorithms::Result(lastDistributedResultStep9Id + 1) {}

NumericTablePtr DistributedResultStep9::get(DistributedResultStep9Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedResultStep9::set(DistributedResultStep9Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedResultStep9::check(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * parameter, int method) const
{
    NumericTablePtr ntNClusters = get(step9NClusters);
    DAAL_CHECK_EX(ntNClusters, ErrorNullNumericTable, ArgumentName, step9NClustersStr());

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS_VAR(checkNumericTable(ntNClusters.get(), step9NClustersStr(), unexpectedLayouts, 0, 1, 1));

    return Status();
}

DistributedPartialResultStep9::DistributedPartialResultStep9() : daal::algorithms::PartialResult(lastDistributedPartialResultStep9Id + 1) {}

DataCollectionPtr DistributedPartialResultStep9::get(DistributedPartialResultStep9Id id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep9::set(DistributedPartialResultStep9Id id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep9::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const DistributedInput<step9Master> * algInput = static_cast<const DistributedInput<step9Master> *>(input);
    const size_t nBlocks                           = algInput->get(partialNClusters)->size();

    DataCollectionPtr dcClusterOffsets = get(clusterOffsets);

    DAAL_CHECK_EX(dcClusterOffsets.get(), ErrorNullPartialResult, ArgumentName, clusterOffsetsStr());
    DAAL_CHECK_EX(dcClusterOffsets->size() == nBlocks + 1, ErrorIncorrectDataCollectionSize, ArgumentName, clusterOffsetsStr());

    for (size_t i = 0; i < nBlocks + 1; i++)
    {
        DAAL_CHECK_EX((*dcClusterOffsets)[i], ErrorNullNumericTable, ArgumentName, clusterOffsetsStr());
        NumericTablePtr ntClusterOffset = NumericTable::cast((*dcClusterOffsets)[i]);
        DAAL_CHECK_EX(ntClusterOffset, ErrorIncorrectElementInNumericTableCollection, ArgumentName, clusterOffsetsStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntClusterOffset.get(), clusterOffsetsStr(), unexpectedLayouts, 0, 1, 1));
    }

    return Status();
}

DistributedPartialResultStep10::DistributedPartialResultStep10() : daal::algorithms::PartialResult(lastDistributedPartialResultStep10CollectionId + 1)
{}

NumericTablePtr DistributedPartialResultStep10::get(DistributedPartialResultStep10NumericTableId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

DataCollectionPtr DistributedPartialResultStep10::get(DistributedPartialResultStep10CollectionId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep10::set(DistributedPartialResultStep10NumericTableId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedPartialResultStep10::set(DistributedPartialResultStep10CollectionId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep10::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    const size_t nBlocks = par->nBlocks;

    const DistributedInput<step10Local> * algInput = static_cast<const DistributedInput<step10Local> *>(input);
    const size_t nRows                             = algInput->get(step10InputClusterStructure)->getNumberOfRows();

    if (nRows != 0)
    {
        NumericTablePtr ntClusterStructure = get(step10ClusterStructure);
        DAAL_CHECK_EX(ntClusterStructure, ErrorNullNumericTable, ArgumentName, step10ClusterStructureStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntClusterStructure.get(), step10ClusterStructureStr(), unexpectedLayouts, 0, 4, nRows));
    }

    {
        NumericTablePtr ntFinishedFlag = get(step10FinishedFlag);
        DAAL_CHECK_EX(ntFinishedFlag, ErrorNullNumericTable, ArgumentName, step10FinishedFlagStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntFinishedFlag.get(), step10FinishedFlagStr(), unexpectedLayouts, 0, 1, 1));
    }

    DataCollectionPtr dcQueries = get(step10Queries);

    DAAL_CHECK_EX(dcQueries.get(), ErrorNullPartialResult, ArgumentName, step10QueriesStr());
    DAAL_CHECK_EX(dcQueries->size() == nBlocks, ErrorIncorrectDataCollectionSize, ArgumentName, step10QueriesStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcQueries)[i], ErrorNullNumericTable, ArgumentName, step10QueriesStr());
        NumericTablePtr ntQueries = NumericTable::cast((*dcQueries)[i]);
        DAAL_CHECK_EX(ntQueries, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step10QueriesStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntQueries.get(), step10QueriesStr(), unexpectedLayouts, 0, 4, 0, false));
    }

    return Status();
}

DistributedPartialResultStep11::DistributedPartialResultStep11() : daal::algorithms::PartialResult(lastDistributedPartialResultStep11CollectionId + 1)
{}

NumericTablePtr DistributedPartialResultStep11::get(DistributedPartialResultStep11NumericTableId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

DataCollectionPtr DistributedPartialResultStep11::get(DistributedPartialResultStep11CollectionId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep11::set(DistributedPartialResultStep11NumericTableId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedPartialResultStep11::set(DistributedPartialResultStep11CollectionId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep11::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    const size_t nBlocks = par->nBlocks;

    const DistributedInput<step11Local> * algInput = static_cast<const DistributedInput<step11Local> *>(input);
    const size_t nRows                             = algInput->get(step11InputClusterStructure)->getNumberOfRows();

    if (nRows != 0)
    {
        NumericTablePtr ntClusterStructure = get(step11ClusterStructure);
        DAAL_CHECK_EX(ntClusterStructure, ErrorNullNumericTable, ArgumentName, step11ClusterStructureStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntClusterStructure.get(), step11ClusterStructureStr(), unexpectedLayouts, 0, 4, nRows));
    }

    {
        NumericTablePtr ntFinishedFlag = get(step11FinishedFlag);
        DAAL_CHECK_EX(ntFinishedFlag, ErrorNullNumericTable, ArgumentName, step11FinishedFlagStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntFinishedFlag.get(), step11FinishedFlagStr(), unexpectedLayouts, 0, 1, 1));
    }

    DataCollectionPtr dcQueries = get(step11Queries);

    DAAL_CHECK_EX(dcQueries.get(), ErrorNullPartialResult, ArgumentName, step11QueriesStr());
    DAAL_CHECK_EX(dcQueries->size() == nBlocks, ErrorIncorrectDataCollectionSize, ArgumentName, step11QueriesStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcQueries)[i], ErrorNullNumericTable, ArgumentName, step11QueriesStr());
        NumericTablePtr ntQueries = NumericTable::cast((*dcQueries)[i]);
        DAAL_CHECK_EX(ntQueries, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step11QueriesStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntQueries.get(), step11QueriesStr(), unexpectedLayouts, 0, 4, 0, false));
    }

    return Status();
}

DistributedPartialResultStep12::DistributedPartialResultStep12() : daal::algorithms::PartialResult(lastDistributedPartialResultStep12Id + 1) {}

DataCollectionPtr DistributedPartialResultStep12::get(DistributedPartialResultStep12Id id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep12::set(DistributedPartialResultStep12Id id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep12::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    const size_t nBlocks = par->nBlocks;

    DataCollectionPtr dcAssignmentQueries = get(assignmentQueries);

    DAAL_CHECK_EX(dcAssignmentQueries.get(), ErrorNullPartialResult, ArgumentName, assignmentQueriesStr());
    DAAL_CHECK_EX(dcAssignmentQueries->size() == nBlocks, ErrorIncorrectDataCollectionSize, ArgumentName, assignmentQueriesStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcAssignmentQueries)[i], ErrorNullNumericTable, ArgumentName, assignmentQueriesStr());
        NumericTablePtr ntAssignmentQueries = NumericTable::cast((*dcAssignmentQueries)[i]);
        DAAL_CHECK_EX(ntAssignmentQueries, ErrorIncorrectElementInNumericTableCollection, ArgumentName, assignmentQueriesStr());

        const int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntAssignmentQueries.get(), assignmentQueriesStr(), unexpectedLayouts, 0, 2, 0, false));
    }

    return Status();
}

DistributedResultStep13::DistributedResultStep13() : daal::algorithms::Result(lastDistributedResultStep13Id + 1) {}

NumericTablePtr DistributedResultStep13::get(DistributedResultStep13Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedResultStep13::set(DistributedResultStep13Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedResultStep13::check(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * parameter, int method) const
{
    NumericTablePtr ntAssignments = get(step13Assignments);
    DAAL_CHECK_EX(ntAssignments, ErrorNullNumericTable, ArgumentName, step13AssignmentsStr());

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS_VAR(checkNumericTable(ntAssignments.get(), step13AssignmentsStr(), unexpectedLayouts, 0, 1, 0, false));

    return Status();
}

DistributedPartialResultStep13::DistributedPartialResultStep13() : daal::algorithms::PartialResult(lastDistributedPartialResultStep13Id + 1) {}

NumericTablePtr DistributedPartialResultStep13::get(DistributedPartialResultStep13Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep13::set(DistributedPartialResultStep13Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep13::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    NumericTablePtr ntAssignmentQueries = get(step13AssignmentQueries);
    DAAL_CHECK_EX(ntAssignmentQueries, ErrorNullNumericTable, ArgumentName, step13AssignmentQueriesStr());

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS_VAR(checkNumericTable(ntAssignmentQueries.get(), step13AssignmentQueriesStr(), unexpectedLayouts, 0, 2, 0, false));

    return Status();
}

} // namespace interface1
} // namespace dbscan
} // namespace algorithms
} // namespace daal
