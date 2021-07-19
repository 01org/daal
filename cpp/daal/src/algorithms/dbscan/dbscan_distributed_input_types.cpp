/* file: dbscan_distributed_input_types.cpp */
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
#include "src/data_management/service_numeric_table.h"
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
DistributedInput<step1Local>::DistributedInput() : daal::algorithms::Input(lastStep1LocalNumericTableInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step1Local>::get(Step1LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step1Local>::set(Step1LocalNumericTableInputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step1Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);
    DAAL_CHECK_EX(par->nBlocks > 0, ErrorIncorrectParameter, ParameterName, nBlocksStr());
    DAAL_CHECK_EX(par->blockIndex < par->nBlocks, ErrorIncorrectParameter, ParameterName, blockIndexStr());

    DAAL_CHECK_STATUS_VAR(checkNumericTable(get(step1Data).get(), step1DataStr(), 0, 0));

    return services::Status();
}

DistributedInput<step2Local>::DistributedInput() : daal::algorithms::Input(lastLocalCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step2Local>::get(LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step2Local>::set(LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step2Local>::add(LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step2Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DataCollectionPtr dcPartialData = get(partialData);
    DAAL_CHECK_EX(dcPartialData, ErrorNullInputDataCollection, ArgumentName, partialDataStr());

    const size_t nBlocks = dcPartialData->size();
    DAAL_CHECK_EX(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialDataStr());

    if (NumericTable::cast((*dcPartialData)[0])->getNumberOfRows() == 0)
    {
        size_t nFeatures = 0;
        for (size_t i = 0; i < nBlocks; i++)
        {
            DAAL_CHECK_EX((*dcPartialData)[i], ErrorNullNumericTable, ArgumentName, partialDataStr());
            NumericTablePtr ntPartialData = NumericTable::cast((*dcPartialData)[i]);
            DAAL_CHECK_EX(ntPartialData, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partialDataStr());

            if (i == 0)
            {
                nFeatures = ntPartialData->getNumberOfColumns();
            }

            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntPartialData.get(), partialDataStr(), 0, 0, nFeatures, 0));
        }
    }
    return services::Status();
}

DistributedInput<step3Local>::DistributedInput() : daal::algorithms::Input(lastStep3LocalCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step3Local>::get(LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step3Local>::get(Step3LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step3Local>::set(LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step3Local>::set(Step3LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step3Local>::add(LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step3Local>::add(Step3LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step3Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    DAAL_CHECK_EX(par->leftBlocks > 0, ErrorIncorrectParameter, ParameterName, leftBlocksStr());
    DAAL_CHECK_EX(par->rightBlocks > 0, ErrorIncorrectParameter, ParameterName, rightBlocksStr());

    const size_t nBlocks = par->leftBlocks + par->rightBlocks;

    DataCollectionPtr dcPartialData = get(partialData);
    DAAL_CHECK_EX(dcPartialData, ErrorNullInputDataCollection, ArgumentName, partialDataStr());

    const size_t nDataBlocks = dcPartialData->size();
    DAAL_CHECK_EX(nDataBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialDataStr());

    size_t nFeatures = NumericTable::cast((*dcPartialData)[0])->getNumberOfColumns();
    if (NumericTable::cast((*dcPartialData)[0])->getNumberOfRows() != 0)
    {
        for (size_t i = 0; i < nDataBlocks; i++)
        {
            DAAL_CHECK_EX((*dcPartialData)[i], ErrorNullNumericTable, ArgumentName, partialDataStr());
            NumericTablePtr ntPartialData = NumericTable::cast((*dcPartialData)[i]);
            DAAL_CHECK_EX(ntPartialData, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partialDataStr());

            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntPartialData.get(), partialDataStr(), 0, 0, nFeatures, 0));
        }
    }

    DataCollectionPtr dcBoundindBoxes = get(step3PartialBoundingBoxes);
    DAAL_CHECK_EX(dcBoundindBoxes, ErrorNullInputDataCollection, ArgumentName, step3PartialBoundingBoxesStr());
    DAAL_CHECK_EX(dcBoundindBoxes->size() == nBlocks, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, step3PartialBoundingBoxesStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcBoundindBoxes)[i], ErrorNullNumericTable, ArgumentName, step3PartialBoundingBoxesStr());
        NumericTablePtr ntBoundingBox = NumericTable::cast((*dcBoundindBoxes)[i]);
        DAAL_CHECK_EX(ntBoundingBox, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step3PartialBoundingBoxesStr());

        int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntBoundingBox.get(), step3PartialBoundingBoxesStr(), unexpectedLayouts, 0, nFeatures, 2));
    }

    return services::Status();
}

DistributedInput<step4Local>::DistributedInput() : daal::algorithms::Input(lastStep4LocalCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step4Local>::get(LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step4Local>::get(Step4LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step4Local>::set(LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step4Local>::set(Step4LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step4Local>::add(LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step4Local>::add(Step4LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step4Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    DAAL_CHECK_EX(par->leftBlocks > 0, ErrorIncorrectParameter, ParameterName, leftBlocksStr());
    DAAL_CHECK_EX(par->rightBlocks > 0, ErrorIncorrectParameter, ParameterName, rightBlocksStr());

    const size_t nBlocks = par->leftBlocks + par->rightBlocks;

    DataCollectionPtr dcPartialData   = get(partialData);
    DataCollectionPtr dcPartialOrders = get(step4PartialOrders);
    DAAL_CHECK_EX(dcPartialData, ErrorNullInputDataCollection, ArgumentName, partialDataStr());
    DAAL_CHECK_EX(dcPartialOrders, ErrorNullInputDataCollection, ArgumentName, step4PartialOrdersStr());

    const size_t nDataBlocks = dcPartialData->size();
    DAAL_CHECK_EX(nDataBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialDataStr());
    DAAL_CHECK_EX(dcPartialOrders->size() == nDataBlocks, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, step4PartialOrdersStr());

    const int unexpectedLayouts = (int)packed_mask;

    size_t nFeatures = NumericTable::cast((*dcPartialData)[0])->getNumberOfColumns();
    if (NumericTable::cast((*dcPartialData)[0])->getNumberOfRows() != 0)
    {
        for (size_t i = 0; i < nDataBlocks; i++)
        {
            DAAL_CHECK_EX((*dcPartialData)[i], ErrorNullNumericTable, ArgumentName, partialDataStr());
            NumericTablePtr ntPartialData = NumericTable::cast((*dcPartialData)[i]);
            DAAL_CHECK_EX(ntPartialData, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partialDataStr());

            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntPartialData.get(), partialDataStr(), 0, 0, nFeatures, 0));

            const size_t nRows = ntPartialData->getNumberOfRows();

            DAAL_CHECK_EX((*dcPartialOrders)[i], ErrorNullNumericTable, ArgumentName, step4PartialOrdersStr());
            NumericTablePtr ntPartialOrder = NumericTable::cast((*dcPartialOrders)[i]);
            DAAL_CHECK_EX(ntPartialOrder, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step4PartialOrdersStr());
            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntPartialOrder.get(), step4PartialOrdersStr(), unexpectedLayouts, 0, 2, nRows));
        }
    }

    DataCollectionPtr dcSplits = get(step4PartialSplits);
    DAAL_CHECK_EX(dcSplits, ErrorNullInputDataCollection, ArgumentName, step4PartialSplitsStr());
    DAAL_CHECK_EX(dcSplits->size() == nBlocks, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, step4PartialSplitsStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcSplits)[i], ErrorNullNumericTable, ArgumentName, step4PartialSplitsStr());
        NumericTablePtr ntPartialSplit = NumericTable::cast((*dcSplits)[i]);
        DAAL_CHECK_EX(ntPartialSplit, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step4PartialSplitsStr());

        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntPartialSplit.get(), step4PartialSplitsStr(), unexpectedLayouts, 0, 2, 1));
    }

    return services::Status();
}

DistributedInput<step5Local>::DistributedInput() : daal::algorithms::Input(lastStep5LocalCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step5Local>::get(LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step5Local>::get(Step5LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step5Local>::set(LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step5Local>::set(Step5LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step5Local>::add(LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step5Local>::add(Step5LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step5Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    DAAL_CHECK_EX(par->nBlocks > 0, ErrorIncorrectParameter, ParameterName, nBlocksStr());
    DAAL_CHECK_EX(par->blockIndex < par->nBlocks, ErrorIncorrectParameter, ParameterName, blockIndexStr());

    const size_t nBlocks = par->nBlocks;

    DataCollectionPtr dcPartialData = get(partialData);
    DAAL_CHECK_EX(dcPartialData, ErrorNullInputDataCollection, ArgumentName, partialDataStr());

    const size_t nDataBlocks = dcPartialData->size();
    DAAL_CHECK_EX(nDataBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialDataStr());

    size_t nFeatures = NumericTable::cast((*dcPartialData)[0])->getNumberOfColumns();
    if (NumericTable::cast((*dcPartialData)[0])->getNumberOfRows() != 0)
    {
        for (size_t i = 0; i < nDataBlocks; i++)
        {
            DAAL_CHECK_EX((*dcPartialData)[i], ErrorNullNumericTable, ArgumentName, partialDataStr());
            NumericTablePtr ntPartialData = NumericTable::cast((*dcPartialData)[i]);
            DAAL_CHECK_EX(ntPartialData, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partialDataStr());

            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntPartialData.get(), partialDataStr(), 0, 0, nFeatures, 0));
        }
    }

    DataCollectionPtr dcBoundindBoxes = get(step5PartialBoundingBoxes);
    DAAL_CHECK_EX(dcBoundindBoxes, ErrorNullInputDataCollection, ArgumentName, step5PartialBoundingBoxesStr());
    DAAL_CHECK_EX(dcBoundindBoxes->size() == nBlocks, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, step5PartialBoundingBoxesStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcBoundindBoxes)[i], ErrorNullNumericTable, ArgumentName, step5PartialBoundingBoxesStr());
        NumericTablePtr ntBoundingBox = NumericTable::cast((*dcBoundindBoxes)[i]);
        DAAL_CHECK_EX(ntBoundingBox, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step5PartialBoundingBoxesStr());

        int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntBoundingBox.get(), step5PartialBoundingBoxesStr(), unexpectedLayouts, 0, nFeatures, 2));
    }

    return services::Status();
}

DistributedInput<step6Local>::DistributedInput() : daal::algorithms::Input(lastStep6LocalCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step6Local>::get(LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step6Local>::get(Step6LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step6Local>::set(LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step6Local>::set(Step6LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step6Local>::add(LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step6Local>::add(Step6LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step6Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    DAAL_CHECK_STATUS_VAR(par->check());
    DAAL_CHECK_EX(par->nBlocks > 0, ErrorIncorrectParameter, ParameterName, nBlocksStr());
    DAAL_CHECK_EX(par->blockIndex < par->nBlocks, ErrorIncorrectParameter, ParameterName, blockIndexStr());

    DataCollectionPtr dcPartialData = get(partialData);
    DAAL_CHECK_EX(dcPartialData, ErrorNullInputDataCollection, ArgumentName, partialDataStr());

    const size_t nDataBlocks = dcPartialData->size();
    DAAL_CHECK_EX(nDataBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialDataStr());

    size_t nFeatures = NumericTable::cast((*dcPartialData)[0])->getNumberOfColumns();
    if (NumericTable::cast((*dcPartialData)[0])->getNumberOfRows() != 0)
    {
        for (size_t i = 0; i < nDataBlocks; i++)
        {
            DAAL_CHECK_EX((*dcPartialData)[i], ErrorNullNumericTable, ArgumentName, partialDataStr());
            NumericTablePtr ntPartialData = NumericTable::cast((*dcPartialData)[i]);
            DAAL_CHECK_EX(ntPartialData, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partialDataStr());

            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntPartialData.get(), partialDataStr(), 0, 0, nFeatures, 0));
        }
    }

    DataCollectionPtr dcHaloData        = get(haloData);
    DataCollectionPtr dcHaloDataIndices = get(haloDataIndices);
    DataCollectionPtr dcHaloBlocks      = get(haloBlocks);
    DAAL_CHECK_EX(dcHaloData, ErrorNullInputDataCollection, ArgumentName, haloDataStr());
    DAAL_CHECK_EX(dcHaloDataIndices, ErrorNullInputDataCollection, ArgumentName, haloDataIndicesStr());
    DAAL_CHECK_EX(dcHaloBlocks, ErrorNullInputDataCollection, ArgumentName, haloBlocksStr());

    const size_t nHaloBlocks = dcHaloData->size();
    DAAL_CHECK_EX(dcHaloDataIndices->size() == nHaloBlocks, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, haloDataIndicesStr());
    DAAL_CHECK_EX(dcHaloBlocks->size() == nHaloBlocks, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, haloBlocksStr());

    int unexpectedLayouts = (int)packed_mask;

    for (size_t i = 0; i < nHaloBlocks; i++)
    {
        DAAL_CHECK_EX((*dcHaloData)[i], ErrorNullNumericTable, ArgumentName, haloDataStr());
        NumericTablePtr ntHaloData = NumericTable::cast((*dcHaloData)[i]);
        DAAL_CHECK_EX(ntHaloData, ErrorIncorrectElementInNumericTableCollection, ArgumentName, haloDataStr());
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntHaloData.get(), haloDataStr(), 0, 0, nFeatures, 0));

        DAAL_CHECK_EX((*dcHaloDataIndices)[i], ErrorNullNumericTable, ArgumentName, haloDataIndicesStr());
        NumericTablePtr ntHaloDataIndices = NumericTable::cast((*dcHaloDataIndices)[i]);
        DAAL_CHECK_EX(ntHaloDataIndices, ErrorIncorrectElementInNumericTableCollection, ArgumentName, haloDataIndicesStr());
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntHaloDataIndices.get(), haloDataIndicesStr(), unexpectedLayouts, 0, 1, 0));

        DAAL_CHECK_EX((*dcHaloBlocks)[i], ErrorNullNumericTable, ArgumentName, haloBlocksStr());
        NumericTablePtr ntHaloBlock = NumericTable::cast((*dcHaloBlocks)[i]);
        DAAL_CHECK_EX(ntHaloBlock, ErrorIncorrectElementInNumericTableCollection, ArgumentName, haloBlocksStr());
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntHaloBlock.get(), haloBlocksStr(), unexpectedLayouts, 0, 1, 1));
    }

    return services::Status();
}

DistributedInput<step7Master>::DistributedInput() : daal::algorithms::Input(lastStep7MasterCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step7Master>::get(Step7MasterCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step7Master>::set(Step7MasterCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step7Master>::add(Step7MasterCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step7Master>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DataCollectionPtr dcFinishedFlags = get(partialFinishedFlags);
    DAAL_CHECK_EX(dcFinishedFlags, ErrorNullInputDataCollection, ArgumentName, partialFinishedFlagsStr());

    const size_t nBlocks = dcFinishedFlags->size();
    DAAL_CHECK_EX(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialFinishedFlagsStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcFinishedFlags)[i], ErrorNullNumericTable, ArgumentName, partialFinishedFlagsStr());
        NumericTablePtr ntFinishedFlag = NumericTable::cast((*dcFinishedFlags)[i]);
        DAAL_CHECK_EX(ntFinishedFlag, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partialFinishedFlagsStr());

        int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntFinishedFlag.get(), partialFinishedFlagsStr(), unexpectedLayouts, 0, 1, 1));
    }

    return services::Status();
}

DistributedInput<step8Local>::DistributedInput() : daal::algorithms::Input(lastStep8LocalCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step8Local>::get(Step8LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step8Local>::get(Step8LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step8Local>::set(Step8LocalNumericTableInputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step8Local>::set(Step8LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step8Local>::add(Step8LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step8Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    DAAL_CHECK_STATUS_VAR(par->check());
    DAAL_CHECK_EX(par->nBlocks > 0, ErrorIncorrectParameter, ParameterName, nBlocksStr());
    DAAL_CHECK_EX(par->blockIndex < par->nBlocks, ErrorIncorrectParameter, ParameterName, blockIndexStr());

    {
        NumericTablePtr ntClusterStructure = get(step8InputClusterStructure);

        if (ntClusterStructure->getNumberOfRows() != 0)
        {
            int unexpectedLayouts = (int)packed_mask;
            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntClusterStructure.get(), step8InputClusterStructureStr(), unexpectedLayouts, 0, 4, 0));
        }
    }

    {
        NumericTablePtr ntNClusters = get(step8InputNClusters);
        DAAL_CHECK_EX(ntNClusters, ErrorNullNumericTable, ArgumentName, step8InputNClustersStr());

        int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntNClusters.get(), step8InputNClustersStr(), unexpectedLayouts, 0, 1, 1));
    }

    DataCollectionPtr dcQueries = get(step8PartialQueries);
    DAAL_CHECK_EX(dcQueries, ErrorNullInputDataCollection, ArgumentName, step8PartialQueriesStr());

    const size_t nQueriesBlocks = dcQueries->size();

    for (size_t i = 0; i < nQueriesBlocks; i++)
    {
        DAAL_CHECK_EX((*dcQueries)[i], ErrorNullNumericTable, ArgumentName, step8PartialQueriesStr());
        NumericTablePtr ntQueries = NumericTable::cast((*dcQueries)[i]);
        DAAL_CHECK_EX(ntQueries, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step8PartialQueriesStr());

        int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntQueries.get(), step8PartialQueriesStr(), unexpectedLayouts, 0, 3, 0));
    }

    return services::Status();
}

DistributedInput<step9Master>::DistributedInput() : daal::algorithms::Input(lastStep9MasterCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step9Master>::get(Step9MasterCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step9Master>::set(Step9MasterCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step9Master>::add(Step9MasterCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step9Master>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DataCollectionPtr dcNClusters = get(partialNClusters);
    DAAL_CHECK_EX(dcNClusters, ErrorNullInputDataCollection, ArgumentName, partialNClustersStr());

    const size_t nBlocks = dcNClusters->size();
    DAAL_CHECK_EX(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialNClustersStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*dcNClusters)[i], ErrorNullNumericTable, ArgumentName, partialNClustersStr());
        NumericTablePtr ntNClusters = NumericTable::cast((*dcNClusters)[i]);
        DAAL_CHECK_EX(ntNClusters, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partialNClustersStr());

        int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntNClusters.get(), partialNClustersStr(), unexpectedLayouts, 0, 1, 1));
    }

    return services::Status();
}

DistributedInput<step10Local>::DistributedInput() : daal::algorithms::Input(lastStep10LocalNumericTableInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step10Local>::get(Step10LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step10Local>::set(Step10LocalNumericTableInputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step10Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    DAAL_CHECK_STATUS_VAR(par->check());
    DAAL_CHECK_EX(par->nBlocks > 0, ErrorIncorrectParameter, ParameterName, nBlocksStr());
    DAAL_CHECK_EX(par->blockIndex < par->nBlocks, ErrorIncorrectParameter, ParameterName, blockIndexStr());

    {
        NumericTablePtr ntClusterStructure = get(step10InputClusterStructure);
        if (ntClusterStructure->getNumberOfRows() != 0)
        {
            int unexpectedLayouts = (int)packed_mask;
            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntClusterStructure.get(), step10InputClusterStructureStr(), unexpectedLayouts, 0, 4, 0));
        }
    }

    {
        NumericTablePtr ntClusterOffset = get(step10ClusterOffset);
        DAAL_CHECK_EX(ntClusterOffset, ErrorNullNumericTable, ArgumentName, step10ClusterOffsetStr());

        int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntClusterOffset.get(), step10ClusterOffsetStr(), unexpectedLayouts, 0, 1, 1));
    }

    return services::Status();
}

DistributedInput<step11Local>::DistributedInput() : daal::algorithms::Input(lastStep11LocalCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step11Local>::get(Step11LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step11Local>::get(Step11LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step11Local>::set(Step11LocalNumericTableInputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step11Local>::set(Step11LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step11Local>::add(Step11LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step11Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    DAAL_CHECK_STATUS_VAR(par->check());
    DAAL_CHECK_EX(par->nBlocks > 0, ErrorIncorrectParameter, ParameterName, nBlocksStr());
    DAAL_CHECK_EX(par->blockIndex < par->nBlocks, ErrorIncorrectParameter, ParameterName, blockIndexStr());

    {
        NumericTablePtr ntClusterStructure = get(step11InputClusterStructure);
        if (ntClusterStructure->getNumberOfRows() != 0)
        {
            int unexpectedLayouts = (int)packed_mask;
            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntClusterStructure.get(), step11InputClusterStructureStr(), unexpectedLayouts, 0, 4, 0));
        }
    }

    DataCollectionPtr dcQueries = get(step11PartialQueries);
    DAAL_CHECK_EX(dcQueries, ErrorNullInputDataCollection, ArgumentName, step11PartialQueriesStr());

    const size_t nQueriesBlocks = dcQueries->size();

    for (size_t i = 0; i < nQueriesBlocks; i++)
    {
        DAAL_CHECK_EX((*dcQueries)[i], ErrorNullNumericTable, ArgumentName, step11PartialQueriesStr());
        NumericTablePtr ntQueries = NumericTable::cast((*dcQueries)[i]);
        DAAL_CHECK_EX(ntQueries, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step11PartialQueriesStr());

        int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntQueries.get(), step11PartialQueriesStr(), unexpectedLayouts, 0, 4, 0));
    }

    return services::Status();
}

DistributedInput<step12Local>::DistributedInput() : daal::algorithms::Input(lastStep12LocalCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step12Local>::get(Step12LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step12Local>::get(Step12LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step12Local>::set(Step12LocalNumericTableInputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step12Local>::set(Step12LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step12Local>::add(Step12LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step12Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * par = static_cast<const Parameter *>(parameter);

    DAAL_CHECK_STATUS_VAR(par->check());
    DAAL_CHECK_EX(par->nBlocks > 0, ErrorIncorrectParameter, ParameterName, nBlocksStr());
    DAAL_CHECK_EX(par->blockIndex < par->nBlocks, ErrorIncorrectParameter, ParameterName, blockIndexStr());

    {
        NumericTablePtr ntClusterStructure = get(step12InputClusterStructure);
        if (ntClusterStructure->getNumberOfRows() != 0)
        {
            int unexpectedLayouts = (int)packed_mask;
            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntClusterStructure.get(), step12InputClusterStructureStr(), unexpectedLayouts, 0, 4, 0));
        }
    }

    DataCollectionPtr dcOrders = get(step12PartialOrders);
    DAAL_CHECK_EX(dcOrders, ErrorNullInputDataCollection, ArgumentName, step12PartialOrdersStr());

    const size_t nQueriesBlocks = dcOrders->size();
    DAAL_CHECK_EX(nQueriesBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, step12PartialOrdersStr());

    if (NumericTable::cast((*dcOrders)[0])->getNumberOfRows() != 0)
    {
        for (size_t i = 0; i < nQueriesBlocks; i++)
        {
            DAAL_CHECK_EX((*dcOrders)[i], ErrorNullNumericTable, ArgumentName, step12PartialOrdersStr());
            NumericTablePtr ntOrders = NumericTable::cast((*dcOrders)[i]);
            DAAL_CHECK_EX(ntOrders, ErrorIncorrectElementInNumericTableCollection, ArgumentName, step12PartialOrdersStr());

            int unexpectedLayouts = (int)packed_mask;
            DAAL_CHECK_STATUS_VAR(checkNumericTable(ntOrders.get(), step12PartialOrdersStr(), unexpectedLayouts, 0, 2, 0));
        }
    }

    return services::Status();
}

DistributedInput<step13Local>::DistributedInput() : daal::algorithms::Input(lastStep13LocalCollectionInputId + 1) {}

/**
 * Returns an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step13Local>::get(Step13LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step13Local>::set(Step13LocalCollectionInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the DBSCAN algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step13Local>::add(Step13LocalCollectionInputId id, const NumericTablePtr & ptr)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!ptr)
    {
        return;
    }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the DBSCAN algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step13Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DataCollectionPtr dcAssignmentQueries = get(partialAssignmentQueries);
    DAAL_CHECK_EX(dcAssignmentQueries, ErrorNullInputDataCollection, ArgumentName, partialAssignmentQueriesStr());

    const size_t nQueriesBlocks = dcAssignmentQueries->size();

    for (size_t i = 0; i < nQueriesBlocks; i++)
    {
        DAAL_CHECK_EX((*dcAssignmentQueries)[i], ErrorNullNumericTable, ArgumentName, partialAssignmentQueriesStr());
        NumericTablePtr ntAssignmentQueries = NumericTable::cast((*dcAssignmentQueries)[i]);
        DAAL_CHECK_EX(ntAssignmentQueries, ErrorIncorrectElementInNumericTableCollection, ArgumentName, partialAssignmentQueriesStr());

        int unexpectedLayouts = (int)packed_mask;
        DAAL_CHECK_STATUS_VAR(checkNumericTable(ntAssignmentQueries.get(), partialAssignmentQueriesStr(), unexpectedLayouts, 0, 2, 0));
    }

    return services::Status();
}

} // namespace interface1
} // namespace dbscan
} // namespace algorithms
} // namespace daal
