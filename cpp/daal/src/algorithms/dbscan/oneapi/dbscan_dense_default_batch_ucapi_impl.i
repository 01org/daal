/* file: dbscan_dense_default_batch_ucapi_impl.i */
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
//++
//  Implementation of default method for DBSCAN algorithm on GPU.
//--
*/

#include "src/algorithms/dbscan/oneapi/dbscan_kernel_ucapi.h"
#include "src/algorithms/dbscan/oneapi/cl_kernels/dbscan_cl_kernels.cl"
#include "src/services/service_data_utils.h"
#include "src/externals/service_ittnotify.h"

using namespace daal::services;
using namespace daal::services::internal::sycl;
using namespace daal::data_management;

constexpr size_t maxInt32AsSizeT   = static_cast<size_t>(daal::services::internal::MaxVal<int32_t>::get());
constexpr size_t maxInt32AsUint32T = static_cast<uint32_t>(daal::services::internal::MaxVal<int32_t>::get());

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::initializeBuffers(uint32_t nRows, NumericTable * weights)
{
    Status s;
    auto & context = Environment::getInstance()->getDefaultExecutionContext();
    _queue         = context.allocate(TypeIds::id<int>(), nRows, s);
    DAAL_CHECK_STATUS_VAR(s);
    _isCore = context.allocate(TypeIds::id<int>(), nRows, s);
    DAAL_CHECK_STATUS_VAR(s);
    context.fill(_isCore, 0, s);
    DAAL_CHECK_STATUS_VAR(s);
    _lastPoint = context.allocate(TypeIds::id<int>(), 1, s);
    DAAL_CHECK_STATUS_VAR(s);
    context.fill(_lastPoint, 0, s);
    DAAL_CHECK_STATUS_VAR(s);
    _queueFront = context.allocate(TypeIds::id<int>(), 1, s);
    DAAL_CHECK_STATUS_VAR(s);
    _useWeights = weights != nullptr;
    if (_useWeights)
    {
        BlockDescriptor<algorithmFPType> weightRows;
        DAAL_CHECK_STATUS_VAR(weights->getBlockOfRows(0, nRows, readOnly, weightRows));
        _weights = UniversalBuffer(weightRows.getBuffer());
    }
    else
    {
        // OpenCL needs it
        _weights = context.allocate(TypeIds::id<algorithmFPType>(), 1, s);
        DAAL_CHECK_STATUS_VAR(s);
    }
    return s;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::processResultsToCompute(DAAL_UINT64 resultsToCompute, NumericTable * ntData,
                                                                        NumericTable * ntCoreIndices, NumericTable * ntCoreObservations)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.processResultsToCompute);
    Status st;

    const uint32_t nRows     = ntData->getNumberOfRows();
    const uint32_t nFeatures = ntData->getNumberOfColumns();

    DAAL_ASSERT_UNIVERSAL_BUFFER(_isCore, int, nRows);
    auto isCoreHost = _isCore.template get<int>().toHost(ReadWriteMode::readOnly, st);
    DAAL_CHECK_STATUS_VAR(st);
    auto isCore = isCoreHost.get();

    uint32_t nCoreObservations = 0;
    for (uint32_t i = 0; i < nRows; i++)
    {
        if (!isCore[i])
        {
            continue;
        }
        nCoreObservations++;
    }

    if (nCoreObservations == 0)
    {
        return Status();
    }

    if (resultsToCompute & computeCoreIndices)
    {
        DAAL_CHECK_STATUS_VAR(ntCoreIndices->resize(nCoreObservations));
        BlockDescriptor<int> indexRows;
        DAAL_CHECK_STATUS_VAR(ntCoreIndices->getBlockOfRows(0, nCoreObservations, writeOnly, indexRows));
        auto coreIndices = indexRows.getBlockPtr();
        if (!coreIndices)
        {
            return Status(ErrorNullPtr);
        }

        uint32_t pos = 0;
        for (uint32_t i = 0; i < nRows; i++)
        {
            if (!isCore[i])
            {
                continue;
            }
            coreIndices[pos] = i;
            pos++;
        }
    }

    if (resultsToCompute & computeCoreObservations)
    {
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->resize(nCoreObservations));
        BlockDescriptor<algorithmFPType> coreObservationsRows;
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->getBlockOfRows(0, nCoreObservations, writeOnly, coreObservationsRows));
        auto coreObservationsPtr = coreObservationsRows.getBuffer().toHost(ReadWriteMode::writeOnly, st);
        DAAL_CHECK_STATUS_VAR(st);
        BlockDescriptor<algorithmFPType> dataRows;
        DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(0, nRows, readOnly, dataRows));
        auto dataPtr = dataRows.getBuffer().toHost(ReadWriteMode::readOnly, st);
        DAAL_CHECK_STATUS_VAR(st);

        uint32_t pos = 0;
        for (uint32_t i = 0; i < nRows; i++)
        {
            if (!isCore[i])
            {
                continue;
            }
            for (uint32_t j = 0; j < nFeatures; j++) coreObservationsPtr.get()[pos * nFeatures + j] = dataPtr.get()[i * nFeatures + j];
            pos++;
        }
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->releaseBlockOfRows(coreObservationsRows));
        DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(dataRows));
    }

    return st;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::compute(const NumericTable * x, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                                        NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                                        const Parameter * par)
{
    Status s;
    auto & context                = Environment::getInstance()->getDefaultExecutionContext();
    const uint32_t minkowskiPower = 2;
    algorithmFPType epsP          = 1.0;
    for (uint32_t i = 0; i < minkowskiPower; i++) epsP *= par->epsilon;
    DAAL_CHECK((par->minObservations > algorithmFPType(0)) && (par->minObservations < algorithmFPType(maxInt32AsSizeT)),
               services::ErrorIncorrectParameter);

    NumericTable * const ntData = const_cast<NumericTable *>(x);
    NumericTable * const ntW    = const_cast<NumericTable *>(ntWeights);

    const size_t nDataRowsAsSizeT    = ntData->getNumberOfRows();
    const size_t nDataColumnsAsSizeT = ntData->getNumberOfColumns();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nDataRowsAsSizeT, nDataColumnsAsSizeT);

    DAAL_CHECK(nDataRowsAsSizeT <= maxInt32AsSizeT, services::ErrorIncorrectNumberOfRowsInInputNumericTable);
    DAAL_CHECK(nDataColumnsAsSizeT <= maxInt32AsSizeT, services::ErrorIncorrectNumberOfColumnsInInputNumericTable);

    if (ntW)
    {
        const size_t nWeightRowsAsSizeT    = ntW->getNumberOfRows();
        const size_t nWeightColumnsAsSizeT = ntW->getNumberOfColumns();
        DAAL_CHECK(nWeightRowsAsSizeT == nDataRowsAsSizeT, services::ErrorIncorrectNumberOfRowsInInputNumericTable);
        DAAL_CHECK(nWeightColumnsAsSizeT == 1, services::ErrorIncorrectNumberOfColumnsInInputNumericTable);
    }

    const uint32_t nRows     = static_cast<uint32_t>(nDataRowsAsSizeT);
    const uint32_t nFeatures = static_cast<uint32_t>(nDataColumnsAsSizeT);

    BlockDescriptor<algorithmFPType> dataRows;
    DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(0, nRows, readOnly, dataRows));
    auto data = dataRows.getBuffer();

    BlockDescriptor<int> assignRows;
    DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(0, nRows, writeOnly, assignRows));
    auto assignBuffer = assignRows.getBuffer();

    UniversalBuffer assignments = assignBuffer;
    context.fill(assignments, noise, s);
    DAAL_CHECK_STATUS_VAR(s);

    DAAL_CHECK_STATUS_VAR(initializeBuffers(nRows, ntW));

    uint32_t nClusters  = 0;
    uint32_t queueBegin = 0;
    uint32_t queueEnd   = 0;

    if (_useWeights)
    {
        DAAL_CHECK_STATUS_VAR(getCoresWithWeights(data, nRows, nFeatures, par->minObservations, epsP));
    }
    else
    {
        DAAL_CHECK_STATUS_VAR(getCores(data, nRows, nFeatures, par->minObservations, epsP));
    }

    bool foundCluster = false;
    DAAL_CHECK_STATUS_VAR(startNextCluster(nClusters, nRows, queueEnd, assignments, foundCluster));
    while (foundCluster)
    {
        ++nClusters;
        ++queueEnd;
        DAAL_CHECK_STATUS_VAR(setQueueFront(queueEnd));
        while (queueBegin < queueEnd)
        {
            updateQueue(nClusters - 1, nRows, nFeatures, epsP, queueBegin, queueEnd, data, assignments);
            queueBegin = queueEnd;
            DAAL_CHECK_STATUS_VAR(getQueueFront(queueEnd));
        }
        DAAL_CHECK_STATUS_VAR(startNextCluster(nClusters, nRows, queueEnd, assignments, foundCluster));
    }
    DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(dataRows));

    BlockDescriptor<int> nClustersRows;
    DAAL_CHECK_STATUS_VAR(ntNClusters->getBlockOfRows(0, 1, writeOnly, nClustersRows));
    auto nClusterHostBuffer = nClustersRows.getBuffer().toHost(ReadWriteMode::writeOnly, s);
    DAAL_CHECK_STATUS_VAR(s);
    *nClusterHostBuffer.get() = nClusters;

    if (par->resultsToCompute & (computeCoreIndices | computeCoreObservations))
    {
        DAAL_CHECK_STATUS_VAR(processResultsToCompute(par->resultsToCompute, ntData, ntCoreIndices, ntCoreObservations));
    }
    return s;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::startNextCluster(uint32_t clusterId, uint32_t nRows, uint32_t queueEnd,
                                                                           UniversalBuffer & clusters, bool & found)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.startNextCluster);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("startNextCluster", st);
    DAAL_CHECK_STATUS_VAR(st);

    int last;
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(_lastPoint, int, 1);
        const auto lastPointHostBuffer = _lastPoint.template get<int>().toHost(ReadWriteMode::readOnly, st);
        DAAL_CHECK_STATUS_VAR(st);
        last = *lastPointHostBuffer.get();
    }

    DAAL_ASSERT_UNIVERSAL_BUFFER(_isCore, int, nRows);
    DAAL_ASSERT_UNIVERSAL_BUFFER(clusters, int, nRows);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_queue, int, nRows);

    KernelArguments args(7, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, static_cast<int32_t>(clusterId));
    args.set(1, static_cast<int32_t>(nRows));
    args.set(2, static_cast<int32_t>(queueEnd));
    args.set(3, _isCore, AccessModeIds::read);
    args.set(4, clusters, AccessModeIds::write);
    args.set(5, _lastPoint, AccessModeIds::write);
    args.set(6, _queue, AccessModeIds::write);

    KernelRange localRange(1, _maxSubgroupSize);
    KernelRange globalRange(1, _maxSubgroupSize);

    KernelNDRange range(2);
    range.global(globalRange, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(localRange, st);
    DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, st);
    DAAL_CHECK_STATUS_VAR(st);
    int newLast;
    {
        const auto lastPointHostBuffer = _lastPoint.template get<int>().toHost(ReadWriteMode::readOnly, st);
        DAAL_CHECK_STATUS_VAR(st);
        newLast = *lastPointHostBuffer.get();
        found   = newLast > last;
    }
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::getCores(const UniversalBuffer & data, uint32_t nRows, uint32_t nFeatures, int nNbrs,
                                                                   algorithmFPType eps)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.getCores);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("computeCores", st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_ASSERT_UNIVERSAL_BUFFER(data, algorithmFPType, nRows * nFeatures);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_weights, algorithmFPType, _useWeights ? nRows : 1);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_isCore, int, nRows);

    KernelArguments args(6, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, static_cast<int32_t>(nRows));
    args.set(1, static_cast<int32_t>(nFeatures));
    args.set(2, nNbrs);
    args.set(3, eps);
    args.set(4, data, AccessModeIds::read);
    args.set(5, _isCore, AccessModeIds::write);

    const uint32_t rangeWidth = (nFeatures < _maxSubgroupSize) ? nFeatures : _maxSubgroupSize;
    KernelRange localRange(1, rangeWidth);
    KernelRange globalRange(nRows, rangeWidth);

    KernelNDRange range(2);
    range.global(globalRange, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(localRange, st);
    DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, st);
    DAAL_CHECK_STATUS_VAR(st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::getCoresWithWeights(const UniversalBuffer & data, uint32_t nRows, uint32_t nFeatures,
                                                                              algorithmFPType nNbrs, algorithmFPType eps)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.getCores);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("computeCoresWithWeights", st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_ASSERT_UNIVERSAL_BUFFER(data, algorithmFPType, nRows * nFeatures);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_weights, algorithmFPType, _useWeights ? nRows : 1);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_isCore, int, nRows);

    KernelArguments args(8, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, static_cast<int32_t>(nRows));
    args.set(1, static_cast<int32_t>(nFeatures));
    args.set(2, nNbrs);
    args.set(3, eps);
    args.set(4, static_cast<int32_t>(_useWeights));
    args.set(5, data, AccessModeIds::read);
    args.set(6, _weights, AccessModeIds::read);
    args.set(7, _isCore, AccessModeIds::write);

    uint32_t rangeWidth = nFeatures < _maxSubgroupSize ? nFeatures : _maxSubgroupSize;
    KernelRange localRange(1, rangeWidth);
    KernelRange globalRange(nRows, rangeWidth);

    KernelNDRange range(2);
    range.global(globalRange, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(localRange, st);
    DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, st);
    DAAL_CHECK_STATUS_VAR(st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::updateQueue(uint32_t clusterId, uint32_t nRows, uint32_t nFeatures, algorithmFPType eps,
                                                                      uint32_t queueBegin, uint32_t queueEnd, const UniversalBuffer & data,
                                                                      UniversalBuffer & clusters)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateQueue);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("updateQueue", st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_ASSERT_UNIVERSAL_BUFFER(data, algorithmFPType, nRows * nFeatures);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_isCore, int, nRows);
    DAAL_ASSERT_UNIVERSAL_BUFFER(clusters, int, nRows);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_queue, int, nRows);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_queueFront, int, 1);

    KernelArguments args(11, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, static_cast<int32_t>(clusterId));
    args.set(1, static_cast<int32_t>(nRows));
    args.set(2, static_cast<int32_t>(nFeatures));
    args.set(3, eps);
    args.set(4, static_cast<int32_t>(queueBegin));
    args.set(5, static_cast<int32_t>(queueEnd));
    args.set(6, data, AccessModeIds::read);
    args.set(7, _isCore, AccessModeIds::read);
    args.set(8, clusters, AccessModeIds::write);
    args.set(9, _queue, AccessModeIds::write);
    args.set(10, _queueFront, AccessModeIds::write);

    uint32_t rangeWidth = nFeatures < _maxSubgroupSize ? nFeatures : _maxSubgroupSize;
    KernelRange localRange(1, rangeWidth);
    KernelRange globalRange(nRows, rangeWidth);

    KernelNDRange range(2);
    range.global(globalRange, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(localRange, st);
    DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, st);
    DAAL_CHECK_STATUS_VAR(st);
    return st;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::buildProgram(ClKernelFactoryIface & kernel_factory)
{
    Status st;
    const auto fptypeName   = services::internal::sycl::getKeyFPType<algorithmFPType>();
    const auto buildOptions = fptypeName;

    services::String cachekey("__daal_algorithms_dbscan_block_");
    cachekey.add(fptypeName);
    cachekey.add(buildOptions);
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), dbscanClKernels, buildOptions.c_str(), st);
        DAAL_CHECK_STATUS_VAR(st);
    }
    return st;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::setQueueFront(uint32_t queueEnd)
{
    Status st;
    DAAL_ASSERT_UNIVERSAL_BUFFER(_queueFront, int, 1);
    const auto val = _queueFront.template get<int>().toHost(ReadWriteMode::readWrite, st);
    DAAL_CHECK_STATUS_VAR(st);
    DAAL_ASSERT(queueEnd <= maxInt32AsUint32T);
    *val.get() = static_cast<int>(queueEnd);
    return st;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::getQueueFront(uint32_t & queueEnd)
{
    Status st;
    DAAL_ASSERT_UNIVERSAL_BUFFER(_queueFront, int, 1);
    const auto val = _queueFront.template get<int>().toHost(ReadWriteMode::readOnly, st);
    DAAL_CHECK_STATUS_VAR(st);
    DAAL_ASSERT(*val.get() >= 0);
    queueEnd = static_cast<uint32_t>(*val.get());
    return st;
}

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal
