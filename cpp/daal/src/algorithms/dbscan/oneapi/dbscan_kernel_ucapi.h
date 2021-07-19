/* file: dbscan_kernel_ucapi.h */
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
//  Declaration of template function that computes DBSCAN for GPU.
//--
*/

#ifndef __DBSCAN_KERNEL_UCAPI_H
#define __DBSCAN_KERNEL_UCAPI_H

#include "algorithms/dbscan/dbscan_types.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "services/internal/sycl/execution_context.h"

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
template <typename algorithmFPType>
class DBSCANBatchKernelUCAPI : public Kernel
{
public:
    services::Status compute(const daal::data_management::NumericTable * ntData, const daal::data_management::NumericTable * ntWeights,
                             daal::data_management::NumericTable * ntAssignments, daal::data_management::NumericTable * ntNClusters,
                             daal::data_management::NumericTable * ntCoreIndices, daal::data_management::NumericTable * ntCoreObservations,
                             const Parameter * par);

private:
    services::Status getCores(const services::internal::sycl::UniversalBuffer & data, uint32_t nRows, uint32_t nFeatures, int nNbrs,
                              algorithmFPType eps);
    services::Status getCoresWithWeights(const services::internal::sycl::UniversalBuffer & data, uint32_t nRows, uint32_t nFeatures,
                                         algorithmFPType nNbrs, algorithmFPType eps);
    services::Status updateQueue(uint32_t clusterId, uint32_t nRows, uint32_t nFeatures, algorithmFPType eps, uint32_t queueBegin, uint32_t queueEnd,
                                 const services::internal::sycl::UniversalBuffer & data, services::internal::sycl::UniversalBuffer & clusters);

    services::Status startNextCluster(uint32_t clusterId, uint32_t nRows, uint32_t queueEnd, services::internal::sycl::UniversalBuffer & clusters,
                                      bool & found);
    services::Status processResultsToCompute(DAAL_UINT64 resultsToCompute, daal::data_management::NumericTable * ntData,
                                             daal::data_management::NumericTable * ntCoreIndices,
                                             daal::data_management::NumericTable * ntCoreObservations);
    services::Status initializeBuffers(uint32_t nRows, daal::data_management::NumericTable * weights);
    services::Status buildProgram(services::internal::sycl::ClKernelFactoryIface & kernel_factory);
    services::Status setQueueFront(uint32_t queueEnd);
    services::Status getQueueFront(uint32_t & queueEnd);

    static constexpr uint32_t _maxSubgroupSize = 32;
    bool _useWeights;

    services::internal::sycl::UniversalBuffer _weights;
    services::internal::sycl::UniversalBuffer _queue;
    services::internal::sycl::UniversalBuffer _isCore;
    services::internal::sycl::UniversalBuffer _lastPoint;
    services::internal::sycl::UniversalBuffer _queueFront;
};

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
