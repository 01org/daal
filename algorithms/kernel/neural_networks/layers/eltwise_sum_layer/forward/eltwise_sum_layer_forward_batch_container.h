/* file: eltwise_sum_layer_forward_batch_container.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of element-wise sum calculation algorithm container.
//--
*/

#ifndef __ELTWISE_SUM_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __ELTWISE_SUM_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "algorithms/neural_networks/layers/eltwise_sum/eltwise_sum_layer.h"
#include "algorithms/kernel/neural_networks/layers/eltwise_sum_layer/forward/eltwise_sum_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace eltwise_sum
{
namespace forward
{
namespace interface1
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::EltwiseSumKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    using namespace daal::internal;

    eltwise_sum::forward::Input * input   = static_cast<eltwise_sum::forward::Input *>(_in);
    eltwise_sum::forward::Result * result = static_cast<eltwise_sum::forward::Result *>(_res);

    const size_t nInputs = input->get(layers::forward::inputLayerData)->size();

    TArray<Tensor *, cpu> inputBlock(nInputs);
    Tensor ** inputTensors = inputBlock.get();
    if (!inputTensors)
    {
        return services::Status(services::ErrorMemoryAllocationFailed);
    }

    for (size_t i = 0; i < nInputs; i++)
    {
        inputTensors[i] = input->get(layers::forward::inputLayerData, i).get();
    }

    Tensor * coefficients               = input->get(eltwise_sum::forward::coefficients).get();
    Tensor * value                      = result->get(layers::forward::value).get();
    Tensor * auxCoefficients            = result->get(eltwise_sum::auxCoefficients).get();
    NumericTable * numberOfCoefficients = result->get(eltwise_sum::auxNumberOfCoefficients).get();

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::EltwiseSumKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inputTensors, value, coefficients,
                       auxCoefficients, numberOfCoefficients, nInputs);
}
} // namespace interface1
} // namespace forward
} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
