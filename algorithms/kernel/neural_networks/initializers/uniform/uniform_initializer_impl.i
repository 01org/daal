/* file: uniform_initializer_impl.i */
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
//  Implementation of uniform algorithm
//--
*/

#ifndef __UNIFORM_INITIALIZER_IMPL_I__
#define __UNIFORM_INITIALIZER_IMPL_I__

#include "initializers_impl.i"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace uniform
{
namespace internal
{

using namespace daal::algorithms::distributions::uniform::internal;

template<typename algorithmFPType, Method method, CpuType cpu>
Status UniformKernel<algorithmFPType, method, cpu>::compute(const UniformInitializerTaskDescriptor &desc)
{
    initializers::internal::EngineImpl<cpu> engine(desc.engine);
    DAAL_CHECK_MALLOC(engine.get());

    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultSubtensor(desc.result, 0, 0, 0, desc.result->getDimensionSize(0));
    DAAL_CHECK_BLOCK_STATUS(resultSubtensor);
    algorithmFPType *resultArray = resultSubtensor.get();

    size_t size = desc.result->getSize();

    return UniformKernelDefault<algorithmFPType, cpu>::compute((algorithmFPType)desc.a, (algorithmFPType)desc.b,
                                                               *engine, size, resultArray);
}

} // internal
} // namespace uniform
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
