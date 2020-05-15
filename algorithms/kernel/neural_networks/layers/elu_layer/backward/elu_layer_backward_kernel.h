/* file: elu_layer_backward_kernel.h */
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

//++
//  Declaration of template function that calculate relus.
//--

#ifndef __ELU_LAYER_BACKWARD_KERNEL_H__
#define __ELU_LAYER_BACKWARD_KERNEL_H__

#include "kernel.h"
#include "elu_common.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace elu
{
namespace backward
{
namespace internal
{
using namespace daal::services;
using namespace daal::data_management;

using elu::internal::BlockSizeType;
using elu::internal::ScalableTlsBuffer;

/**
 *  \brief Kernel for ELU calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class ELUKernel : public Kernel
{
private:
    ScalableTlsBuffer<algorithmFPType, cpu> _intermediateValuesTls;
    ScalableTlsBuffer<BlockSizeType, cpu> _indicesTls;

public:
    ELUKernel()
        : _intermediateValuesTls(elu::internal::getMaxBlockSize<algorithmFPType, cpu>()),
          _indicesTls(elu::internal::getMaxBlockSize<algorithmFPType, cpu>())
    {}

    Status compute(const Parameter & parameter, const Tensor & inputGradientTensor, const Tensor & auxDataTensor, const Tensor * auxValueTensor,
                   Tensor & gradientTensor);

private:
    Status computeLayoutAgnostic(const Tensor & inputGradientTensor, const Tensor & auxDataTensor, const Tensor & auxValueTensor,
                                 Tensor & gradientTensor);

    Status computeInMklLayout(const Tensor & inputGradientTensor, const Tensor & auxDataTensor, const Tensor & auxValueTensor,
                              Tensor & gradientTensor);

    void computeInRawLayout(const algorithmFPType * inputGradient, const algorithmFPType * auxData, const algorithmFPType * auxValue,
                            algorithmFPType * gradient, size_t dataSize);

    void computeBlock(const algorithmFPType * inputGradient, const algorithmFPType * auxData, const algorithmFPType * auxValue,
                      algorithmFPType * gradient, size_t blockSize);

    Status computeWithoutAuxValues(const Tensor & inputGradientTensor, const Tensor & auxDataTensor, Tensor & gradientTensor, algorithmFPType alpha);

    void computeBlockWithoutAuxValues(const algorithmFPType * inputGradient, const algorithmFPType * auxData, algorithmFPType * gradient,
                                      algorithmFPType alpha, size_t blockSize);
};

} // namespace internal
} // namespace backward
} // namespace elu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
