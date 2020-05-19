/* file: average_pooling2d_layer_backward_kernel.h */
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
//  Declaration of template function that calculate backward pooling layer relults.
//--

#ifndef __AVERAGE_POOLING2D_LAYER_BACKWARD_KERNEL_H__
#define __AVERAGE_POOLING2D_LAYER_BACKWARD_KERNEL_H__

#include "algorithms/neural_networks/layers/pooling2d/average_pooling2d_layer_backward.h"
#include "algorithms/neural_networks/layers/pooling2d/average_pooling2d_layer_backward_types.h"
#include "algorithms/kernel/neural_networks/layers/pooling2d_layer/pooling2d_layer_internal_parameter.h"
#include "data_management/data/tensor.h"
#include "algorithms/kernel/neural_networks/layers/pooling2d_layer/backward/pooling2d_layer_backward_impl.i"
#include "externals/service_dnn.h"
#include "algorithms/kernel/service_dnn_internal.h"
#include "algorithms/kernel/neural_networks/layers/layers_threading.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling2d
{
namespace backward
{
namespace internal
{
/**
 *  \brief Kernel for backward pooling layer results computation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class PoolingKernel : public pooling2d::backward::internal::PoolingKernel<algorithmFPType, cpu>
{
public:
    services::Status compute(const Tensor & inputGradTensor, const pooling2d::Parameter & parameter, Tensor & gradTensor, const Tensor * dataTensor);

    services::Status initialize(const services::Collection<size_t> & inDimsFull, const services::Collection<size_t> & outDimsFull);

    ~PoolingKernel()
    {
        if (avePoolPrim)
        {
            dnn::xDelete(avePoolPrim);
        }
    }

protected:
    using pooling2d::backward::internal::PoolingKernel<algorithmFPType, cpu>::defaultCompute;

    virtual void defaultInnerLoop(const pooling2d::internal::Parameter & par, DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s,
                                  const algorithmFPType * inputGradPtr, const int * selectedPosPtr, algorithmFPType * grad);

private:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;
    typedef daal::internal::DnnLayout<algorithmFPType, cpu> xDnnLayout;

    dnnPrimitive_t avePoolPrim = NULL;

    TArray<size_t, cpu> inputSizePtr;
    size_t * inputSize = NULL;

    TArray<size_t, cpu> inputStridesPtr;
    size_t * inputStrides = NULL;

    TArray<size_t, cpu> outputSizePtr;
    size_t * outputSize = NULL;

    TArray<size_t, cpu> outputStridesPtr;
    size_t * outputStrides = NULL;

    xDnnLayout ltUserInput;
    xDnnLayout ltUserOutput;
};

} // namespace internal
} // namespace backward
} // namespace average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
