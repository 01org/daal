/* file: average_pooling2d_layer_forward_kernel.h */
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
//  Declaration of template function that calculate forward pooling layer results.
//--

#ifndef __AVERAGE_POOLING2D_LAYER_FORWARD_KERNEL_H__
#define __AVERAGE_POOLING2D_LAYER_FORWARD_KERNEL_H__

#include "algorithms/neural_networks/layers/pooling2d/average_pooling2d_layer_forward.h"
#include "algorithms/neural_networks/layers/pooling2d/average_pooling2d_layer_forward_types.h"
#include "algorithms/kernel/neural_networks/layers/pooling2d_layer/pooling2d_layer_internal_parameter.h"
#include "data_management/data/tensor.h"
#include "algorithms/kernel/neural_networks/layers/pooling2d_layer/forward/pooling2d_layer_forward_impl.i"
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
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for forward pooling layer results computation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class PoolingKernel : public pooling2d::forward::internal::PoolingKernel<algorithmFPType, cpu>
{
public:
    /* Computes the results of forward pooling layer */
    services::Status compute(const Tensor & dataTensor, const average_pooling2d::Parameter & parameter, Tensor & valueTensor);

    services::Status initialize(const services::Collection<size_t> & inDimsFull, const services::Collection<size_t> & outDimsFull);

    ~PoolingKernel()
    {
        if (avePoolPrim)
        {
            dnn::xDelete(avePoolPrim);
        }
    }

protected:
    typedef daal::internal::Dnn<algorithmFPType, cpu> dnn;
    typedef daal::internal::DnnLayout<algorithmFPType, cpu> xDnnLayout;

    using pooling2d::forward::internal::PoolingKernel<algorithmFPType, cpu>::defaultCompute;

    virtual void defaultInnerLoop(const pooling2d::internal::Parameter & par, DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                                  const algorithmFPType * data, algorithmFPType * valuePtr);

    virtual void defaultInnerLoop(const pooling2d::internal::Parameter & par, DAAL_INT i, DAAL_INT f, DAAL_INT k, DAAL_INT s, DAAL_INT j,
                                  const algorithmFPType * data, algorithmFPType * valuePtr, int * selectedPos)
    {
        defaultInnerLoop(par, i, f, k, s, j, data, valuePtr);
    }

    dnnPrimitive_t avePoolPrim = NULL;

    size_t * outputSize = NULL;
    TArray<size_t, cpu> outputSizePtr;

    size_t * outputStrides = NULL;
    TArray<size_t, cpu> outputStridesPtr;

    xDnnLayout ltUserOutput;
};

} // namespace internal
} // namespace forward
} // namespace average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
