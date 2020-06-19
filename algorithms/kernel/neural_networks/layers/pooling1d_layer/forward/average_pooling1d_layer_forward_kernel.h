/* file: average_pooling1d_layer_forward_kernel.h */
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

#ifndef __AVERAGE_POOLING1D_LAYER_FORWARD_KERNEL_H__
#define __AVERAGE_POOLING1D_LAYER_FORWARD_KERNEL_H__

#include "algorithms/neural_networks/layers/pooling1d/average_pooling1d_layer_forward.h"
#include "algorithms/neural_networks/layers/pooling1d/average_pooling1d_layer_forward_types.h"
#include "algorithms/kernel/kernel.h"
#include "data_management/data/tensor.h"

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
namespace average_pooling1d
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for forward pooling layer results computation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class PoolingKernel : public Kernel
{
public:
    /* Computes the results of forward batch normalization layer */
    services::Status compute(const Tensor & dataTensor, const average_pooling1d::Parameter & parameter, Tensor & valueTensor);
};
} // namespace internal
} // namespace forward
} // namespace average_pooling1d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
