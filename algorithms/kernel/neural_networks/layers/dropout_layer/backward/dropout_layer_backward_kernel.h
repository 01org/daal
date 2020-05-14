/* file: dropout_layer_backward_kernel.h */
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
//  Implementation of the backward dropout layer
//--

#ifndef __DROPOUT_LAYER_BACKWARD_KERNEL_H__
#define __DROPOUT_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/dropout/dropout_layer.h"
#include "neural_networks/layers/dropout/dropout_layer_types.h"
#include "kernel.h"
#include "service_math.h"
#include "numeric_table.h"
#include "service_tensor.h"

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
namespace dropout
{
namespace backward
{
namespace internal
{
/**
 *  \brief Kernel for dropout calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DropoutKernel : public Kernel
{
public:
    services::Status compute(const Tensor & inputGradientTable, const Tensor & maskTable, Tensor & resultTable);

private:
    const size_t _nRowsInBlock = 5000;

    inline services::Status processBlock(const Tensor & inputGradientTable, const Tensor & maskTable, const size_t nProcessedRows,
                                         const size_t nRowsInCurrentBlock, Tensor & resultTable);
};

} // namespace internal
} // namespace backward
} // namespace dropout
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
