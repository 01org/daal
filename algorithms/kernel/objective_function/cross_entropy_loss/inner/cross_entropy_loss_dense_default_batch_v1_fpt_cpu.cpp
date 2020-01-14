/* file: cross_entropy_loss_dense_default_batch_v1_fpt_cpu.cpp */
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
//  Implementation of cross_entropy_loss calculation functions.
//--

#include "cross_entropy_loss_dense_default_batch_kernel_v1.h"
#include "cross_entropy_loss_dense_default_batch_v1_impl.i"
#include "cross_entropy_loss_dense_default_batch_container_v1.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace cross_entropy_loss
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}

namespace internal
{
template class I1CrossEntropyLossKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}

} // namespace cross_entropy_loss

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
