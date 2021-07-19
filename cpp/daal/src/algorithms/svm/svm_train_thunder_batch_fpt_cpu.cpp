/* file: svm_train_thunder_batch_fpt_cpu.cpp */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Implementation of SVM thunder training algorithm.
//--
*/

#include "src/algorithms/svm/svm_train_batch_container.h"
#include "src/algorithms/svm/svm_train_thunder_kernel.h"
#include "src/algorithms/svm/svm_train_thunder_impl.i"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace interface2
{
template class BatchContainer<DAAL_FPTYPE, thunder, DAAL_CPU>;
}
namespace internal
{
template class BatchContainer<DAAL_FPTYPE, thunder, DAAL_CPU>;
template struct DAAL_EXPORT SVMTrainImpl<thunder, DAAL_FPTYPE, DAAL_CPU>;

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
