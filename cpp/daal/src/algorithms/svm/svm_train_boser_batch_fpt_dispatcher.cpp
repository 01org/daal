/* file: svm_train_boser_batch_fpt_dispatcher.cpp */
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
//  Implementation of SVM training algorithm container.
//--
*/

#include "src/algorithms/svm/svm_train_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(svm::training::BatchContainer, batch, DAAL_FPTYPE, svm::training::boser)
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(svm::training::internal::BatchContainer, batch, DAAL_FPTYPE, svm::training::boser)
} // namespace algorithms
} // namespace daal
