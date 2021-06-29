/* file: naivebayes_train_csr_fast_online_fpt_cpu.cpp */
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
//  Implementation of multinomial naive bayes algorithm.
//--
*/

#include "src/algorithms/naivebayes/naivebayes_train_kernel.h"
#include "src/algorithms/naivebayes/naivebayes_train_impl.i"
#include "src/algorithms/naivebayes/naivebayes_train_container.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
namespace interface2
{
template class OnlineContainer<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
namespace internal
{
template class NaiveBayesOnlineTrainKernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
} // namespace internal
} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
