/* file: svm_predict_batch_container.h */
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
//  Implementation of SVM prediction algorithm container.
//--
*/

#include "algorithms/svm/svm_predict.h"
#include "algorithms/kernel/svm/svm_predict_kernel.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace interface2
{
/**
*  \brief Initialize list of SVM kernels with implementations for supported architectures
*/
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu)
    {
        __DAAL_INITIALIZE_KERNELS(internal::SVMPredictImpl, method, algorithmFPType);
    }
    else
    {
        __DAAL_INITIALIZE_KERNELS_SYCL(internal::SVMPredictImplOneAPI, method, algorithmFPType);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::prediction::Input * input   = static_cast<classifier::prediction::Input *>(_in);
    classifier::prediction::Result * result = static_cast<classifier::prediction::Result *>(_res);

    NumericTablePtr a           = input->get(classifier::prediction::data);
    daal::algorithms::Model * m = static_cast<daal::algorithms::Model *>(input->get(classifier::prediction::model).get());
    NumericTablePtr r           = result->get(classifier::prediction::prediction);

    daal::algorithms::Parameter * par      = _par;
    daal::services::Environment::env & env = *_env;

    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu)
    {
        __DAAL_CALL_KERNEL(env, internal::SVMPredictImpl, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m, *r, par);
    }
    else
    {
        __DAAL_INITIALIZE_KERNELS_SYCL(env, internal::SVMPredictImplOneAPI, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, m, *r, par);
    }
}
} // namespace interface2
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal
