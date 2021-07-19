/* file: svm_train.cpp */
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
//  Implementation of svm algorithm and types methods.
//--
*/

#include "algorithms/svm/svm_train_types.h"
#include "src/algorithms/svm/svm_model.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_SVM_MODEL_ID);

}

namespace interface2
{
services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());
    if (nClasses != 2)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nClassesStr()));
    }
    if (C <= 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, cBoundStr()));
    }
    if (accuracyThreshold <= 0 || accuracyThreshold >= 1)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, accuracyThresholdStr()));
    }
    if (tau <= 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, tauStr()));
    }
    if (maxIterations == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr()));
    }
    if (!kernel.get())
    {
        return services::Status(services::Error::create(services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, kernelFunctionStr()));
    }
    if (shrinkingStep == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, shrinkingStepStr()));
    }
    return s;
}
} // namespace interface2

namespace internal
{
services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());
    if (nClasses != 2)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nClassesStr()));
    }
    if (C <= 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, cBoundStr()));
    }
    if (accuracyThreshold <= 0 || accuracyThreshold >= 1)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, accuracyThresholdStr()));
    }
    if (tau <= 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, tauStr()));
    }
    if (maxIterations == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr()));
    }
    if (!kernel.get())
    {
        return services::Status(services::Error::create(services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, kernelFunctionStr()));
    }
    if (shrinkingStep == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, shrinkingStepStr()));
    }
    return s;
}
} // namespace internal

namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_SVM_TRAINING_RESULT_ID);
Result::Result() : classifier::training::Result() {}

/**
 * Returns the model trained with the SVM algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the SVM algorithm
 */
daal::algorithms::svm::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::svm::Model, data_management::SerializationIface>(Argument::get(id));
}

Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::training::Result::check(input, parameter, method));
    daal::algorithms::svm::ModelPtr m = get(classifier::training::model);
    if (!m->getSupportVectors()) s.add(services::Error::create(ErrorModelNotFullInitialized, services::ArgumentName, supportVectorsStr()));
    if (!m->getClassificationCoefficients())
        s.add(services::Error::create(ErrorModelNotFullInitialized, services::ArgumentName, classificationCoefficientsStr()));
    return s;
}

} // namespace interface1
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
