/* file: bf_knn_classification_model_impl.cpp */
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

#include "algorithms/kernel/k_nearest_neighbors/oneapi/bf_knn_classification_model_ucapi_impl.h"
#include "service/kernel/serialization_utils.h"
#include "service/kernel/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_K_NEAREST_NEIGHBOR_BF_MODEL_ID);

Model::Model(size_t nFeatures) : daal::algorithms::classifier::Model(), _impl(new ModelImpl(nFeatures)) {}

Model::~Model()
{
    delete _impl;
    _impl = nullptr;
}

Model::Model(size_t nFeatures, services::Status &st) : _impl(new ModelImpl(nFeatures))
{
    DAAL_CHECK_COND_ERROR(_impl, st, services::ErrorMemoryAllocationFailed);
}

services::Status Model::serializeImpl(data_management::InputDataArchive  * arch)
{
    daal::algorithms::classifier::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    return _impl->serialImpl<data_management::InputDataArchive, false>(arch);
}

services::Status Model::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    daal::algorithms::classifier::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    return _impl->serialImpl<const data_management::OutputDataArchive, true>(arch);
}

size_t Model::getNumberOfFeatures() const
{
    return _impl->getNumberOfFeatures();
}

services::Status Parameter::check() const
{
    // Inherited.
    return daal::algorithms::classifier::Parameter::check();
}

} // namespace interface1
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
