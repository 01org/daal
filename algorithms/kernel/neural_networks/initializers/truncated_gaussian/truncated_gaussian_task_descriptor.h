/* file: truncated_gaussian_task_descriptor.h */
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

#ifndef __TRUNCATED_GAUSSIAN_INITIALIZER_TASK_DESCRIPTOR_H__
#define __TRUNCATED_GAUSSIAN_INITIALIZER_TASK_DESCRIPTOR_H__

#include "neural_networks/initializers/truncated_gaussian/truncated_gaussian_initializer.h"
#include "neural_networks/initializers/truncated_gaussian/truncated_gaussian_initializer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace truncated_gaussian
{
namespace internal
{
template <typename algorithmFPType>
class TruncatedGaussianInitializerTaskDescriptor
{
public:
    TruncatedGaussianInitializerTaskDescriptor(Result * re, Parameter<algorithmFPType> * pa);

    engines::BatchBase * engine;
    data_management::Tensor * result;
    layers::forward::LayerIface * layer;
    double mean;
    double sigma;
    algorithmFPType a;
    algorithmFPType b;
};

} // namespace internal
} // namespace truncated_gaussian
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
