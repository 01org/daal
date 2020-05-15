/* file: gaussian_initializer_task_descriptor.cpp */
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

#include "gaussian_initializer_task_descriptor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace gaussian
{
namespace internal
{
GaussianInitializerTaskDescriptor::GaussianInitializerTaskDescriptor(Result * re, Parameter * pa)
{
    a      = pa->a;
    sigma  = pa->sigma;
    layer  = pa->layer.get();
    engine = pa->engine.get();
    result = re->get(initializers::value).get();
}

} // namespace internal
} // namespace gaussian
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
