/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_service_kernels_def_dpc.hpp"

namespace oneapi::dal::decision_forest::backend {

INSTANTIATE(double, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(double, std::uint32_t, std::int32_t, task::regression);

} // namespace oneapi::dal::decision_forest::backend
