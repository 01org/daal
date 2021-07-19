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

#pragma once

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::backend {

template <typename Float>
inline constexpr Float exp_low_threshold() {
    static_assert(detail::is_floating_point<Float>());

    // minimal double value ~ 2.3e-308, exp(-650.0) ~ 5.1e-283
    constexpr double exp_low_double_threshold = -650.0;
    // minimal float value ~ 1.2e-38, exp(-75.0) ~ 2.6e-33
    constexpr float exp_low_float_threshold = -75.0;

    return std::is_same_v<Float, double> ? exp_low_double_threshold : exp_low_float_threshold;
}

} // namespace oneapi::dal::backend
