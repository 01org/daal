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

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::dbscan::backend {

inline array<int> fill_core_flags(const table& core_observation_indices, std::int64_t row_count) {
    array<int> arr_core_flags = array<int>::full(row_count * 1, 0);
    if (core_observation_indices.get_row_count() > 0) {
        auto index_block = row_accessor<const int>(core_observation_indices).pull();
        auto arr_core_flags_ptr = arr_core_flags.get_mutable_data();
        for (int index = 0; index < core_observation_indices.get_row_count(); index++) {
            arr_core_flags_ptr[index_block[index]] = 1;
        }
    }
    return arr_core_flags;
}

} // namespace oneapi::dal::dbscan::backend
