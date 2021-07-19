/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/io/csv/detail/read_graph_kernel_impl.hpp"
#include "oneapi/dal/io/csv/backend/cpu/read_graph.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::read_graph::detail {

template <>
ONEDAL_EXPORT std::int64_t compute_prefix_sum(const std::int32_t *degrees,
                                              std::int64_t degrees_count,
                                              std::int64_t *edge_offsets) {
    return dal::backend::dispatch_by_cpu(
        dal::backend::context_cpu{ dal::detail::host_policy::get_default() },
        [&](auto cpu) {
            return backend::compute_prefix_sum<decltype(cpu)>(degrees, degrees_count, edge_offsets);
        });
}

template <>
ONEDAL_EXPORT void fill_filtered_neighs(const std::int64_t *unfiltered_offsets,
                                        const std::int32_t *unfiltered_neighs,
                                        const std::int32_t *filtered_degrees,
                                        const std::int64_t *filtered_offsets,
                                        std::int32_t *filtered_neighs,
                                        std::int64_t vertex_count) {
    dal::backend::dispatch_by_cpu(
        dal::backend::context_cpu{ dal::detail::host_policy::get_default() },
        [&](auto cpu) {
            return backend::fill_filtered_neighs<decltype(cpu)>(unfiltered_offsets,
                                                                unfiltered_neighs,
                                                                filtered_degrees,
                                                                filtered_offsets,
                                                                filtered_neighs,
                                                                vertex_count);
        });
}

template <>
ONEDAL_EXPORT void filter_neighbors_and_fill_new_degrees(std::int32_t *unfiltered_neighs,
                                                         std::int64_t *unfiltered_offsets,
                                                         std::int32_t *new_degrees,
                                                         std::int64_t vertex_count) {
    dal::backend::dispatch_by_cpu(
        dal::backend::context_cpu{ dal::detail::host_policy::get_default() },
        [&](auto cpu) {
            return backend::filter_neighbors_and_fill_new_degrees<decltype(cpu)>(unfiltered_neighs,
                                                                                 unfiltered_offsets,
                                                                                 new_degrees,
                                                                                 vertex_count);
        });
}

} // namespace oneapi::dal::preview::read_graph::detail
