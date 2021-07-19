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

#include "oneapi/dal/io/csv/backend/cpu/read_graph.hpp"

namespace oneapi::dal::preview::read_graph::backend {

template std::int64_t get_vertex_count_from_edge_list<__CPU_TAG__>(
    const edge_list<std::int32_t> &edges);

template std::int64_t compute_prefix_sum<__CPU_TAG__>(const std::int32_t *degrees,
                                                      std::int64_t degrees_count,
                                                      std::int64_t *edge_offsets);

template void fill_filtered_neighs<__CPU_TAG__>(const std::int64_t *unfiltered_offsets,
                                                const std::int32_t *unfiltered_neighs,
                                                const std::int32_t *filtered_degrees,
                                                const std::int64_t *filtered_offsets,
                                                std::int32_t *filtered_neighs,
                                                std::int64_t vertex_count);

template void filter_neighbors_and_fill_new_degrees<__CPU_TAG__>(std::int32_t *unfiltered_neighs,
                                                                 std::int64_t *unfiltered_offsets,
                                                                 std::int32_t *new_degrees,
                                                                 std::int64_t vertex_count);

} // namespace oneapi::dal::preview::read_graph::backend
