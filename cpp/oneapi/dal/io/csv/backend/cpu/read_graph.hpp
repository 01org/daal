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

#pragma once

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/io/csv/detail/common.hpp"

namespace oneapi::dal::preview::read_graph::backend {

template <typename Cpu>
std::int64_t get_vertex_count_from_edge_list(const edge_list<std::int32_t> &edges) {
    std::int32_t max_id = edges[0].first;
    for (std::int64_t i = 0; i < edges.size(); i++) {
        std::int32_t edge_max = std::max(edges[i].first, edges[i].second);
        max_id = std::max(max_id, edge_max);
    }
    const std::int64_t vertex_count = max_id + 1;
    return vertex_count;
}

template <typename Cpu>
std::int64_t compute_prefix_sum(const std::int32_t *degrees,
                                std::int64_t degrees_count,
                                std::int64_t *edge_offsets) {
    std::int64_t total_sum_degrees = 0;
    edge_offsets[0] = total_sum_degrees;
    for (std::int64_t i = 0; i < degrees_count; ++i) {
        total_sum_degrees += degrees[i];
        edge_offsets[i + 1] = total_sum_degrees;
    }
    return total_sum_degrees;
}

template <typename Cpu>
void fill_filtered_neighs(const std::int64_t *unfiltered_offsets,
                          const std::int32_t *unfiltered_neighs,
                          const std::int32_t *filtered_degrees,
                          const std::int64_t *filtered_offsets,
                          std::int32_t *filtered_neighs,
                          std::int64_t vertex_count) {
    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int32_t u) {
        auto u_neighs = filtered_neighs + filtered_offsets[u];
        auto u_neighs_unf = unfiltered_neighs + unfiltered_offsets[u];
        for (std::int32_t i = 0; i < filtered_degrees[u]; i++) {
            u_neighs[i] = u_neighs_unf[i];
        }
    });
}

template <typename Cpu>
void filter_neighbors_and_fill_new_degrees(std::int32_t *unfiltered_neighs,
                                           std::int64_t *unfiltered_offsets,
                                           std::int32_t *new_degrees,
                                           std::int64_t vertex_count) {
    //removing self-loops,  multiple edges from graph, and make neighbors in CSR sorted
    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int32_t u) {
        auto start_p = unfiltered_neighs + unfiltered_offsets[u];
        auto end_p = unfiltered_neighs + unfiltered_offsets[u + 1];
        dal::detail::parallel_sort(start_p, end_p);
        auto neighs_u_new_end = std::unique(start_p, end_p);
        neighs_u_new_end = std::remove(start_p, neighs_u_new_end, u);
        new_degrees[u] = (std::int32_t)std::distance(start_p, neighs_u_new_end);
    });
}

} // namespace oneapi::dal::preview::read_graph::backend
