/* file: common.hpp */
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

/// @file
/// Io related common data type aliases

#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/io/csv/detail/graph_service.hpp"

namespace oneapi::dal::preview {
/// Type of graph representation as an edge list
/// @tparam IndexType Type of the graph vertex indicies
template <typename IndexType = std::int32_t>
using edge_list =
    detail::edge_list_container<std::pair<IndexType, IndexType>, std::allocator<char>>;

/// Type of graph with edge weights representation as an edge list
/// @tparam IndexType Type of the graph vertex indicies
template <typename IndexType = std::int32_t, typename WeightType = std::int32_t>
using weighted_edge_list =
    detail::edge_list_container<std::tuple<IndexType, IndexType, WeightType>, std::allocator<char>>;

} // namespace oneapi::dal::preview
