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

#include <iostream>

#include "example_util/output_helpers_graph.hpp"
#include "example_util/utils.hpp"
#include "oneapi/dal/algo/jaccard.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/table/common.hpp"

namespace dal = oneapi::dal;

using namespace dal;

int main(int argc, char **argv) {
    const auto filename = get_data_path("graph.csv");

    // read the graph
    using graph_t = preview::undirected_adjacency_vector_graph<>;
    const auto graph = read<graph_t>(csv::data_source{ filename });

    // set blocks ranges
    const std::int64_t row_range_begin = 0;
    const std::int64_t row_range_end = 2;
    const std::int64_t column_range_begin = 0;
    const std::int64_t column_range_end = 3;

    // set algorithm parameters
    const auto jaccard_desc =
        preview::jaccard::descriptor<>().set_block({ row_range_begin, row_range_end },
                                                   { column_range_begin, column_range_end });

    // create caching builder for jaccard result
    preview::jaccard::caching_builder builder;

    // compute Jaccard similarity coefficients
    const auto result_vertex_similarity = preview::vertex_similarity(jaccard_desc, graph, builder);

    // extract the result
    const auto jaccard_coeffs = result_vertex_similarity.get_coeffs();
    const auto vertex_pairs = result_vertex_similarity.get_vertex_pairs();
    const std::int64_t nonzero_coeff_count = result_vertex_similarity.get_nonzero_coeff_count();

    std::cout << "The number of nonzero Jaccard coeffs in the block: " << nonzero_coeff_count
              << std::endl;

    print_vertex_similarity_result(result_vertex_similarity);
}
