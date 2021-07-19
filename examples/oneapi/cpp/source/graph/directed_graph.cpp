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

#include "example_util/utils.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/directed_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/csv.hpp"

using namespace std;

namespace dal = oneapi::dal;

int main(int argc, char** argv) {
    const auto filename = get_data_path("weighted_edge_list.csv");

    using vertex_type = int32_t;
    using weight_type = double;
    using graph_t = dal::preview::directed_adjacency_vector_graph<vertex_type, weight_type>;

    auto graph = dal::read<graph_t>(dal::csv::data_source{ filename },
                                    dal::preview::read_mode::weighted_edge_list);

    std::cout << "Number of vertices: " << dal::preview::get_vertex_count(graph) << std::endl;
    std::cout << "Number of edges: " << dal::preview::get_edge_count(graph) << std::endl;

    dal::preview::vertex_outward_edge_size_type<graph_t> vertex_id = 0;
    std::cout << "Degree of " << vertex_id << ": "
              << dal::preview::get_vertex_outward_degree(graph, vertex_id) << std::endl;

    for (dal::preview::vertex_outward_edge_size_type<graph_t> j = 0;
         j < dal::preview::get_vertex_count(graph);
         ++j) {
        std::cout << "Neighbors of " << j << ": ";
        const auto neigh = dal::preview::get_vertex_outward_neighbors(graph, j);
        for (auto i = neigh.first; i != neigh.second; ++i) {
            std::cout << *i << "-" << dal::preview::get_edge_value(graph, j, *i) << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
