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
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_builder.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"

#include <vector>

using namespace std;

namespace dal = oneapi::dal;

int main(int argc, char** argv) {
    //const auto filename = get_data_path("graph.csv");

    //const dal::preview::graph_csv_data_source ds(filename);
    //const dal::preview::load_graph::descriptor<> d;
    //const auto my_graph = dal::preview::load_graph::load(d, ds);
    vector<int64_t> rows = { 0, 1, 3, 4 };
    vector<int32_t> cols = { 1, 0, 4, 1 };
    dal::preview::detail::directed_adjacency_vector_graph_builder<> builder(
        (const int64_t*)rows.data(),
        (const int32_t*)cols.data(),
        3,
        4);
    const auto& my_graph = builder.get_graph();
    //dal::preview::directed_adjacency_vector_graph<> my_graph;

    //std::cout << "Graph is read from file: " << filename << std::endl;
    std::cout << "Number of vertices: " << dal::preview::get_vertex_count(my_graph) << std::endl;
    std::cout << "Number of edges: " << dal::preview::get_edge_count(my_graph) << std::endl;

    using my_graph_type = dal::preview::directed_adjacency_vector_graph<>;
    dal::preview::vertex_outward_edge_size_type<my_graph_type> vertex_id = 0;
    std::cout << "Degree of " << vertex_id << ": "
              << dal::preview::get_vertex_outward_degree(my_graph, vertex_id) << std::endl;

    for (dal::preview::vertex_outward_edge_size_type<my_graph_type> j = 0;
         j < dal::preview::get_vertex_count(my_graph);
         ++j) {
        std::cout << "Neighbors of " << j << ": ";
        const auto neigh = dal::preview::get_vertex_outward_neighbors(my_graph, j);
        for (auto i = neigh.first; i != neigh.second; ++i) {
            std::cout << *i << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
